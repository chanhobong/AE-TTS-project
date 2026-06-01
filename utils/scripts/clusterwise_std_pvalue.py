#!/usr/bin/env python3
"""
Cluster-wise comparison of within-patient variability (std) between Normal vs TTS.

Goal
----
For each latent source (plain_ae / monai_ae) and each slice-cluster id k:
  - For each patient:
      take slices whose cluster_ids == k (and apply mask if present)
      compute std_vector_k = std(embeddings_k, axis=0, ddof=0)  # (D,)
      reduce to scalar:
          mean_std_k = mean(std_vector_k)
          l2_std_k   = ||std_vector_k||_2
  - Compare distributions of these patient-level scalars between Normal and TTS.
  - Report p-values (Mann–Whitney U by default) and BH-FDR q-values.

Why patient-level?
------------------
Keeps the unit of analysis as patient (avoids pseudo-replication from slices).

Notes
-----
- Requires cluster_ids in NPZ.
- If a patient has <2 slices in cluster k after masking, that patient is skipped for that k.
- p-values:
    * tries SciPy mannwhitneyu; if SciPy not available, falls back to a permutation test.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import _merge_labels_and_clinical  # noqa: E402


def _discover_npz_files(npz_dir: str) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    return [os.path.join(npz_dir, f) for f in names]


def _infer_k(npz_paths: list[str]) -> int:
    k_max = -1
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        if "cluster_ids" not in z.files:
            continue
        cid = np.asarray(z["cluster_ids"]).reshape(-1)
        if cid.size == 0:
            continue
        try:
            k_max = max(k_max, int(np.max(cid)))
        except Exception:
            continue
    return int(k_max + 1)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 42) -> float:
    """
    Two-sided permutation p-value on difference of means.
    """
    rng = np.random.default_rng(int(seed))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    obs = float(np.mean(a) - np.mean(b))
    x = np.concatenate([a, b], axis=0)
    na = a.shape[0]
    cnt = 0
    for _ in range(int(n_perm)):
        rng.shuffle(x)
        d = float(np.mean(x[:na]) - np.mean(x[na:]))
        if abs(d) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def _p_value(a: np.ndarray, b: np.ndarray, method: str, seed: int) -> float:
    if method == "mwu":
        p = _mannwhitney_p(a, b)
        if np.isfinite(p):
            return float(p)
        # fallback
        return _perm_p(a, b, seed=seed)
    if method == "perm":
        return _perm_p(a, b, seed=seed)
    raise ValueError(method)


@dataclass
class ClusterPatientFeature:
    patient_id: str
    label: int
    cluster_id: int
    n_slices_in_cluster: int
    mean_std: float
    l2_std: float


def _extract_cluster_features(npz_dir: str, clinical: pd.DataFrame, min_slices: int) -> tuple[list[ClusterPatientFeature], int]:
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)
    k = _infer_k(paths)
    if k <= 0:
        raise RuntimeError("No cluster_ids found to infer k.")

    rows: list[ClusterPatientFeature] = []
    for p in paths:
        pid = os.path.basename(p).replace(".npz", "")
        if pid not in clinical.index:
            continue
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            continue

        z = np.load(p, allow_pickle=True)
        if "embeddings" not in z.files or "cluster_ids" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)  # (S,D)
        cid = np.asarray(z["cluster_ids"]).reshape(-1).astype(np.int64)
        if cid.shape[0] != emb.shape[0]:
            continue

        keep = np.ones((emb.shape[0],), dtype=bool)
        if "mask" in z.files:
            m = np.asarray(z["mask"]).reshape(-1)
            if m.shape[0] == emb.shape[0]:
                keep = keep & m.astype(bool)

        emb = emb[keep]
        cid = cid[keep]
        if emb.shape[0] < int(min_slices):
            continue

        for kk in range(int(k)):
            idx = cid == kk
            if int(idx.sum()) < int(min_slices):
                continue
            e = emb[idx]
            stdv = e.std(axis=0, ddof=0)
            rows.append(
                ClusterPatientFeature(
                    patient_id=str(pid),
                    label=int(row["label"]),
                    cluster_id=int(kk),
                    n_slices_in_cluster=int(e.shape[0]),
                    mean_std=float(np.mean(stdv)),
                    l2_std=float(np.linalg.norm(stdv)),
                )
            )

    return rows, int(k)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster-wise std comparison (Normal vs TTS) with p-values.")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_slices_per_cluster", type=int, default=2)
    ap.add_argument("--p_method", choices=["mwu", "perm"], default="mwu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    rows, k = _extract_cluster_features(
        os.path.abspath(args.npz_dir), clinical, min_slices=int(args.min_slices_per_cluster)
    )
    if not rows:
        raise RuntimeError("No cluster patient features extracted (check min_slices, cluster_ids, masks).")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(out_dir / f"{args.latent_source}_cluster_patient_features.csv", index=False)

    out_rows = []
    for kk in range(int(k)):
        d = df[df["cluster_id"] == kk]
        a = d[d["label"] == 0]
        b = d[d["label"] == 1]
        if len(a) < 3 or len(b) < 3:
            continue

        for metric in ["mean_std", "l2_std"]:
            xa = a[metric].to_numpy(dtype=np.float64)
            xb = b[metric].to_numpy(dtype=np.float64)
            pval = _p_value(xa, xb, method=str(args.p_method), seed=int(args.seed))
            out_rows.append(
                {
                    "latent_source": str(args.latent_source),
                    "cluster_id": int(kk),
                    "metric": metric,
                    "n_normal": int(len(xa)),
                    "n_tts": int(len(xb)),
                    "normal_mean": float(np.mean(xa)),
                    "tts_mean": float(np.mean(xb)),
                    "diff_tts_minus_normal": float(np.mean(xb) - np.mean(xa)),
                    "p_value": float(pval),
                }
            )

    df_out = pd.DataFrame(out_rows)
    if df_out.empty:
        raise RuntimeError("No clusters had enough samples in both groups for testing.")

    # FDR correction per metric
    qvals = []
    for metric in df_out["metric"].unique().tolist():
        m = df_out["metric"] == metric
        q = _bh_fdr(df_out.loc[m, "p_value"].to_numpy(dtype=np.float64))
        qvals.append(pd.Series(q, index=df_out.loc[m].index))
    df_out["q_value_bh"] = pd.concat(qvals).sort_index().to_numpy()

    df_out = df_out.sort_values(["metric", "q_value_bh", "p_value"])
    out_csv = out_dir / f"{args.latent_source}_cluster_std_pvalues.csv"
    df_out.to_csv(out_csv, index=False)

    # quick summary print
    top = df_out.head(10)
    print("Wrote:", out_csv)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

