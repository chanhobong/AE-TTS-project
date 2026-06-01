#!/usr/bin/env python3
"""
Pairwise Distance Matrix validation using cluster centroids (patient-level).

Idea (as requested)
-------------------
For each patient:
  1) Compute cluster centroids: c_k = mean(embeddings[slices in cluster k])
  2) Compute pairwise Euclidean distance matrix D (K x K), K~64
  3) Reduce to a scalar per patient:
       - sum_dist = sum_{i<j} D[i,j] over valid pairs
       - mean_dist = mean_{i<j} D[i,j] over valid pairs (recommended; controls missing clusters)

Group comparison:
  - Mean matrix: average D over Normal vs TTS (pairwise mean with per-entry counts)
  - Scalar comparison: t-test (Welch) on sum_dist and mean_dist across patients

Missing clusters
----------------
Some patients may not have slices in every cluster.
We mark missing centroids as NaN and exclude those pairs from sum/mean.
We also store n_valid_pairs per patient for transparency.

Outputs
-------
out_dir/
  per_patient_scalar.csv
  group_mean_matrix_normal.npy / .png
  group_mean_matrix_tts.npy / .png
  group_mean_matrix_diff_tts_minus_normal.npy / .png
  group_counts_matrix_normal.npy
  group_counts_matrix_tts.npy
  stats.txt
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
import matplotlib.pyplot as plt

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


def _nan_euclidean_matrix(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    C: (K,D) centroids with NaNs for missing clusters.
    Returns:
      D: (K,K) with NaNs where pair invalid
      valid: (K,K) boolean mask of valid pairs (both centroids finite)
    """
    C = np.asarray(C, dtype=np.float64)
    finite = np.isfinite(C).all(axis=1)  # (K,)
    valid = finite[:, None] & finite[None, :]
    # compute full distance matrix (will contain garbage for NaN rows, then masked)
    diff = C[:, None, :] - C[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))
    D[~valid] = np.nan
    np.fill_diagonal(D, 0.0)
    return D, valid


def _upper_triangle_sum_mean(D: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    K = D.shape[0]
    iu = np.triu_indices(K, k=1)
    vv = valid[iu]
    dd = D[iu]
    m = vv & np.isfinite(dd)
    n = int(np.sum(m))
    if n == 0:
        return float("nan"), float("nan"), 0
    s = float(np.sum(dd[m]))
    return s, float(s / n), n


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Welch t-test; tries SciPy, falls back to NaN p-value if SciPy missing.
    Returns (t_stat, p_value).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind  # type: ignore

        res = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # compute t-stat, but p-value requires t CDF; keep NaN if SciPy absent
        ma = np.nanmean(a)
        mb = np.nanmean(b)
        va = np.nanvar(a, ddof=1)
        vb = np.nanvar(b, ddof=1)
        na = int(np.sum(np.isfinite(a)))
        nb = int(np.sum(np.isfinite(b)))
        denom = np.sqrt(va / max(na, 1) + vb / max(nb, 1))
        t = float((ma - mb) / denom) if denom > 0 else float("nan")
        return t, float("nan")


def _perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 42) -> float:
    """Two-sided permutation p-value on difference of means."""
    rng = np.random.default_rng(int(seed))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    obs = float(np.mean(a) - np.mean(b))
    x = np.concatenate([a, b], axis=0)
    na = a.size
    cnt = 0
    for _ in range(int(n_perm)):
        rng.shuffle(x)
        d = float(np.mean(x[:na]) - np.mean(x[na:]))
        if abs(d) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def _save_matrix_png(mat: np.ndarray, out_path: Path, title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("cluster id")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise distance matrix validation (cluster centroids).")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_slices_per_cluster", type=int, default=2)
    ap.add_argument("--perm_n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    clinical = clinical.set_index("patient_id", drop=False)

    paths = _discover_npz_files(os.path.abspath(args.npz_dir))
    K = _infer_k(paths)
    if K <= 0:
        raise RuntimeError("No cluster_ids found to infer K.")

    # group accumulators for mean matrices: sum and count per entry
    sum0 = np.zeros((K, K), dtype=np.float64)
    cnt0 = np.zeros((K, K), dtype=np.int64)
    sum1 = np.zeros((K, K), dtype=np.float64)
    cnt1 = np.zeros((K, K), dtype=np.int64)

    rows = []
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
        if emb.ndim != 2 or cid.shape[0] != emb.shape[0]:
            continue

        keep = np.ones((emb.shape[0],), dtype=bool)
        if "mask" in z.files:
            m = np.asarray(z["mask"]).reshape(-1)
            if m.shape[0] == emb.shape[0]:
                keep = keep & m.astype(bool)
        emb = emb[keep]
        cid = cid[keep]
        if emb.shape[0] < 2:
            continue

        Ddim = int(emb.shape[1])
        C = np.full((K, Ddim), np.nan, dtype=np.float64)
        n_per_k = np.zeros((K,), dtype=np.int64)
        for k in range(K):
            idx = cid == k
            n = int(idx.sum())
            n_per_k[k] = n
            if n < int(args.min_slices_per_cluster):
                continue
            C[k] = emb[idx].mean(axis=0)

        D, valid = _nan_euclidean_matrix(C)
        # exclude diagonal from validity for counts matrix to be consistent
        np.fill_diagonal(valid, False)

        sum_dist, mean_dist, n_pairs = _upper_triangle_sum_mean(D, valid)
        lab = int(row["label"])
        rows.append(
            {
                "latent_source": str(args.latent_source),
                "patient_id": str(pid),
                "label": lab,
                "age": float(row["age"]),
                "sex": str(row["sex"]),
                "K": int(K),
                "D": int(Ddim),
                "min_slices_per_cluster": int(args.min_slices_per_cluster),
                "n_valid_pairs": int(n_pairs),
                "sum_dist_upper": float(sum_dist),
                "mean_dist_upper": float(mean_dist),
                "n_clusters_present": int(np.sum(n_per_k >= int(args.min_slices_per_cluster))),
            }
        )

        # accumulate group mean matrices entrywise
        m = np.isfinite(D) & valid
        if lab == 0:
            sum0[m] += D[m]
            cnt0[m] += 1
        else:
            sum1[m] += D[m]
            cnt1[m] += 1

    df = pd.DataFrame(rows)
    out_csv = out_dir / "per_patient_scalar.csv"
    df.to_csv(out_csv, index=False)

    mean0 = np.divide(sum0, np.maximum(cnt0, 1), dtype=np.float64)
    mean1 = np.divide(sum1, np.maximum(cnt1, 1), dtype=np.float64)
    mean0[cnt0 == 0] = np.nan
    mean1[cnt1 == 0] = np.nan
    diff = mean1 - mean0

    np.save(out_dir / "group_mean_matrix_normal.npy", mean0)
    np.save(out_dir / "group_mean_matrix_tts.npy", mean1)
    np.save(out_dir / "group_mean_matrix_diff_tts_minus_normal.npy", diff)
    np.save(out_dir / "group_counts_matrix_normal.npy", cnt0)
    np.save(out_dir / "group_counts_matrix_tts.npy", cnt1)

    # plots
    _save_matrix_png(mean0, out_dir / "group_mean_matrix_normal.png", f"{args.latent_source} mean D (Normal)")
    _save_matrix_png(mean1, out_dir / "group_mean_matrix_tts.png", f"{args.latent_source} mean D (TTS)")
    # diverging for diff
    vmax = float(np.nanpercentile(np.abs(diff), 98)) if np.isfinite(diff).any() else None
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax if vmax else None, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{args.latent_source} mean D diff (TTS - Normal)")
    ax.set_xlabel("cluster id")
    ax.set_ylabel("cluster id")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "group_mean_matrix_diff_tts_minus_normal.png", dpi=180)
    plt.close(fig)

    # stats on scalars
    a_sum = df[df["label"] == 0]["sum_dist_upper"].to_numpy(dtype=np.float64)
    b_sum = df[df["label"] == 1]["sum_dist_upper"].to_numpy(dtype=np.float64)
    a_mean = df[df["label"] == 0]["mean_dist_upper"].to_numpy(dtype=np.float64)
    b_mean = df[df["label"] == 1]["mean_dist_upper"].to_numpy(dtype=np.float64)

    t_sum, p_sum = _welch_ttest(a_sum, b_sum)
    t_mean, p_mean = _welch_ttest(a_mean, b_mean)
    p_sum_perm = _perm_pvalue(a_sum, b_sum, n_perm=int(args.perm_n), seed=int(args.seed))
    p_mean_perm = _perm_pvalue(a_mean, b_mean, n_perm=int(args.perm_n), seed=int(args.seed))

    stats_txt = out_dir / "stats.txt"
    stats_txt.write_text(
        "\n".join(
            [
                f"latent_source: {args.latent_source}",
                f"npz_dir: {os.path.abspath(args.npz_dir)}",
                f"K(inferred): {K}",
                f"min_slices_per_cluster: {int(args.min_slices_per_cluster)}",
                f"n_patients_total: {len(df)}",
                f"n_normal: {int((df['label']==0).sum())}",
                f"n_tts: {int((df['label']==1).sum())}",
                "",
                "Scalar = sum_dist_upper (upper triangle sum of distances)",
                f"  normal mean±std: {np.nanmean(a_sum):.4f} ± {np.nanstd(a_sum):.4f}",
                f"  tts    mean±std: {np.nanmean(b_sum):.4f} ± {np.nanstd(b_sum):.4f}",
                f"  welch t={t_sum:.4f}, p={p_sum}",
                f"  perm p (n={int(args.perm_n)}): {p_sum_perm}",
                "",
                "Scalar = mean_dist_upper (upper triangle mean distance; recommended)",
                f"  normal mean±std: {np.nanmean(a_mean):.4f} ± {np.nanstd(a_mean):.4f}",
                f"  tts    mean±std: {np.nanmean(b_mean):.4f} ± {np.nanstd(b_mean):.4f}",
                f"  welch t={t_mean:.4f}, p={p_mean}",
                f"  perm p (n={int(args.perm_n)}): {p_mean_perm}",
                "",
                "Note: sum_dist depends on #valid pairs; mean_dist controls for missing clusters.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Wrote:", out_dir)
    print(" -", out_csv)
    print(" -", stats_txt)


if __name__ == "__main__":
    main()

