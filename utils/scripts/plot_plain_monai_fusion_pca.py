#!/usr/bin/env python3
"""
2D PCA of **early-fused** Plain + MONAI representations (same patient intersection).

- **Plain** (trajectory-ordered NPZ): robust **P90–P10** slice-wise latent spread (same block as
  ``plot_pooling_pca_triptych`` right panel).
- **MONAE** (`embeddings` + `cluster_ids`): **cluster histogram** (normalized), same rules as eval.

Each block is **StandardScaler** fit on train patients only, then concatenated; PCA(2) is fit on
train only (no leakage). Interpreting PCs: joint variance after balancing block scales.

Colors: Normal blue, TTS orange (matches Plain triptych). Axis labels show per-PC explained variance ratio.

Example:
  python3 utils/scripts/plot_plain_monai_fusion_pca.py \\
    --plain_npz_dir .../diff3dformer_stageB_plain_ae_slice_meta/patients \\
    --monai_npz_dir .../diff3dformer_stageB_monai_ae_slice_meta/patients \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --out_png Report/Draft/figures/fig_ensemble_fusion_pca_plain_p90_monai_hist.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import trajectory_volatility_classifier_cv as traj_cv  # noqa: E402
from repeated_stratified_shuffle_eval import (  # noqa: E402
    _cluster_hist,
    _discover_npz_files,
    _infer_k,
    _merge_labels_and_clinical,
    _sex_to_bin,
)


def _npz_paths(npz_dir: str, recursive: bool) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if recursive:
        return traj_cv._discover_npz_paths(npz_dir, True)
    return _discover_npz_files(npz_dir)


def _resolve_npz_dir(path: str, *, tower: str) -> str:
    """Ensure directory exists; try slice_meta → non-slice_meta sibling (same as training scripts)."""
    p = os.path.abspath(os.path.expanduser(path.strip()))
    if os.path.isdir(p):
        return p
    cand: str | None = None
    if tower == "MONAI" and "monai_ae_slice_meta" in p:
        cand = p.replace(
            "diff3dformer_stageB_monai_ae_slice_meta",
            "diff3dformer_stageB_monai_ae",
        )
    elif tower == "PLAIN" and "plain_ae_slice_meta" in p:
        cand = p.replace(
            "diff3dformer_stageB_plain_ae_slice_meta",
            "diff3dformer_stageB_plain_ae",
        )
    if cand and os.path.isdir(cand):
        print(
            f"[plot_plain_monai_fusion_pca] {tower}: path not found, using sibling:\n  {cand}",
            file=sys.stderr,
        )
        return cand
    hint = ""
    if tower == "MONAI":
        hint = (
            "\nOften only .../diff3dformer_stageB_monai_ae/patients exists (no slice_meta export). "
            "Pass that path as --monai_npz_dir."
        )
    raise SystemExit(f"{tower} NPZ directory not found:\n  {p}{hint}")


def _train_patient_ids(train_csv: str) -> set[str]:
    df = pd.read_csv(train_csv)
    id_col = "patient_id" if "patient_id" in df.columns else "ID"
    return set(df[id_col].astype(str))


def _plain_p90_dict(
    npz_dir: str,
    labels_csv: list[str],
    eps_mm: float,
    recursive_npz: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _npz_paths(npz_dir, recursive_npz)
    feats: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for p in paths:
        pid = os.path.basename(p).replace(".npz", "")
        if pid not in clinical.index:
            continue
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            continue
        if not np.isfinite(_sex_to_bin(row["sex"])):
            continue
        z = np.load(p, allow_pickle=True)
        try:
            emb, zmm, _ = traj_cv._ordered_rows(z)
        except (ValueError, KeyError, IndexError):
            continue
        if emb.shape[0] < 2:
            continue
        bl = traj_cv.trajectory_feature_blocks_extended(emb, zmm, eps_mm)
        p90_v = bl["p90_p10"].astype(np.float64)
        if np.any(~np.isfinite(p90_v)):
            continue
        feats[pid] = p90_v
        labels[pid] = int(row["label"])
    return feats, labels


def _monai_hist_dict(
    npz_dir: str,
    labels_csv: list[str],
    recursive_npz: bool,
    cluster_hist_k: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _npz_paths(npz_dir, recursive_npz)

    eligible: list[str] = []
    for p in paths:
        pid = os.path.basename(p).replace(".npz", "")
        if pid not in clinical.index:
            continue
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            continue
        if not np.isfinite(_sex_to_bin(row["sex"])):
            continue
        eligible.append(p)

    if len(eligible) < 5:
        raise RuntimeError(f"MONAI: too few patients after clinical gate: {len(eligible)}")

    k_inf = _infer_k(eligible)
    if k_inf <= 0:
        raise RuntimeError("MONAI: no cluster_ids in NPZ.")

    if cluster_hist_k is not None:
        k_bins = int(cluster_hist_k)
        if k_bins < 1:
            raise ValueError("cluster_hist_k must be >= 1")
        drop_out = k_bins < k_inf
    else:
        k_bins = k_inf
        drop_out = False

    feats: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for p in eligible:
        pid = os.path.basename(p).replace(".npz", "")
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        z = np.load(p, allow_pickle=True)
        if "embeddings" not in z.files:
            continue
        h = _cluster_hist(p, k=k_bins, drop_outside=drop_out)
        if h is None:
            continue
        h = h.astype(np.float64)
        if np.any(~np.isfinite(h)):
            continue
        feats[pid] = h
        labels[pid] = int(row["label"])
    return feats, labels


def _align(
    plain_f: dict[str, np.ndarray],
    plain_y: dict[str, int],
    monai_f: dict[str, np.ndarray],
    monai_y: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    common = sorted(set(plain_f.keys()) & set(monai_f.keys()))
    if len(common) < 5:
        raise RuntimeError(f"Too few patients in Plain∩MONAI after features: {len(common)}")
    Xp: list[np.ndarray] = []
    Xh: list[np.ndarray] = []
    y: list[int] = []
    for pid in common:
        if plain_y[pid] != monai_y[pid]:
            raise RuntimeError(f"Label mismatch for patient {pid!r} between Plain and MONAI merges.")
        Xp.append(plain_f[pid])
        Xh.append(monai_f[pid])
        y.append(plain_y[pid])
    d_hist = Xh[0].shape[0]
    return (
        np.stack(Xp, axis=0),
        np.stack(Xh, axis=0),
        np.asarray(y, dtype=np.int64),
        common,
        d_hist,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA on block-standardized [Plain P90–P10 | MONAI cluster_hist].")
    ap.add_argument("--plain_npz_dir", required=True)
    ap.add_argument("--monai_npz_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--train_csv", default=None, help="default: first labels_csv")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--eps_mm", type=float, default=1e-3)
    ap.add_argument("--plain_recursive_npz", action="store_true")
    ap.add_argument("--monai_recursive_npz", action="store_true")
    ap.add_argument("--cluster_hist_k", type=int, default=None, help="MONAI histogram length; default infer.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="Plain P90–P10  ⊕  MONAI cluster-hist (block-z, train PCA)")
    args = ap.parse_args()

    train_csv = args.train_csv or args.labels_csv[0]
    train_ids = _train_patient_ids(train_csv)

    plain_dir = _resolve_npz_dir(str(args.plain_npz_dir), tower="PLAIN")
    monai_dir = _resolve_npz_dir(str(args.monai_npz_dir), tower="MONAI")

    plain_f, plain_y = _plain_p90_dict(
        plain_dir,
        list(args.labels_csv),
        float(args.eps_mm),
        bool(args.plain_recursive_npz),
    )
    monai_f, monai_y = _monai_hist_dict(
        monai_dir,
        list(args.labels_csv),
        bool(args.monai_recursive_npz),
        int(args.cluster_hist_k) if args.cluster_hist_k is not None else None,
    )
    Xp, Xh, y, pids, k_hist = _align(plain_f, plain_y, monai_f, monai_y)

    train_mask = np.array([p in train_ids for p in pids], dtype=bool)
    if train_mask.sum() < 3:
        raise SystemExit(f"Too few train patients in intersection (got {train_mask.sum()}).")

    sc_p = StandardScaler()
    sc_h = StandardScaler()
    sc_p.fit(Xp[train_mask])
    sc_h.fit(Xh[train_mask])
    X = np.hstack([sc_p.transform(Xp), sc_h.transform(Xh)]).astype(np.float64)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X[train_mask])
    Z = pca.transform(X)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    label_names = {0: "Normal", 1: "TTS"}
    colors = {0: "#4C72B0", 1: "#DD8452"}

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab in (0, 1):
        m = y == lab
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=32,
            alpha=0.78,
            c=colors[lab],
            edgecolors="white",
            linewidths=0.35,
            label=label_names[lab],
        )
    ax.set_xlabel(f"PC1 ({100.0 * float(evr[0]):.1f}% var.)")
    ax.set_ylabel(f"PC2 ({100.0 * float(evr[1]):.1f}% var.)")
    tit = str(args.title).strip()
    if tit:
        ax.set_title(tit, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", framealpha=0.92)
    fig.text(
        0.5,
        0.01,
        f"n={len(pids)}  |  MONAI cluster bins K={k_hist}  |  train-fit scaler+PCA",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    out = Path(args.out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
