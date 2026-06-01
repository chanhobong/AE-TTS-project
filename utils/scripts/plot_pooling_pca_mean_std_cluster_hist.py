#!/usr/bin/env python3
"""
3-panel figure: PCA-2D of patient-level features — mean | std | cluster histogram.

Order matches repeated_stratified_shuffle_eval pooling:
  - mean / std: slice axis on raw ``embeddings`` (same as eval, not trajectory re-ordered).
  - cluster_hist: normalized cluster_id counts (mask-aware), global K from max cluster id in cohort.

Colors: Normal (0) vs TTS (1). Scaler + PCA fit on train patients only (no leakage).
Axis labels report each PC's explained variance ratio (of standardized features).

Example:
  python3 utils/scripts/plot_pooling_pca_mean_std_cluster_hist.py \\
    --npz_dir /path/to/diff3dformer_stageB_monai_ae/patients \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --out_png Report/Draft/fig_pooling_pca_mean_std_hist.png
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
    _aggregate_mean,
    _aggregate_std,
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


def _train_patient_ids(train_csv: str) -> set[str]:
    df = pd.read_csv(train_csv)
    id_col = "patient_id" if "patient_id" in df.columns else "ID"
    return set(df[id_col].astype(str))


def _collect_mean_std_hist(
    npz_dir: str,
    labels_csv: list[str],
    recursive_npz: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
        sb = _sex_to_bin(row["sex"])
        if not np.isfinite(sb):
            continue
        eligible.append(p)

    if len(eligible) < 5:
        raise RuntimeError(f"Too few patients after clinical filters: {len(eligible)}")

    k = _infer_k(eligible)
    if k <= 0:
        raise RuntimeError("No cluster_ids in NPZ files; cannot build cluster histogram.")

    xs_m: list[np.ndarray] = []
    xs_s: list[np.ndarray] = []
    xs_h: list[np.ndarray] = []
    ys: list[int] = []
    pids: list[str] = []

    for p in eligible:
        pid = os.path.basename(p).replace(".npz", "")
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        z = np.load(p, allow_pickle=True)
        if "embeddings" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)
        if emb.size == 0:
            continue
        h = _cluster_hist(p, k=k)
        if h is None:
            continue
        mean_v = _aggregate_mean(emb)
        std_v = _aggregate_std(emb)
        if np.any(~np.isfinite(mean_v)) or np.any(~np.isfinite(std_v)) or np.any(~np.isfinite(h)):
            continue
        xs_m.append(mean_v)
        xs_s.append(std_v)
        xs_h.append(h)
        ys.append(int(row["label"]))
        pids.append(pid)

    if len(pids) < 5:
        raise RuntimeError(f"Too few patients with valid mean/std/cluster_hist: {len(pids)}")

    return (
        np.stack(xs_m, axis=0),
        np.stack(xs_s, axis=0),
        np.stack(xs_h, axis=0),
        np.asarray(ys, dtype=np.int64),
        pids,
    )


def _fit_transform_2d(
    X: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    pca = PCA(n_components=2, random_state=0)
    Xtr = X[train_mask]
    if Xtr.shape[0] < 3:
        raise RuntimeError("Need at least 3 train samples for scaler+PCA.")
    Xt = sc.fit_transform(Xtr)
    pca.fit(Xt)
    Z = pca.transform(sc.transform(X))
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return Z, evr


def _set_pc_axis_labels(ax, evr: np.ndarray) -> None:
    e0, e1 = float(evr[0]), float(evr[1])
    ax.set_xlabel(f"PC1 ({100.0 * e0:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({100.0 * e1:.1f}% var.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean / std / cluster histogram PCA triptych (Normal vs TTS).")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument(
        "--train_csv",
        default=None,
        help="CSV whose patient_id rows define PCA/scaler fit (default: first --labels_csv).",
    )
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--recursive_npz", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    train_csv = args.train_csv or args.labels_csv[0]
    train_ids = _train_patient_ids(train_csv)

    Xm, Xs, Xh, y, pids = _collect_mean_std_hist(
        str(args.npz_dir),
        list(args.labels_csv),
        bool(args.recursive_npz),
    )
    train_mask = np.array([p in train_ids for p in pids], dtype=bool)
    if train_mask.sum() < 3:
        raise SystemExit(f"Too few train patients in NPZ∩labels (got {train_mask.sum()}). Check --train_csv.")

    Zm, evm = _fit_transform_2d(Xm, train_mask)
    Zs, evs = _fit_transform_2d(Xs, train_mask)
    Zh, evh = _fit_transform_2d(Xh, train_mask)

    label_names = {0: "Normal", 1: "TTS"}
    colors = {0: "#4C72B0", 1: "#DD8452"}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    panels = [
        (Zm, evm, "Mean pooling\n(slice-wise latent mean)"),
        (Zs, evs, "Std pooling\n(slice-wise latent variability)"),
        (Zh, evh, "Cluster histogram\n(normalized cluster proportions)"),
    ]
    for ax, (Z, ev, title) in zip(axes, panels):
        for lab in (0, 1):
            m = y == lab
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=28,
                alpha=0.78,
                c=colors[lab],
                edgecolors="white",
                linewidths=0.35,
                label=label_names[lab],
            )
        _set_pc_axis_labels(ax, ev)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")
    axes[0].legend(loc="best", framealpha=0.92)
    supt = str(args.title).strip()
    if supt:
        fig.suptitle(supt, fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path(args.out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
