#!/usr/bin/env python3
"""
3-panel figure: PCA-2D of MONAI patient-level features — mean | std | cluster_hist.

Same cohort per panel (patients with valid ``embeddings`` + ``cluster_ids`` cluster histogram).
Colors: Normal blue, TTS yellow (paper match to pooling PCA triptych, TTS as yellow).

Example:
  python3 utils/scripts/plot_monai_pooling_pca_triptych.py \\
    --npz_dir /path/to/diff3dformer_stageB_monai_ae_slice_meta/patients \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --out_png Report/Draft/figures/fig_monai_pooling_pca_triptych.png \\
    --cluster_hist_k 64
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

from repeated_stratified_shuffle_eval import (  # noqa: E402
    _aggregate_mean,
    _aggregate_std,
    _cluster_hist,
    _discover_npz_files,
    _infer_k,
    _merge_labels_and_clinical,
    _sex_to_bin,
)


def _train_patient_ids(train_csv: str) -> set[str]:
    df = pd.read_csv(train_csv)
    id_col = "patient_id" if "patient_id" in df.columns else "ID"
    return set(df[id_col].astype(str))


def _collect_monai(
    npz_dir: str,
    labels_csv: list[str],
    cluster_hist_k: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)

    gated: list[tuple[str, str]] = []
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
        gated.append((pid, p))

    if len(gated) < 5:
        raise RuntimeError(f"Too few patients after clinical gate: {len(gated)}")

    path_list = [t[1] for t in gated]
    k_inf = _infer_k(path_list)
    if k_inf <= 0:
        raise RuntimeError("cluster_hist: no cluster_ids in NPZ files under this directory.")

    if cluster_hist_k is not None:
        k_bins = int(cluster_hist_k)
        if k_bins < 1:
            raise ValueError("--cluster_hist_k must be >= 1")
        drop_out = k_bins < k_inf
    else:
        k_bins = k_inf
        drop_out = False

    xs_m: list[np.ndarray] = []
    xs_s: list[np.ndarray] = []
    xs_c: list[np.ndarray] = []
    ys: list[int] = []
    pids: list[str] = []

    for pid, p in gated:
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        z = np.load(p, allow_pickle=True)
        if "embeddings" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] < 1:
            continue
        mean_v = _aggregate_mean(emb).astype(np.float64)
        std_v = _aggregate_std(emb).astype(np.float64)
        h = _cluster_hist(p, k=k_bins, drop_outside=drop_out)
        if h is None:
            continue
        h = h.astype(np.float64)
        if np.any(~np.isfinite(mean_v)) or np.any(~np.isfinite(std_v)) or np.any(~np.isfinite(h)):
            continue
        xs_m.append(mean_v)
        xs_s.append(std_v)
        xs_c.append(h)
        ys.append(int(row["label"]))
        pids.append(pid)

    if len(pids) < 5:
        raise RuntimeError(f"Too few patients after mean/std/cluster_hist filters: {len(pids)}")

    return (
        np.stack(xs_m, axis=0),
        np.stack(xs_s, axis=0),
        np.stack(xs_c, axis=0),
        np.asarray(ys, dtype=np.int64),
        pids,
        k_bins,
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
    ap = argparse.ArgumentParser(description="MONAI mean / std / cluster_hist PCA triptych (Normal vs TTS).")
    ap.add_argument("--npz_dir", required=True, help="Directory of per-patient MONAE .npz (embeddings + cluster_ids).")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--train_csv", default=None, help="PCA/scaler fit subset (default: first --labels_csv).")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--cluster_hist_k", type=int, default=None, help="Histogram length K; default infer from data.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    train_csv = args.train_csv or args.labels_csv[0]
    train_ids = _train_patient_ids(train_csv)

    Xm, Xs, Xc, y, pids, k_bins = _collect_monai(
        str(args.npz_dir),
        list(args.labels_csv),
        int(args.cluster_hist_k) if args.cluster_hist_k is not None else None,
    )
    train_mask = np.array([p in train_ids for p in pids], dtype=bool)
    if train_mask.sum() < 3:
        raise SystemExit(f"Too few train patients in NPZ∩labels (got {train_mask.sum()}). Check --train_csv.")

    Zm, evm = _fit_transform_2d(Xm, train_mask)
    Zs, evs = _fit_transform_2d(Xs, train_mask)
    Zc, evc = _fit_transform_2d(Xc, train_mask)

    label_names = {0: "Normal", 1: "TTS"}
    colors = {0: "#4C72B0", 1: "#E6C229"}

    hist_sub = f" (K={k_bins})"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    panels = [
        (Zm, evm, "Mean pooling\n(slice-wise latent mean)"),
        (Zs, evs, "Std pooling\n(slice-wise latent variability)"),
        (Zc, evc, f"Cluster-hist pooling{hist_sub}\n(normalized bin counts)"),
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
    print("Wrote", out, f"(n={len(pids)}, cluster_hist K={k_bins})", file=sys.stderr)


if __name__ == "__main__":
    main()
