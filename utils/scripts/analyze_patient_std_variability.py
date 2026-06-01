#!/usr/bin/env python3
"""
Patient-level latent variability analysis based on std pooling (raw slice-wise std).

Given StageB per-patient NPZ files with `embeddings` (num_slices, dim), we compute per-patient:
  std_vector = std(embeddings, axis=0, ddof=0)  -> shape (dim,)

Outputs
-------
1) Violin plot of ||std_vector||_2 by group (Normal vs TTS)
2) Violin plot of mean(std_vector) by group
3) Top-k latent dimensions by mean std difference (TTS - Normal) + bar plot
4) Heatmap: patients x latent_dims of std_vector (optionally sorted)
5) CSVs: per-patient scalars + dim-wise summary

Notes
-----
- Uses *raw* std_vector (not StandardScaler-transformed), since the goal is variability interpretation.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class StdDataset:
    patient_id: np.ndarray  # (n,)
    y: np.ndarray  # (n,)
    std_vec: np.ndarray  # (n, d)
    age: np.ndarray  # (n,)
    sex: np.ndarray  # (n,)


def _load_std_vectors(npz_dir: str, clinical: pd.DataFrame) -> StdDataset:
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)

    pids: list[str] = []
    ys: list[int] = []
    ages: list[float] = []
    sexes: list[str] = []
    vecs: list[np.ndarray] = []

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
        if "embeddings" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] < 2:
            continue
        stdv = emb.std(axis=0, ddof=0).astype(np.float64)

        pids.append(pid)
        ys.append(int(row["label"]))
        ages.append(float(row["age"]))
        sexes.append(str(row["sex"]))
        vecs.append(stdv)

    if len(vecs) < 5:
        raise RuntimeError(f"Too few patients after join: {len(vecs)}")
    X = np.stack(vecs, axis=0)
    return StdDataset(
        patient_id=np.array(pids, dtype=str),
        y=np.array(ys, dtype=np.int64),
        std_vec=X,
        age=np.array(ages, dtype=np.float64),
        sex=np.array(sexes, dtype=str),
    )


def _violin(ax, data_by_group: list[np.ndarray], labels: list[str], title: str, ylabel: str) -> None:
    parts = ax.violinplot(data_by_group, showmeans=True, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patient-level latent variability analysis (std pooling).")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--max_patients_heatmap", type=int, default=200)
    ap.add_argument(
        "--sort_patients_by",
        choices=["label", "l2", "mean"],
        default="label",
        help="How to order patients in the heatmap.",
    )
    ap.add_argument(
        "--dims_for_heatmap",
        choices=["all", "topk_diff"],
        default="topk_diff",
        help="Which latent dims to include in the heatmap.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    ds = _load_std_vectors(os.path.abspath(args.npz_dir), clinical)

    X = ds.std_vec
    y = ds.y
    pids = ds.patient_id
    d = int(X.shape[1])

    # per-patient scalar summaries
    l2 = np.linalg.norm(X, ord=2, axis=1)
    mstd = X.mean(axis=1)

    df_pat = pd.DataFrame(
        {
            "patient_id": pids,
            "label": y,
            "l2_std": l2,
            "mean_std": mstd,
            "age": ds.age,
            "sex": ds.sex,
        }
    )
    df_pat.to_csv(out_dir / f"{args.latent_source}_patient_std_scalars.csv", index=False)

    # group split
    Xn = X[y == 0]
    Xt = X[y == 1]
    if Xn.shape[0] < 3 or Xt.shape[0] < 3:
        raise RuntimeError("Need >=3 per group.")

    # dim-wise mean/std and difference
    mean_n = Xn.mean(axis=0)
    mean_t = Xt.mean(axis=0)
    diff = mean_t - mean_n  # TTS - Normal
    absdiff = np.abs(diff)
    topk = int(min(max(args.topk, 1), d))
    top_idx = np.argsort(absdiff)[::-1][:topk]

    df_dim = pd.DataFrame(
        {
            "dim": np.arange(d, dtype=int),
            "mean_std_normal": mean_n,
            "mean_std_tts": mean_t,
            "diff_tts_minus_normal": diff,
            "abs_diff": absdiff,
        }
    ).sort_values("abs_diff", ascending=False)
    df_dim.to_csv(out_dir / f"{args.latent_source}_std_dimwise_summary.csv", index=False)

    # --- Plots: violin scalars
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.6))
    _violin(
        ax[0],
        [l2[y == 0], l2[y == 1]],
        ["Normal", "TTS"],
        title=f"{args.latent_source} — ||std_vector||₂ distribution",
        ylabel="||std||₂ (raw)",
    )
    _violin(
        ax[1],
        [mstd[y == 0], mstd[y == 1]],
        ["Normal", "TTS"],
        title=f"{args.latent_source} — mean(std_vector) distribution",
        ylabel="mean(std) (raw)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.latent_source}_std_scalar_violins.png", dpi=180)
    plt.close(fig)

    # --- Plot: top-k dim difference bar
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(topk)
    vals = diff[top_idx]
    colors = ["tab:orange" if v > 0 else "tab:blue" for v in vals]
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in top_idx], rotation=90)
    ax.set_title(f"{args.latent_source} — top-{topk} dims by |mean std diff| (TTS - Normal)")
    ax.set_xlabel("latent dim index")
    ax.set_ylabel("Δ mean(std) = mean_TTS - mean_Normal")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.latent_source}_top{topk}_meanstd_diff_dims.png", dpi=180)
    plt.close(fig)

    # --- Heatmap: patients x dims (std)
    # choose dims
    if str(args.dims_for_heatmap) == "topk_diff":
        dim_sel = top_idx
    else:
        dim_sel = np.arange(d)

    Xh = X[:, dim_sel]

    # sort patients
    if str(args.sort_patients_by) == "label":
        order = np.argsort(y)
    elif str(args.sort_patients_by) == "l2":
        order = np.argsort(l2)[::-1]
    else:
        order = np.argsort(mstd)[::-1]

    order = order[: int(min(len(order), int(args.max_patients_heatmap)))]
    Xh = Xh[order]
    yh = y[order]
    pid_h = pids[order]

    # normalize heatmap per-dim for visibility (z-score within selected patients)
    mu = Xh.mean(axis=0, keepdims=True)
    sd = Xh.std(axis=0, ddof=0, keepdims=True)
    Xz = (Xh - mu) / np.maximum(sd, 1e-6)

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    im = ax.imshow(Xz, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title(
        f"{args.latent_source} — per-patient std heatmap (dims={len(dim_sel)}, patients={Xz.shape[0]})\n"
        f"Rows sorted by {args.sort_patients_by}; values are per-dim z-scored for visualization"
    )
    ax.set_xlabel("latent dim" + (" (top-k)" if str(args.dims_for_heatmap) == "topk_diff" else ""))
    ax.set_ylabel("patients (row order)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("z-scored std")

    # thin label strip on left: group
    # add as colored side bar using a twin axis
    ax2 = ax.inset_axes([-0.02, 0, 0.015, 1], transform=ax.transAxes)
    ax2.imshow(yh.reshape(-1, 1), aspect="auto", cmap="Wistia", vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("y", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{args.latent_source}_patient_by_dim_std_heatmap.png", dpi=180)
    plt.close(fig)

    # Save the heatmap matrix with ids for downstream
    df_h = pd.DataFrame(Xh, columns=[f"dim_{int(i)}" for i in dim_sel])
    df_h.insert(0, "patient_id", pid_h)
    df_h.insert(1, "label", yh.astype(int))
    df_h.to_csv(out_dir / f"{args.latent_source}_patient_std_matrix_for_heatmap.csv", index=False)

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()

