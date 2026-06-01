#!/usr/bin/env python3
"""
t-SNE visual evidence + confidence ellipse area + outlier/ambiguous sample listing.

What this does
--------------
1) Build a patient-level feature matrix X from StageB `patients/*.npz`:
   - pooling: mean / std / mean_std / cluster_hist
2) 2D embedding with t-SNE (sklearn).
3) Plot Normal vs TTS scatter and draw group-wise confidence ellipses.
4) Quantify ellipse area per group (2D covariance-based, chi2 95% scaling).
5) List:
   - outliers: points outside the group's 95% ellipse (in 2D)
   - (optional) ambiguous test samples: low-confidence predictions from a train+val trained classifier

Notes
-----
- t-SNE is non-linear + stochastic; use --seed and keep it fixed.
- Ellipse is computed in the *2D t-SNE space* (descriptive visualization, not a formal hypothesis test).
- No-leakage rule is enforced only for the optional classifier-based ambiguity on a fixed test split
  (train+val fit -> test transform/predict).
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
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the exact same NPZ+clinical alignment logic as repeated evaluation
from repeated_stratified_shuffle_eval import (  # noqa: E402
    _load_features,
    _merge_labels_and_clinical,
)


@dataclass
class EllipseStats:
    mean: np.ndarray  # (2,)
    cov: np.ndarray  # (2,2)
    chi2_val: float
    area: float


def _chi2_val_2d(conf: float) -> float:
    """
    Chi-square quantile for df=2.
    Hardcode common values to avoid scipy dependency:
    - 0.95 -> 5.991464547...
    - 0.90 -> 4.605170186...
    - 0.99 -> 9.210340372...
    """
    if abs(conf - 0.95) < 1e-12:
        return 5.991464547107979
    if abs(conf - 0.90) < 1e-12:
        return 4.605170185988092
    if abs(conf - 0.99) < 1e-12:
        return 9.210340371976182
    raise ValueError("Only conf in {0.90,0.95,0.99} supported without SciPy.")


def _ellipse_stats(X2: np.ndarray, conf: float) -> EllipseStats:
    X2 = np.asarray(X2, dtype=np.float64)
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError(f"Expected (n,2), got {X2.shape}")
    mu = X2.mean(axis=0)
    cov = np.cov(X2.T, ddof=0)
    cov = 0.5 * (cov + cov.T)
    chi2_val = _chi2_val_2d(conf)
    # area = pi * a * b where a,b are semi-axis lengths at given chi2 scale.
    # eigenvalues of cov are variances along principal axes.
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    a = np.sqrt(chi2_val * eigvals[1])
    b = np.sqrt(chi2_val * eigvals[0])
    area = float(np.pi * a * b)
    return EllipseStats(mean=mu, cov=cov, chi2_val=float(chi2_val), area=area)


def _ellipse_patch(stats: EllipseStats, *, color: str, lw: float = 2.0, label: Optional[str] = None) -> Ellipse:
    # width/height are full lengths (2a, 2b)
    vals, vecs = np.linalg.eigh(stats.cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    a = np.sqrt(stats.chi2_val * max(vals[0], 1e-12))
    b = np.sqrt(stats.chi2_val * max(vals[1], 1e-12))
    e = Ellipse(
        xy=stats.mean,
        width=float(2.0 * a),
        height=float(2.0 * b),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        label=label,
    )
    return e


def _is_inside_ellipse(x: np.ndarray, stats: EllipseStats) -> np.ndarray:
    """
    Mahalanobis distance check in 2D: (x-mu)^T cov^{-1} (x-mu) <= chi2_val
    Returns boolean mask (n,).
    """
    X = np.asarray(x, dtype=np.float64)
    mu = stats.mean.reshape(1, 2)
    cov = stats.cov
    inv = np.linalg.inv(cov + 1e-12 * np.eye(2))
    d = X - mu
    md2 = np.einsum("ni,ij,nj->n", d, inv, d)
    return md2 <= stats.chi2_val


def _trainval_test_split_ids(split_dir: str) -> tuple[set[str], set[str]]:
    import pandas as _pd

    def _ids(p: str) -> set[str]:
        df = _pd.read_csv(p)
        col = "patient_id" if "patient_id" in df.columns else ("ID" if "ID" in df.columns else None)
        if col is None:
            raise ValueError(f"No patient id column in {p}")
        return set(df[col].astype(str).tolist())

    split_dir = os.path.abspath(split_dir)
    tr = _ids(os.path.join(split_dir, "train.csv"))
    va = _ids(os.path.join(split_dir, "val.csv"))
    te = _ids(os.path.join(split_dir, "test.csv"))
    return (tr | va), te


def main() -> None:
    ap = argparse.ArgumentParser(description="t-SNE + confidence ellipse + outlier/ambiguous sample analysis")
    ap.add_argument("--labels_csv", action="append", required=True, help="Repeat: train/val/test clinical CSVs")
    ap.add_argument("--npz_dir", required=True, help="StageB patients/*.npz directory")
    ap.add_argument("--latent_source", default="plain_ae", help="Name used in plots/outputs")
    ap.add_argument("--pooling", choices=["mean", "std", "mean_std", "cluster_hist"], default="std")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=20.0)
    ap.add_argument("--tsne_lr", type=float, default=200.0)
    ap.add_argument("--tsne_iter", type=int, default=2000)
    ap.add_argument("--ellipse_conf", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    ap.add_argument(
        "--standardize_before_tsne",
        action="store_true",
        help="Apply StandardScaler to X before t-SNE (fit on all X; visualization-only).",
    )
    ap.add_argument(
        "--split_dir",
        default=None,
        help="If provided: compute 'ambiguous' samples on the fixed test split (train+val fit -> test predict).",
    )
    ap.add_argument(
        "--ambiguous_topk",
        type=int,
        default=10,
        help="How many lowest-confidence test samples to list (closest to 0.5 probability).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    ds = _load_features(os.path.abspath(args.npz_dir), clinical, pooling=str(args.pooling))

    X = ds.X.astype(np.float64)
    y = ds.y.astype(int)
    pids = ds.patient_ids.astype(str)

    X_for_tsne = X
    if bool(args.standardize_before_tsne):
        X_for_tsne = StandardScaler().fit_transform(X_for_tsne)

    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        learning_rate=float(args.tsne_lr),
        n_iter=int(args.tsne_iter),
        init="pca",
        random_state=int(args.seed),
        metric="euclidean",
    )
    Z = tsne.fit_transform(X_for_tsne).astype(np.float64)

    # group stats
    m0 = y == 0
    m1 = y == 1
    if m0.sum() < 3 or m1.sum() < 3:
        raise RuntimeError("Too few samples per class for ellipse stats.")

    s0 = _ellipse_stats(Z[m0], conf=float(args.ellipse_conf))
    s1 = _ellipse_stats(Z[m1], conf=float(args.ellipse_conf))
    in0 = _is_inside_ellipse(Z[m0], s0)
    in1 = _is_inside_ellipse(Z[m1], s1)

    outlier_rows: list[dict] = []
    clin_idx = clinical.set_index("patient_id", drop=False)
    for pid, zz, lab, inside in zip(pids[m0], Z[m0], y[m0], in0):
        if bool(inside):
            continue
        row = clin_idx.loc[str(pid)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        outlier_rows.append(
            {
                "patient_id": str(pid),
                "label": int(lab),
                "group": "Normal",
                "z1": float(zz[0]),
                "z2": float(zz[1]),
                "age": float(row["age"]),
                "sex": str(row["sex"]),
            }
        )
    for pid, zz, lab, inside in zip(pids[m1], Z[m1], y[m1], in1):
        if bool(inside):
            continue
        row = clin_idx.loc[str(pid)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        outlier_rows.append(
            {
                "patient_id": str(pid),
                "label": int(lab),
                "group": "TTS",
                "z1": float(zz[0]),
                "z2": float(zz[1]),
                "age": float(row["age"]),
                "sex": str(row["sex"]),
            }
        )

    df_out = pd.DataFrame(outlier_rows).sort_values(["group", "patient_id"])
    out_csv = out_dir / f"tsne_outliers_{args.latent_source}_{args.pooling}.csv"
    df_out.to_csv(out_csv, index=False)

    df_area = pd.DataFrame(
        [
            {
                "latent_source": str(args.latent_source),
                "pooling": str(args.pooling),
                "tsne_seed": int(args.seed),
                "ellipse_conf": float(args.ellipse_conf),
                "group": "Normal",
                "n": int(m0.sum()),
                "ellipse_area": float(s0.area),
            },
            {
                "latent_source": str(args.latent_source),
                "pooling": str(args.pooling),
                "tsne_seed": int(args.seed),
                "ellipse_conf": float(args.ellipse_conf),
                "group": "TTS",
                "n": int(m1.sum()),
                "ellipse_area": float(s1.area),
            },
        ]
    )
    area_csv = out_dir / f"tsne_ellipse_area_{args.latent_source}_{args.pooling}.csv"
    df_area.to_csv(area_csv, index=False)

    # plot
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(Z[m0, 0], Z[m0, 1], s=35, alpha=0.75, c="tab:blue", label=f"Normal (n={m0.sum()})")
    ax.scatter(Z[m1, 0], Z[m1, 1], s=35, alpha=0.75, c="tab:orange", label=f"TTS (n={m1.sum()})")
    ax.add_patch(_ellipse_patch(s0, color="tab:blue", label=f"Normal {int(args.ellipse_conf*100)}% ellipse"))
    ax.add_patch(_ellipse_patch(s1, color="tab:orange", label=f"TTS {int(args.ellipse_conf*100)}% ellipse"))
    ax.set_title(f"{args.latent_source} / {args.pooling} — t-SNE(512d→2d) + ellipse area\n"
                 f"Area(N)={s0.area:.2f}, Area(TTS)={s1.area:.2f}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_ellipse_{args.latent_source}_{args.pooling}.png", dpi=180)
    plt.close(fig)

    # optional: ambiguous test samples (fixed split)
    if args.split_dir:
        tv_ids, te_ids = _trainval_test_split_ids(str(args.split_dir))
        tv_mask = np.array([pid in tv_ids for pid in pids], dtype=bool)
        te_mask = np.array([pid in te_ids for pid in pids], dtype=bool)
        if not np.any(te_mask) or not np.any(tv_mask):
            raise RuntimeError("split_dir provided but no aligned train+val/test patients found.")
        if np.unique(y[tv_mask]).size < 2 or np.unique(y[te_mask]).size < 2:
            raise RuntimeError("Need both classes in train+val and test to compute ambiguity scores.")

        # Standardize X on train+val only for the classifier (separate from t-SNE standardization)
        sc = StandardScaler()
        Xtv = sc.fit_transform(X[tv_mask])
        Xte = sc.transform(X[te_mask])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=int(args.seed), solver="lbfgs")
        clf.fit(Xtv, y[tv_mask])
        p = clf.predict_proba(Xte)[:, 1]
        margin = np.abs(p - 0.5)

        pid_te = pids[te_mask]
        y_te = y[te_mask]
        order = np.argsort(margin)  # smallest margin = most ambiguous
        k = min(int(args.ambiguous_topk), int(order.shape[0]))
        amb = []
        for idx in order[:k]:
            pid = str(pid_te[idx])
            row = clin_idx.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            amb.append(
                {
                    "patient_id": pid,
                    "label": int(y_te[idx]),
                    "prob_tts": float(p[idx]),
                    "margin_abs(p-0.5)": float(margin[idx]),
                    "age": float(row["age"]),
                    "sex": str(row["sex"]),
                }
            )
        amb_csv = out_dir / f"ambiguous_test_top{int(args.ambiguous_topk)}_{args.latent_source}_{args.pooling}.csv"
        pd.DataFrame(amb).to_csv(amb_csv, index=False)

    print("Wrote:", out_dir)
    print(" -", out_csv)
    print(" -", area_csv)


if __name__ == "__main__":
    main()

