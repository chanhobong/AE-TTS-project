#!/usr/bin/env python3
"""
Repeated StratifiedShuffleSplit evaluation for a custom feature set:
  X = [std_vector(512), mean_dist_upper(1)]

This mirrors `repeated_stratified_shuffle_eval.py` output structure so it can be merged into
`repeated_summary_merged_with_fixed.csv` using the same method (collect run_*/**/summary.csv
and compare_fixed_test_positions.csv).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import _load_fixed_test_positions, _load_features, _merge_labels_and_clinical  # noqa: E402


@dataclass
class Dataset:
    patient_ids: np.ndarray
    y: np.ndarray
    X: np.ndarray


def _build_dataset(
    *, npz_dir: str, mean_dist_csv: str, labels_csv: list[str], pooling: str = "std"
) -> Dataset:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    ds = _load_features(os.path.abspath(npz_dir), clinical, pooling=pooling)
    df_std = pd.DataFrame(ds.X.astype(np.float64))
    df_std.insert(0, "patient_id", ds.patient_ids.astype(str))
    df_std.insert(1, "label", ds.y.astype(int))

    df_md = pd.read_csv(mean_dist_csv)
    df_md["patient_id"] = df_md["patient_id"].astype(str)
    if "mean_dist_upper" not in df_md.columns:
        raise ValueError(f"mean_dist_csv missing 'mean_dist_upper': {mean_dist_csv}")

    df = df_std.merge(df_md[["patient_id", "mean_dist_upper"]], on="patient_id", how="inner")
    df = df.dropna(subset=["mean_dist_upper"])
    if df.shape[0] < 10:
        raise RuntimeError(f"Too few merged patients: {df.shape[0]}")

    feat_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    feat_cols = sorted(feat_cols, key=lambda x: int(x))
    X_std = df[feat_cols].to_numpy(dtype=np.float64)
    md = df["mean_dist_upper"].to_numpy(dtype=np.float64).reshape(-1, 1)
    X = np.concatenate([X_std, md], axis=1).astype(np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    pid = df["patient_id"].to_numpy(dtype=str)
    return Dataset(patient_ids=pid, y=y, X=X)


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    q025, q975 = np.quantile(x, [0.025, 0.975])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p2p5": float(q025),
        "p97p5": float(q975),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeated eval for std_vector + mean_dist_upper (logistic)")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--mean_dist_csv", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--pooling", default="std_plus_mean_dist")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--fixed_test_csv", default=None)
    args = ap.parse_args()

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(os.path.abspath(args.out_dir), f"run_{tag}")
    os.makedirs(out_root, exist_ok=True)

    ds = _build_dataset(
        npz_dir=os.path.abspath(args.npz_dir),
        mean_dist_csv=os.path.abspath(args.mean_dist_csv),
        labels_csv=[os.path.abspath(p) for p in args.labels_csv],
        pooling="std",
    )
    if np.unique(ds.y).size < 2:
        raise RuntimeError("Only one class present after join.")

    fixed_df = _load_fixed_test_positions(args.fixed_test_csv) if args.fixed_test_csv else None
    fixed_rows: list[dict] = []

    splitter = StratifiedShuffleSplit(
        n_splits=int(args.n_splits),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    rep_rows: list[dict] = []
    for repeat_id, (tr, te) in enumerate(splitter.split(ds.X, ds.y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(ds.X[tr])
        Xte = sc.transform(ds.X[te])
        ytr, yte = ds.y[tr], ds.y[te]
        pid_te = ds.patient_ids[te]

        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, solver="lbfgs")
        clf.fit(Xtr, ytr)
        s_tr = clf.predict_proba(Xtr)[:, 1]
        s_te = clf.predict_proba(Xte)[:, 1]
        rep_rows.append(
            {
                "latent_source": str(args.latent_source),
                "pooling": str(args.pooling),
                "model": "logistic",
                "repeat_id": int(repeat_id),
                "train_size": int(len(tr)),
                "test_size": int(len(te)),
                "roc_auc": float(roc_auc_score(yte, s_te)),
                "pr_auc": float(average_precision_score(yte, s_te)),
                "train_roc_auc": float(roc_auc_score(ytr, s_tr)),
                "train_pr_auc": float(average_precision_score(ytr, s_tr)),
                "test_patient_ids": ";".join(pid_te.tolist()),
            }
        )

    df_rep = pd.DataFrame(rep_rows)
    sub = os.path.join(out_root, str(args.latent_source), str(args.pooling), "logistic")
    os.makedirs(sub, exist_ok=True)
    df_rep.to_csv(os.path.join(sub, "repeats.csv"), index=False)

    roc_s = _summary_stats(df_rep["roc_auc"].to_numpy())
    pr_s = _summary_stats(df_rep["pr_auc"].to_numpy())
    df_sum = pd.DataFrame(
        [
            {
                "latent_source": str(args.latent_source),
                "pooling": str(args.pooling),
                "model": "logistic",
                **{f"roc_{k}": v for k, v in roc_s.items()},
                **{f"pr_{k}": v for k, v in pr_s.items()},
            }
        ]
    )
    df_sum.to_csv(os.path.join(sub, "summary.csv"), index=False)

    # hist plots
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(df_rep["roc_auc"].to_numpy(), bins=20, color="tab:blue", alpha=0.85)
    ax.set_title(f"{args.latent_source} / {args.pooling} / logistic — ROC-AUC (n={args.n_splits})")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("count")
    if fixed_df is not None:
        m = (
            (fixed_df["latent_source"].astype(str) == str(args.latent_source))
            & (fixed_df["pooling"].astype(str) == str(args.pooling))
            & (fixed_df["model"].astype(str) == "logistic")
        )
        if m.any() and "test_roc_auc" in fixed_df.columns:
            fv = float(fixed_df.loc[m, "test_roc_auc"].iloc[0])
            ax.axvline(fv, color="tab:red", linewidth=2, label=f"fixed={fv:.3f}")
            ax.legend(fontsize=8)
            pct = float(np.mean(df_rep["roc_auc"].to_numpy() <= fv))
            fixed_rows.append(
                {
                    "latent_source": str(args.latent_source),
                    "pooling": str(args.pooling),
                    "model": "logistic",
                    "fixed_test_roc_auc": fv,
                    "cdf_position": pct,
                }
            )
    fig.tight_layout()
    fig.savefig(os.path.join(sub, "hist_roc_auc_logistic.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(df_rep["pr_auc"].to_numpy(), bins=20, color="tab:green", alpha=0.85)
    ax.set_title(f"{args.latent_source} / {args.pooling} / logistic — PR-AUC (n={args.n_splits})")
    ax.set_xlabel("PR-AUC")
    ax.set_ylabel("count")
    if fixed_df is not None:
        m = (
            (fixed_df["latent_source"].astype(str) == str(args.latent_source))
            & (fixed_df["pooling"].astype(str) == str(args.pooling))
            & (fixed_df["model"].astype(str) == "logistic")
        )
        if m.any() and "test_pr_auc" in fixed_df.columns:
            fv = float(fixed_df.loc[m, "test_pr_auc"].iloc[0])
            ax.axvline(fv, color="tab:red", linewidth=2, label=f"fixed={fv:.3f}")
            ax.legend(fontsize=8)
            pct = float(np.mean(df_rep["pr_auc"].to_numpy() <= fv))
            fixed_rows.append(
                {
                    "latent_source": str(args.latent_source),
                    "pooling": str(args.pooling),
                    "model": "logistic",
                    "fixed_test_pr_auc": fv,
                    "pr_cdf_position": pct,
                }
            )
    fig.tight_layout()
    fig.savefig(os.path.join(sub, "hist_pr_auc_logistic.png"), dpi=150)
    plt.close(fig)

    if fixed_rows:
        pd.DataFrame(fixed_rows).to_csv(os.path.join(out_root, "compare_fixed_test_positions.csv"), index=False)

    print("Wrote:", out_root)


if __name__ == "__main__":
    main()

