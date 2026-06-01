#!/usr/bin/env python3
"""
Plain AE: Logistic on std pooling vector (+1 feature) mean_dist_upper.

This matches the user's intent:
  X = [std_vector (512-d), mean_dist_upper (1-d)]  => (513-d)

Where:
  - std_vector is patient-level slice-wise std from embeddings:
      std_vector = std(embeddings, axis=0, ddof=0)  # raw (not standardized)
      loaded using the exact same function as repeated evaluation: _load_features(..., pooling="std")
  - mean_dist_upper is the per-patient mean of upper-triangle distances between cluster centroids:
      produced by cluster_pairwise_distance_matrix.py -> per_patient_scalar.csv

Protocol (no leakage)
---------------------
Fixed split_dir {train,val,test}. Unit is patient.
1) CV on train+val only with StratifiedKFold:
   - StandardScaler fit on fold-train only (on concatenated 513-d X)
   - LogisticRegression(class_weight="balanced") train on fold-train only
   - Evaluate ROC-AUC / PR-AUC on fold-val only
2) Final model:
   - Fit StandardScaler on full train+val
   - Train logistic on full train+val
   - Evaluate once on held-out test
3) Bootstrap CI on test (default n=1000): ROC-AUC & PR-AUC
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import _load_features, _merge_labels_and_clinical  # noqa: E402


def _bootstrap_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    rng = np.random.default_rng(int(seed))
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n = int(y_true.shape[0])
    roc_vals: list[float] = []
    pr_vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        ys = y_score[idx]
        roc_vals.append(float(roc_auc_score(yt, ys)))
        pr_vals.append(float(average_precision_score(yt, ys)))
    if not roc_vals or not pr_vals:
        nan3 = (float("nan"), float("nan"), float("nan"))
        return nan3, nan3
    roc = np.quantile(np.array(roc_vals), [0.025, 0.5, 0.975]).tolist()
    pr = np.quantile(np.array(pr_vals), [0.025, 0.5, 0.975]).tolist()
    return (float(roc[0]), float(roc[1]), float(roc[2])), (float(pr[0]), float(pr[1]), float(pr[2]))


def _read_ids(path: str) -> set[str]:
    df = pd.read_csv(path)
    col = "patient_id" if "patient_id" in df.columns else ("ID" if "ID" in df.columns else None)
    if col is None:
        raise ValueError(f"No patient id column in {path}")
    return set(df[col].astype(str).tolist())


@dataclass
class Dataset:
    patient_id: np.ndarray
    y: np.ndarray
    X: np.ndarray


def _build_dataset(
    *,
    npz_dir: str,
    mean_dist_csv: str,
    labels_csv: list[str],
) -> Dataset:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    ds_std = _load_features(os.path.abspath(npz_dir), clinical, pooling="std")

    df_std = pd.DataFrame(ds_std.X.astype(np.float64))
    df_std.insert(0, "patient_id", ds_std.patient_ids.astype(str))
    df_std.insert(1, "label", ds_std.y.astype(int))

    df_md = pd.read_csv(mean_dist_csv)
    df_md["patient_id"] = df_md["patient_id"].astype(str)
    need = {"patient_id", "mean_dist_upper"}
    if not need.issubset(df_md.columns):
        raise ValueError(f"mean_dist_csv missing {sorted(need)}; got {list(df_md.columns)}")

    df = df_std.merge(df_md[["patient_id", "mean_dist_upper"]], on="patient_id", how="inner")
    df = df.dropna(subset=["mean_dist_upper"])
    if df.shape[0] < 10:
        raise RuntimeError(f"Too few merged patients: {df.shape[0]}")

    feat_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    # pandas int columns from df_std are 0..511; ensure order
    feat_cols = sorted(feat_cols, key=lambda x: int(x))
    X_std = df[feat_cols].to_numpy(dtype=np.float64)
    md = df["mean_dist_upper"].to_numpy(dtype=np.float64).reshape(-1, 1)
    X = np.concatenate([X_std, md], axis=1).astype(np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    pid = df["patient_id"].to_numpy(dtype=str)
    return Dataset(patient_id=pid, y=y, X=X)


def main() -> None:
    ap = argparse.ArgumentParser(description="CV + bootstrap: logistic on [std_vector(512), mean_dist_upper(1)].")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True, help="Repeat: train/val/test clinical CSVs")
    ap.add_argument("--npz_dir", required=True, help="plain_ae StageB patients dir")
    ap.add_argument("--mean_dist_csv", required=True, help="per_patient_scalar.csv from cluster_pairwise_distance_matrix.py")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_bootstrap_n", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ds = _build_dataset(
        npz_dir=os.path.abspath(args.npz_dir),
        mean_dist_csv=os.path.abspath(args.mean_dist_csv),
        labels_csv=[os.path.abspath(p) for p in args.labels_csv],
    )

    tr_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "train.csv"))
    va_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "val.csv"))
    te_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "test.csv"))
    tv_set = tr_ids | va_ids

    tv_mask = np.array([p in tv_set for p in ds.patient_id], dtype=bool)
    te_mask = np.array([p in te_ids for p in ds.patient_id], dtype=bool)
    if not np.any(tv_mask) or not np.any(te_mask):
        raise RuntimeError("No aligned train+val or test patients after merge.")

    X_tv, y_tv = ds.X[tv_mask], ds.y[tv_mask]
    X_te, y_te = ds.X[te_mask], ds.y[te_mask]
    pid_te = ds.patient_id[te_mask]
    if np.unique(y_tv).size < 2 or np.unique(y_te).size < 2:
        raise RuntimeError("Need both classes in train+val and test.")

    # --- CV on train+val
    skf = StratifiedKFold(n_splits=int(args.k), shuffle=True, random_state=int(args.seed))
    cv_rows = []
    for fold, (tr, va) in enumerate(skf.split(X_tv, y_tv)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tv[tr])
        Xva = sc.transform(X_tv[va])
        ytr, yva = y_tv[tr], y_tv[va]
        if np.unique(yva).size < 2:
            continue
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, solver="lbfgs")
        clf.fit(Xtr, ytr)
        pva = clf.predict_proba(Xva)[:, 1]
        cv_rows.append(
            {
                "k": int(args.k),
                "fold": int(fold),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "roc_auc": float(roc_auc_score(yva, pva)),
                "pr_auc": float(average_precision_score(yva, pva)),
            }
        )
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)

    # --- Final fit on all train+val, eval on test
    sc = StandardScaler()
    Xtr_full = sc.fit_transform(X_tv)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, solver="lbfgs")
    clf.fit(Xtr_full, y_tv)
    pte = clf.predict_proba(Xte)[:, 1]
    roc = float(roc_auc_score(y_te, pte))
    pr = float(average_precision_score(y_te, pte))
    roc_q, pr_q = _bootstrap_roc_pr(y_te, pte, n_boot=int(args.test_bootstrap_n), seed=int(args.seed))

    # coefficient table (after scaling): last coefficient corresponds to mean_dist_upper feature
    coef = clf.coef_.reshape(-1)
    df_coef = pd.DataFrame(
        {
            "feature": [f"std_dim_{i}" for i in range(512)] + ["mean_dist_upper"],
            "coef": coef.tolist(),
        }
    )
    df_coef.to_csv(os.path.join(out_dir, "coef_scaled_space.csv"), index=False)

    df_test = pd.DataFrame(
        [
            {
                "feature_set": "std_vector(512) + mean_dist_upper(1)",
                "model": "logistic",
                "n_trainval": int(len(y_tv)),
                "n_test": int(len(y_te)),
                "test_roc_auc": roc,
                "test_pr_auc": pr,
                "test_roc_q025": roc_q[0],
                "test_roc_q50": roc_q[1],
                "test_roc_q975": roc_q[2],
                "test_pr_q025": pr_q[0],
                "test_pr_q50": pr_q[1],
                "test_pr_q975": pr_q[2],
            }
        ]
    )
    df_test.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)

    df_pred = pd.DataFrame({"patient_id": pid_te, "label": y_te.astype(int), "prob_tts": pte.astype(float)})
    df_pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    summ = {
        "cv_roc_mean": float(df_cv["roc_auc"].mean()) if not df_cv.empty else float("nan"),
        "cv_roc_std": float(df_cv["roc_auc"].std(ddof=0)) if not df_cv.empty else float("nan"),
        "cv_pr_mean": float(df_cv["pr_auc"].mean()) if not df_cv.empty else float("nan"),
        "cv_pr_std": float(df_cv["pr_auc"].std(ddof=0)) if not df_cv.empty else float("nan"),
        "test_roc_auc": roc,
        "test_pr_auc": pr,
        "test_roc_ci": roc_q,
        "test_pr_ci": pr_q,
        "coef_mean_dist_upper_scaled_space": float(df_coef.loc[df_coef["feature"] == "mean_dist_upper", "coef"].iloc[0]),
    }
    Path(os.path.join(out_dir, "summary.txt")).write_text(str(summ) + "\n", encoding="utf-8")
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()

