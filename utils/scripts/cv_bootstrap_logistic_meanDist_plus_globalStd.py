#!/usr/bin/env python3
"""
Plain AE: Logistic on 2 features
  X = [mean_dist_upper, global_std]

Where:
  - mean_dist_upper: per-patient mean of upper-triangle distances between cluster centroids
      from `cluster_pairwise_distance_matrix.py` -> per_patient_scalar.csv
  - global_std: scalar summary of within-patient std_vector from embeddings
      from `analyze_patient_std_variability.py` -> plain_ae_patient_std_scalars.csv

Protocol (matches our previous CV + bootstrap style)
----------------------------------------------------
1) Use fixed split_dir {train,val,test}. Unit is patient.
2) CV on train+val only with StratifiedKFold:
   - StandardScaler fit on fold-train only
   - LogisticRegression(class_weight="balanced") on fold-train only
   - Evaluate ROC-AUC / PR-AUC on fold-val only
3) Final model:
   - Fit StandardScaler on full train+val
   - Train logistic on full train+val
   - Evaluate once on held-out test
4) Bootstrap CI on test (n=1000): ROC-AUC & PR-AUC
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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


def _build_dataset(mean_dist_csv: str, global_std_csv: str) -> Dataset:
    a = pd.read_csv(mean_dist_csv)
    b = pd.read_csv(global_std_csv)
    a["patient_id"] = a["patient_id"].astype(str)
    b["patient_id"] = b["patient_id"].astype(str)

    # Choose global std scalar: L2 norm of std_vector (matches earlier l2_std usage)
    b = b.rename(columns={"l2_std": "global_std_l2", "mean_std": "global_std_mean"})
    need_a = {"patient_id", "label", "mean_dist_upper"}
    need_b = {"patient_id", "global_std_l2"}
    if not need_a.issubset(a.columns):
        raise ValueError(f"mean_dist_csv missing {sorted(need_a)}; got {list(a.columns)}")
    if not need_b.issubset(b.columns):
        raise ValueError(f"global_std_csv missing {sorted(need_b)}; got {list(b.columns)}")

    df = a[["patient_id", "label", "mean_dist_upper"]].merge(
        b[["patient_id", "global_std_l2", "global_std_mean"]], on="patient_id", how="inner"
    )
    df = df.dropna(subset=["mean_dist_upper", "global_std_l2"])
    if df.shape[0] < 10:
        raise RuntimeError(f"Too few merged patients: {df.shape[0]}")

    X = df[["mean_dist_upper", "global_std_l2"]].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    pid = df["patient_id"].to_numpy(dtype=str)
    return Dataset(patient_id=pid, y=y, X=X)


def main() -> None:
    ap = argparse.ArgumentParser(description="CV + bootstrap for logistic on [mean_dist_upper, global_std].")
    ap.add_argument("--split_dir", required=True, help="Directory containing train.csv/val.csv/test.csv")
    ap.add_argument("--mean_dist_csv", required=True, help="per_patient_scalar.csv from cluster_pairwise_distance_matrix.py")
    ap.add_argument("--global_std_csv", required=True, help="plain_ae_patient_std_scalars.csv from analyze_patient_std_variability.py")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_bootstrap_n", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ds = _build_dataset(os.path.abspath(args.mean_dist_csv), os.path.abspath(args.global_std_csv))
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

    df_test = pd.DataFrame(
        [
            {
                "feature_set": "mean_dist_upper + global_std_l2",
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

    # summary
    summ = {
        "cv_roc_mean": float(df_cv["roc_auc"].mean()) if not df_cv.empty else float("nan"),
        "cv_roc_std": float(df_cv["roc_auc"].std(ddof=0)) if not df_cv.empty else float("nan"),
        "cv_pr_mean": float(df_cv["pr_auc"].mean()) if not df_cv.empty else float("nan"),
        "cv_pr_std": float(df_cv["pr_auc"].std(ddof=0)) if not df_cv.empty else float("nan"),
        "test_roc_auc": roc,
        "test_pr_auc": pr,
        "test_roc_ci": roc_q,
        "test_pr_ci": pr_q,
    }
    Path(os.path.join(out_dir, "summary.txt")).write_text(str(summ) + "\n", encoding="utf-8")
    print("Wrote:", out_dir)


if __name__ == "__main__":
    from pathlib import Path

    main()

