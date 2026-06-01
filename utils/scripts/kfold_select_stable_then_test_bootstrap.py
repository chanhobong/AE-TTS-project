#!/usr/bin/env python3
"""
Train+val K-fold selection (small grids) -> pick stable configs -> test once + bootstrap.

Scope
-----
Fixed feature: pooling="std" on patient-level embeddings (512-d).
Latent sources: pass multiple --latent_source NAME NPZ_DIR

Models/grids (same spirit as repeated_head_sweep_std.py)
-------------------------------------------------------
- logistic_l2: C ∈ {0.01,0.1,1,10}
- logistic_l1: C ∈ {0.01,0.1,1,10}
- logistic_en: C ∈ {0.01,0.1,1,10}, l1_ratio ∈ {0.2,0.5,0.8}
- linear_svc: C ∈ {0.01,0.1,1,10}

Selection rule
--------------
Compute per (latent_source, config) fold metrics on train+val only.
Then for each latent_source output:
  - best_mean_roc: config with highest cv ROC mean
  - min_roc_std:   config with lowest cv ROC std (tie-break by higher mean)
  - min_pr_std:    config with lowest cv PR std  (tie-break by higher mean)

Then refit each selected config on full train+val and evaluate on held-out test once.
Bootstrap CIs (default n=1000) on the test set.

Outputs
-------
out_dir/
  cv_metrics.csv
  cv_summary.csv
  selected_configs.csv
  test_metrics_selected.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
class DS:
    patient_id: np.ndarray
    y: np.ndarray
    X: np.ndarray


def _build_ds(npz_dir: str, labels_csv: list[str]) -> DS:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    ds = _load_features(os.path.abspath(npz_dir), clinical, pooling="std")
    return DS(patient_id=ds.patient_ids.astype(str), y=ds.y.astype(np.int64), X=ds.X.astype(np.float64))


def _grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        out.append({"model": "logistic_l2", "C": C})
    for C in [0.01, 0.1, 1.0, 10.0]:
        out.append({"model": "logistic_l1", "C": C})
    for C in [0.01, 0.1, 1.0, 10.0]:
        for r in [0.2, 0.5, 0.8]:
            out.append({"model": "logistic_en", "C": C, "l1_ratio": r})
    for C in [0.01, 0.1, 1.0, 10.0]:
        out.append({"model": "linear_svc", "C": C})
    return out


def _fit(kind: str, params: dict[str, Any], seed: int):
    if kind == "logistic_l2":
        return LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            penalty="l2",
            C=float(params["C"]),
        )
    if kind == "logistic_l1":
        return LogisticRegression(
            max_iter=12000,
            class_weight="balanced",
            random_state=seed,
            solver="saga",
            penalty="l1",
            C=float(params["C"]),
        )
    if kind == "logistic_en":
        return LogisticRegression(
            max_iter=12000,
            class_weight="balanced",
            random_state=seed,
            solver="saga",
            penalty="elasticnet",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
        )
    if kind == "linear_svc":
        return LinearSVC(
            max_iter=30000, class_weight="balanced", random_state=seed, dual=False, C=float(params["C"])
        )
    raise ValueError(kind)


def _scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _cfg_id(cfg: dict[str, Any]) -> str:
    m = str(cfg["model"])
    if m in ("logistic_l2", "logistic_l1", "linear_svc"):
        return f"{m}_C{cfg['C']}"
    if m == "logistic_en":
        return f"{m}_C{cfg['C']}_r{cfg['l1_ratio']}"
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="K-fold select stable configs -> test once + bootstrap")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--latent_source", action="append", nargs=2, metavar=("NAME", "NPZ_DIR"), required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_bootstrap_n", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "train.csv"))
    va_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "val.csv"))
    te_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "test.csv"))
    tv_set = tr_ids | va_ids

    grid = _grid()
    cv_rows: list[dict[str, Any]] = []
    sel_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for src_name, npz_dir in args.latent_source:
        src_name = str(src_name)
        ds = _build_ds(str(npz_dir), list(args.labels_csv))
        tv_mask = np.array([p in tv_set for p in ds.patient_id], dtype=bool)
        te_mask = np.array([p in te_ids for p in ds.patient_id], dtype=bool)
        if not np.any(tv_mask) or not np.any(te_mask):
            print(f"[{src_name}] skip: missing aligned train+val or test patients.")
            continue

        X_tv, y_tv = ds.X[tv_mask], ds.y[tv_mask]
        X_te, y_te = ds.X[te_mask], ds.y[te_mask]
        if np.unique(y_tv).size < 2 or np.unique(y_te).size < 2:
            print(f"[{src_name}] skip: need both classes in train+val and test.")
            continue

        skf = StratifiedKFold(n_splits=int(args.k), shuffle=True, random_state=int(args.seed))

        for cfg in grid:
            cfg_id = _cfg_id(cfg)
            for fold, (tr, va) in enumerate(skf.split(X_tv, y_tv)):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_tv[tr])
                Xva = sc.transform(X_tv[va])
                ytr, yva = y_tv[tr], y_tv[va]
                if np.unique(yva).size < 2:
                    continue
                model = _fit(str(cfg["model"]), cfg, seed=int(args.seed))
                model.fit(Xtr, ytr)
                s = _scores(model, Xva)
                cv_rows.append(
                    {
                        "latent_source": src_name,
                        "pooling": "std",
                        "config": cfg_id,
                        "model_family": str(cfg["model"]),
                        "k": int(args.k),
                        "fold": int(fold),
                        "roc_auc": float(roc_auc_score(yva, s)),
                        "pr_auc": float(average_precision_score(yva, s)),
                    }
                )

        df_cv_src = pd.DataFrame([r for r in cv_rows if r["latent_source"] == src_name])
        if df_cv_src.empty:
            print(f"[{src_name}] skip: no CV rows.")
            continue

        df_sum = (
            df_cv_src.groupby(["config", "model_family"], as_index=False)
            .agg(
                cv_roc_mean=("roc_auc", "mean"),
                cv_roc_std=("roc_auc", "std"),
                cv_pr_mean=("pr_auc", "mean"),
                cv_pr_std=("pr_auc", "std"),
                n_folds=("fold", "nunique"),
            )
            .sort_values(["cv_roc_mean", "cv_roc_std"], ascending=[False, True])
        )

        # selections
        best_mean = df_sum.iloc[0]
        min_roc_std = df_sum.sort_values(["cv_roc_std", "cv_roc_mean"], ascending=[True, False]).iloc[0]
        min_pr_std = df_sum.sort_values(["cv_pr_std", "cv_pr_mean"], ascending=[True, False]).iloc[0]
        picks = [
            ("best_mean_roc", best_mean),
            ("min_roc_std", min_roc_std),
            ("min_pr_std", min_pr_std),
        ]
        for tag, row in picks:
            sel_rows.append(
                {
                    "latent_source": src_name,
                    "pooling": "std",
                    "selection": tag,
                    "config": str(row["config"]),
                    "model_family": str(row["model_family"]),
                    "cv_roc_mean": float(row["cv_roc_mean"]),
                    "cv_roc_std": float(row["cv_roc_std"]),
                    "cv_pr_mean": float(row["cv_pr_mean"]),
                    "cv_pr_std": float(row["cv_pr_std"]),
                    "cv_n_folds": int(row["n_folds"]),
                }
            )

        # evaluate selected on test
        for tag, row in picks:
            cfg_id = str(row["config"])
            # map back to cfg params by parsing id (simple approach: search original grid)
            cfg_match = None
            for cfg in grid:
                if _cfg_id(cfg) == cfg_id:
                    cfg_match = cfg
                    break
            if cfg_match is None:
                continue
            sc = StandardScaler()
            Xtr_full = sc.fit_transform(X_tv)
            Xte = sc.transform(X_te)
            model = _fit(str(cfg_match["model"]), cfg_match, seed=int(args.seed))
            model.fit(Xtr_full, y_tv)
            s = _scores(model, Xte)
            roc = float(roc_auc_score(y_te, s))
            pr = float(average_precision_score(y_te, s))
            roc_q, pr_q = _bootstrap_roc_pr(y_te, s, n_boot=int(args.test_bootstrap_n), seed=int(args.seed))
            test_rows.append(
                {
                    "latent_source": src_name,
                    "pooling": "std",
                    "selection": tag,
                    "config": cfg_id,
                    "model_family": str(cfg_match["model"]),
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
            )

    pd.DataFrame(cv_rows).to_csv(out_dir / "cv_metrics.csv", index=False)
    if cv_rows:
        df_cv = pd.DataFrame(cv_rows)
        df_cv_sum = (
            df_cv.groupby(["latent_source", "pooling", "config", "model_family"], as_index=False)
            .agg(
                cv_roc_mean=("roc_auc", "mean"),
                cv_roc_std=("roc_auc", "std"),
                cv_pr_mean=("pr_auc", "mean"),
                cv_pr_std=("pr_auc", "std"),
                n_folds=("fold", "nunique"),
            )
            .sort_values(["latent_source", "cv_roc_std", "cv_roc_mean"], ascending=[True, True, False])
        )
        df_cv_sum.to_csv(out_dir / "cv_summary.csv", index=False)

    pd.DataFrame(sel_rows).to_csv(out_dir / "selected_configs.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test_metrics_selected.csv", index=False)
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()

