#!/usr/bin/env python3
"""
1) Stratified GridSearchCV (roc_auc) on train+val only — Pipeline(StandardScaler, LogisticRegression).
2) Repeated StratifiedShuffleSplit eval (same protocol as repeated_stratified_shuffle_eval.py)
   with default sklearn-style logistic (C=1, l2, lbfgs) and with tuned hyperparameters.
3) Writes one CSV with both rows + tuning metadata for comparison.

Example (plain AE, std pooling):
  python utils/scripts/tune_logistic_std_repeat_compare.py \\
    --npz_dir .../diff3dformer_stageB_plain_ae/patients \\
    --train_csv data/train.csv --val_csv data/val.csv \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --out_dir latent_data/out_repeated_from_stable \\
    --baseline_merged_csv latent_data/out_repeated_from_stable/repeated_summary_merged_with_fixed.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import (  # noqa: E402
    _load_features,
    _merge_labels_and_clinical,
)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    return df[keep].copy()


def _pick_baseline_row(df: pd.DataFrame, latent_source: str, pooling: str, model: str) -> pd.DataFrame:
    m = (
        (df["latent_source"].astype(str) == latent_source)
        & (df["pooling"].astype(str) == pooling)
        & (df["model"].astype(str) == model)
    )
    return df.loc[m].copy()


def _merge_fixed_into_summary(run_root: Path) -> pd.DataFrame:
    sums: list[pd.DataFrame] = []
    for p in sorted(run_root.rglob("summary.csv")):
        sums.append(pd.read_csv(p))
    if not sums:
        raise FileNotFoundError(f"No summary.csv under {run_root}")
    df_sum = pd.concat(sums, ignore_index=True)
    fx = run_root / "compare_fixed_test_positions.csv"
    if not fx.exists():
        return df_sum
    df_fx = pd.read_csv(fx)
    keys = ["latent_source", "pooling", "model"]
    roc_part = (
        df_fx.dropna(subset=["fixed_test_roc_auc"])[keys + ["fixed_test_roc_auc", "cdf_position"]]
        .drop_duplicates(keys)
    )
    pr_part = (
        df_fx.dropna(subset=["fixed_test_pr_auc"])[keys + ["fixed_test_pr_auc", "pr_cdf_position"]]
        .drop_duplicates(keys)
    )
    df_fixed = roc_part.merge(pr_part, on=keys, how="outer")
    return df_sum.merge(df_fixed, on=keys, how="left")


def _stratified_cv_folds(y: np.ndarray, max_splits: int = 5) -> StratifiedKFold:
    counts = np.bincount(y.astype(int))
    pos, neg = int(counts[1]) if len(counts) > 1 else 0, int(counts[0])
    m = min(pos, neg)
    n_splits = max(2, min(max_splits, m))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def _tune_logistic(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[dict, float]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=20000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    param_grid: list[dict] = [
        {"clf__penalty": ["l2"], "clf__C": [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0], "clf__solver": ["lbfgs"]},
        {"clf__penalty": ["l1"], "clf__C": [1e-3, 1e-2, 0.1, 1.0, 10.0], "clf__solver": ["saga"]},
        {
            "clf__penalty": ["elasticnet"],
            "clf__solver": ["saga"],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.2, 0.5, 0.8],
        },
    ]
    cv = _stratified_cv_folds(y)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    grid.fit(X, y)
    best = grid.best_estimator_.named_steps["clf"]
    params = {
        "C": float(best.C),
        "penalty": str(best.penalty),
        "solver": str(best.solver),
    }
    if best.penalty == "elasticnet":
        params["l1_ratio"] = float(best.l1_ratio)
    return params, float(grid.best_score_)


def _run_repeated(
    *,
    script: Path,
    python: str,
    out_dir: str,
    run_tag: str,
    npz_dir: str,
    latent_name: str,
    pooling: str,
    labels_csv: list[str],
    fixed_test_csv: str | None,
    n_splits: int,
    test_size: float,
    random_state: int,
    logistic_C: float,
    logistic_penalty: str,
    logistic_solver: str,
    logistic_l1_ratio: float | None,
) -> Path:
    cmd = [
        python,
        str(script),
        "--pooling",
        pooling,
        "--classifier",
        "logistic",
        "--n_splits",
        str(n_splits),
        "--test_size",
        str(test_size),
        "--random_state",
        str(random_state),
        "--out_dir",
        out_dir,
        "--run_tag",
        run_tag,
        "--latent_source",
        latent_name,
        npz_dir,
        "--logistic_C",
        str(logistic_C),
        "--logistic_penalty",
        logistic_penalty,
        "--logistic_solver",
        logistic_solver,
    ]
    if logistic_l1_ratio is not None:
        cmd.extend(["--logistic_l1_ratio", str(logistic_l1_ratio)])
    for p in labels_csv:
        cmd.extend(["--labels_csv", p])
    if fixed_test_csv:
        cmd.extend(["--fixed_test_csv", fixed_test_csv])
    subprocess.check_call(cmd)
    return Path(out_dir) / f"run_{run_tag}"


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="StageB patients dir (plain_ae, ...).")
    ap.add_argument("--train_csv", default=str(repo_root / "data" / "train.csv"))
    ap.add_argument("--val_csv", default=str(repo_root / "data" / "val.csv"))
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--pooling", default="std")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--baseline_merged_csv",
        default=None,
        help="If set and contains plain_ae/std/logistic row, use it as the default row (skip default repeat).",
    )
    ap.add_argument(
        "--fixed_test_csv",
        default=None,
        help="Optional; passed to repeated eval for CDF vs fixed split (e.g. test_metrics_stable_minstd.csv).",
    )
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clinical_tune = _merge_labels_and_clinical(
        [os.path.abspath(args.train_csv), os.path.abspath(args.val_csv)]
    )
    ds_tune = _load_features(os.path.abspath(args.npz_dir), clinical_tune, pooling=str(args.pooling))
    if np.unique(ds_tune.y).size < 2:
        raise RuntimeError("Tuning set needs both classes after NPZ join.")

    best_params, cv_mean_roc = _tune_logistic(ds_tune.X, ds_tune.y)
    meta = {
        "tune_scope": "train_plus_val_only",
        "tune_cv_metric": "roc_auc",
        "tune_cv_mean_roc_auc": cv_mean_roc,
        "tune_best": best_params,
    }
    meta_path = Path(out_dir) / f"logistic_tune_meta_{args.latent_source}_{args.pooling}_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[tune] wrote", meta_path)
    print("[tune] best:", best_params, "cv mean roc_auc:", cv_mean_roc)

    script = here / "repeated_stratified_shuffle_eval.py"
    if not script.exists():
        raise FileNotFoundError(script)

    l1_ratio = best_params.get("l1_ratio")
    if l1_ratio is not None:
        l1_ratio = float(l1_ratio)

    # --- default row: from file or second repeated run
    default_row: pd.DataFrame | None = None
    if args.baseline_merged_csv and Path(args.baseline_merged_csv).is_file():
        base = _strip_unnamed_columns(pd.read_csv(args.baseline_merged_csv))
        sel = _pick_baseline_row(base, args.latent_source, args.pooling, "logistic")
        if len(sel) == 1:
            default_row = sel.copy()
            default_row["variant"] = "default_from_merged_csv"
            print("[baseline] using row from", args.baseline_merged_csv)
        elif len(sel) > 1:
            print("[baseline] multiple rows in merged CSV; using first.", file=sys.stderr)
            default_row = sel.iloc[[0]].copy()
            default_row["variant"] = "default_from_merged_csv"

    if default_row is None:
        tag_def = f"{ts}_{args.latent_source}_{args.pooling}_logistic_defaultC1"
        run_root_def = _run_repeated(
            script=script,
            python=args.python,
            out_dir=out_dir,
            run_tag=tag_def,
            npz_dir=os.path.abspath(args.npz_dir),
            latent_name=args.latent_source,
            pooling=args.pooling,
            labels_csv=[os.path.abspath(p) for p in args.labels_csv],
            fixed_test_csv=os.path.abspath(args.fixed_test_csv) if args.fixed_test_csv else None,
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.random_state,
            logistic_C=1.0,
            logistic_penalty="l2",
            logistic_solver="lbfgs",
            logistic_l1_ratio=None,
        )
        df_def = _merge_fixed_into_summary(run_root_def)
        default_row = df_def.copy()
        default_row["variant"] = "default_C1_l2_lbfgs"

    tag_tuned = f"{ts}_{args.latent_source}_{args.pooling}_logistic_tuned"
    run_root_tuned = _run_repeated(
        script=script,
        python=args.python,
        out_dir=out_dir,
        run_tag=tag_tuned,
        npz_dir=os.path.abspath(args.npz_dir),
        latent_name=args.latent_source,
        pooling=args.pooling,
        labels_csv=[os.path.abspath(p) for p in args.labels_csv],
        fixed_test_csv=os.path.abspath(args.fixed_test_csv) if args.fixed_test_csv else None,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
        logistic_C=float(best_params["C"]),
        logistic_penalty=str(best_params["penalty"]),
        logistic_solver=str(best_params["solver"]),
        logistic_l1_ratio=l1_ratio if str(best_params["penalty"]) == "elasticnet" else None,
    )
    df_tuned = _merge_fixed_into_summary(run_root_tuned)
    tuned_row = df_tuned.copy()
    tuned_row["variant"] = "tuned_gridsearch_trainval"
    tuned_row["tune_cv_mean_roc_auc"] = cv_mean_roc
    tuned_row["tune_best_C"] = best_params["C"]
    tuned_row["tune_best_penalty"] = best_params["penalty"]
    tuned_row["tune_best_solver"] = best_params["solver"]
    if "l1_ratio" in best_params:
        tuned_row["tune_best_l1_ratio"] = best_params["l1_ratio"]

    combo = pd.concat([default_row, tuned_row], ignore_index=True)
    combo = _strip_unnamed_columns(combo)
    out_csv = Path(out_dir) / f"repeated_plain_ae_std_logistic_default_vs_tuned_{ts}.csv"
    combo.to_csv(out_csv, index=False)
    print("[out]", out_csv)


if __name__ == "__main__":
    main()
