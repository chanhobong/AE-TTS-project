#!/usr/bin/env python3
"""
RBF-SVC: tune once on train+val, then repeated StratifiedShuffleSplit (same protocol as
repeated_stratified_shuffle_eval.py) — parallel to tune_logistic_std_repeat_compare.py.

1) GridSearchCV( Pipeline(StandardScaler, SVC(RBF)), roc_auc ) on **train ∪ val** only
   (no test patients in tuning).
2) Run repeated eval twice:
   - **default**: --svc_C 1 --svc_gamma scale (sklearn-style baseline),
   - **tuned**: best C, gamma from step (1).
3) Writes merged comparison CSV + tune metadata JSON.

Example (MONAI, cluster_hist K=64):
  python3 utils/scripts/tune_rbf_svc_repeated_shuffle.py \\
    --npz_dir .../diff3dformer_stageB_monai_ae/patients \\
    --train_csv data/train.csv --val_csv data/val.csv \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --latent_source monai_ae --pooling cluster_hist --cluster_hist_k 64 \\
    --out_dir latent_data/out_rbf_tune_compare
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import (  # noqa: E402
    _load_features,
    _merge_labels_and_clinical,
)
from tune_logistic_std_repeat_compare import (  # noqa: E402
    _merge_fixed_into_summary,
    _pick_baseline_row,
    _strip_unnamed_columns,
    _stratified_cv_folds,
)


def _parse_c_grid(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    if not out:
        raise ValueError("Empty --C_grid")
    return out


def _parse_gamma_grid(s: str) -> list[float | str]:
    out: list[float | str] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in ("scale", "auto"):
            out.append(p)
        else:
            out.append(float(p))
    if not out:
        raise ValueError("Empty --gamma_grid")
    return out


def _tune_rbf_svc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C_list: list[float],
    gamma_list: list[float | str],
    inner_cv_splits: int,
    grid_n_jobs: int,
    random_state: int = 42,
) -> tuple[dict[str, Any], float]:
    param_grid = {"svc__C": C_list, "svc__gamma": gamma_list}
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    cv = _stratified_cv_folds(y, max_splits=inner_cv_splits)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=int(grid_n_jobs),
        refit=True,
        error_score="raise",
    )
    grid.fit(X, y)
    bc = grid.best_estimator_.named_steps["svc"]
    g = bc.gamma
    best = {"C": float(bc.C), "gamma": g if isinstance(g, str) else float(g)}
    return best, float(grid.best_score_)


def _run_repeated_rbf(
    *,
    script: Path,
    python: str,
    out_dir: str,
    run_tag: str,
    npz_dir: str,
    latent_name: str,
    pooling: str,
    cluster_hist_k: int | None,
    labels_csv: list[str],
    fixed_test_csv: str | None,
    n_splits: int,
    test_size: float,
    random_state: int,
    svc_C: float,
    svc_gamma: str,
) -> Path:
    cmd = [
        python,
        str(script),
        "--pooling",
        pooling,
        "--classifier",
        "rbf_svc",
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
        "--svc_C",
        str(svc_C),
        "--svc_gamma",
        str(svc_gamma),
    ]
    if cluster_hist_k is not None:
        cmd.extend(["--cluster_hist_k", str(int(cluster_hist_k))])
    for p in labels_csv:
        cmd.extend(["--labels_csv", p])
    if fixed_test_csv:
        cmd.extend(["--fixed_test_csv", fixed_test_csv])
    subprocess.check_call(cmd)
    return Path(out_dir) / f"run_{run_tag}"


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    ap = argparse.ArgumentParser(description="RBF-SVC: tune on train+val, then default vs tuned repeated shuffle eval.")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--train_csv", default=str(repo_root / "data" / "train.csv"))
    ap.add_argument("--val_csv", default=str(repo_root / "data" / "val.csv"))
    ap.add_argument("--labels_csv", action="append", required=True, help="Typically train, val, test (for repeated eval).")
    ap.add_argument("--latent_source", default="monai_ae")
    ap.add_argument("--pooling", choices=["mean", "std", "mean_std", "cluster_hist"], required=True)
    ap.add_argument("--cluster_hist_k", type=int, default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--default_svc_C",
        type=float,
        default=1.0,
        help="Baseline repeated eval RBF C (matches former sklearn default).",
    )
    ap.add_argument(
        "--default_svc_gamma",
        type=str,
        default="scale",
        help="Baseline repeated eval gamma (scale|auto|float string).",
    )
    ap.add_argument(
        "--C_grid",
        type=str,
        default="0.001,0.01,0.1,1,10,100,1000",
    )
    ap.add_argument(
        "--gamma_grid",
        type=str,
        default="0.001,0.01,0.1,1,10,scale,auto",
    )
    ap.add_argument("--inner_cv_splits", type=int, default=5)
    ap.add_argument("--grid_n_jobs", type=int, default=1)
    ap.add_argument(
        "--baseline_merged_csv",
        default=None,
        help="If set and has latent_source/pooling/model=rbf_svc row, use as default row (skip default repeat run).",
    )
    ap.add_argument("--fixed_test_csv", default=None)
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if args.cluster_hist_k is not None and str(args.pooling) != "cluster_hist":
        raise SystemExit("--cluster_hist_k only valid with --pooling cluster_hist")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clinical_tune = _merge_labels_and_clinical(
        [os.path.abspath(args.train_csv), os.path.abspath(args.val_csv)]
    )
    ch_k = int(args.cluster_hist_k) if args.cluster_hist_k is not None else None
    ds_tune = _load_features(
        os.path.abspath(args.npz_dir),
        clinical_tune,
        pooling=str(args.pooling),
        cluster_hist_k=ch_k,
    )
    if np.unique(ds_tune.y).size < 2:
        raise RuntimeError("Tuning set needs both classes after NPZ join.")

    C_list = _parse_c_grid(str(args.C_grid))
    gamma_list = _parse_gamma_grid(str(args.gamma_grid))
    best_params, cv_mean_roc = _tune_rbf_svc(
        ds_tune.X,
        ds_tune.y,
        C_list=C_list,
        gamma_list=gamma_list,
        inner_cv_splits=int(args.inner_cv_splits),
        grid_n_jobs=int(args.grid_n_jobs),
    )

    meta = {
        "tune_scope": "train_plus_val_only",
        "tune_cv_metric": "roc_auc",
        "tune_cv_mean_roc_auc": cv_mean_roc,
        "tune_best": best_params,
        "param_grid": {"C": C_list, "gamma": [g if isinstance(g, str) else float(g) for g in gamma_list]},
        "latent_source": str(args.latent_source),
        "pooling": str(args.pooling),
        "cluster_hist_k": ch_k,
    }
    meta_path = Path(out_dir) / f"rbf_svc_tune_meta_{args.latent_source}_{args.pooling}_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("[tune] wrote", meta_path)
    print("[tune] best:", best_params, "cv mean roc_auc:", cv_mean_roc)

    script = here / "repeated_stratified_shuffle_eval.py"
    if not script.exists():
        raise FileNotFoundError(script)

    g_tune = best_params["gamma"]
    gamma_str = str(g_tune) if isinstance(g_tune, str) else str(float(g_tune))

    default_row: pd.DataFrame | None = None
    if args.baseline_merged_csv and Path(args.baseline_merged_csv).is_file():
        base = _strip_unnamed_columns(pd.read_csv(args.baseline_merged_csv))
        sel = _pick_baseline_row(base, args.latent_source, args.pooling, "rbf_svc")
        if len(sel) == 1:
            default_row = sel.copy()
            default_row["variant"] = "default_from_merged_csv"
            print("[baseline] using row from", args.baseline_merged_csv)
        elif len(sel) > 1:
            print("[baseline] multiple rows; using first.", file=sys.stderr)
            default_row = sel.iloc[[0]].copy()
            default_row["variant"] = "default_from_merged_csv"

    if default_row is None:
        tag_def = f"{ts}_{args.latent_source}_{args.pooling}_rbf_default"
        run_root_def = _run_repeated_rbf(
            script=script,
            python=args.python,
            out_dir=out_dir,
            run_tag=tag_def,
            npz_dir=os.path.abspath(args.npz_dir),
            latent_name=args.latent_source,
            pooling=str(args.pooling),
            cluster_hist_k=ch_k,
            labels_csv=[os.path.abspath(p) for p in args.labels_csv],
            fixed_test_csv=os.path.abspath(args.fixed_test_csv) if args.fixed_test_csv else None,
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.random_state,
            svc_C=float(args.default_svc_C),
            svc_gamma=str(args.default_svc_gamma),
        )
        df_def = _merge_fixed_into_summary(run_root_def)
        default_row = df_def.copy()
        default_row["variant"] = f"default_C{args.default_svc_C}_gamma{args.default_svc_gamma}"

    tag_tuned = f"{ts}_{args.latent_source}_{args.pooling}_rbf_tuned"
    run_root_tuned = _run_repeated_rbf(
        script=script,
        python=args.python,
        out_dir=out_dir,
        run_tag=tag_tuned,
        npz_dir=os.path.abspath(args.npz_dir),
        latent_name=args.latent_source,
        pooling=str(args.pooling),
        cluster_hist_k=ch_k,
        labels_csv=[os.path.abspath(p) for p in args.labels_csv],
        fixed_test_csv=os.path.abspath(args.fixed_test_csv) if args.fixed_test_csv else None,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
        svc_C=float(best_params["C"]),
        svc_gamma=gamma_str,
    )
    df_tuned = _merge_fixed_into_summary(run_root_tuned)
    tuned_row = df_tuned.copy()
    tuned_row["variant"] = "tuned_gridsearch_trainval"
    tuned_row["tune_cv_mean_roc_auc"] = cv_mean_roc
    tuned_row["tune_best_C"] = best_params["C"]
    tuned_row["tune_best_gamma"] = best_params["gamma"]

    combo = pd.concat([default_row, tuned_row], ignore_index=True)
    combo = _strip_unnamed_columns(combo)
    out_csv = Path(out_dir) / f"repeated_{args.latent_source}_{args.pooling}_rbf_svc_default_vs_tuned_{ts}.csv"
    combo.to_csv(out_csv, index=False)
    print("[out]", out_csv)


if __name__ == "__main__":
    main()
