#!/usr/bin/env python3
"""
Repeated head-sweep on patient-level latent features (plain_ae / monai_ae).

What it does
------------
For each latent_source NPZ dir:
  - Build patient-level X from StageB `patients/*.npz` (or recursive scan with
    ``--recursive_npz``), controlled by ``--pooling``:
      * ``std``: per-slice std along T (same as before; 512-d)
      * ``p90_p10``: P90−P10 along ordered slice axis (needs mask / slice_z_mm
        as in ``trajectory_volatility_classifier_cv``; 512-d)
      * ``vol_mm`` / ``global_mean_vol_mm``: mean over transitions of |Δz|/Δmm (512-d)
      * ``p90_vol_mm``: dim-wise 90th percentile of transition rates (512-d)
      * ``regional_vol_mm``: basal | mid | apical thirds on transition midpoint
        ``(pos_norm[t]+pos_norm[t+1])/2``; concat 3×512-d
      * ``p90_p10_vol_mm``, ``p90_p10_global_mean_vol_mm``, ``p90_p10_p90_vol_mm``,
        ``p90_p10_regional_vol_mm``: concat blocks (latter is 512+1536)
  - Run Repeated StratifiedShuffleSplit (default 100 splits)
  - For each classifier head + small hyperparam grid:
      * Fit StandardScaler on TRAIN only
      * Fit model on TRAIN only
      * Evaluate ROC-AUC / PR-AUC on TEST only
  - Save into run_* folder structure compatible with our existing merge script:
      run_<tag>/<latent_source>/<pooling>/<head_id>/repeats.csv
      run_<tag>/<latent_source>/<pooling>/<head_id>/summary.csv
      run_<tag>/<latent_source>/<pooling>/<head_id>/hist_roc_auc_<head_id>.png
      run_<tag>/<latent_source>/<pooling>/<head_id>/hist_pr_auc_<head_id>.png

Then it optionally merges all run_* under out_dir into:
  repeated_summary_merged_with_fixed.csv
using `merge_repeated_runs_to_merged_csv.py`.

Heads included (minimal, small grids)
-------------------------------------
- logistic_l2: LogisticRegression(penalty="l2", solver="lbfgs"), C ∈ {0.01,0.1,1,10}
- logistic_l1: penalty="l1", solver="saga", C ∈ {0.01,0.1,1,10}
- logistic_en: penalty="elasticnet", solver="saga",
               C ∈ {0.01,0.1,1,10}, l1_ratio ∈ {0.2,0.5,0.8}
- linear_svc: LinearSVC, C ∈ {0.01,0.1,1,10}
- calib_linear_svc (optional): CalibratedClassifierCV(LinearSVC), C ∈ {0.1,1,10}
- mlp_small (optional): MLPClassifier(16 hidden units), alpha ∈ {1e-4,1e-3,1e-2}

Notes
-----
- Calibration & MLP can be slow / unstable on small data. Keep optional.
- This script is for *distribution / stability* analysis (repeated splits), not “best score hunting”.
- **Parallel CPU:** ``--n_jobs -1`` runs each repeated CV split in a separate worker (joblib loky).
  Each classifier *head* still runs splits sequentially in parallel batches; different heads run one after another.
  If you also launch multiple sweep processes (e.g. several poolings in parallel shells), reduce ``--n_jobs``
  to avoid RAM/CPU oversubscription.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None  # type: ignore[misc, assignment]
    delayed = None  # type: ignore[misc, assignment]

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import (  # noqa: E402
    _discover_npz_files,
    _load_features,
    _merge_labels_and_clinical,
    _sex_to_bin,
)
import trajectory_volatility_classifier_cv as traj_cv  # noqa: E402


HEAD_SWEEP_TRAJECTORY_POOLINGS: tuple[str, ...] = (
    "p90_p10",
    "vol_mm",
    "global_mean_vol_mm",
    "p90_vol_mm",
    "regional_vol_mm",
    "p90_p10_vol_mm",
    "p90_p10_global_mean_vol_mm",
    "p90_p10_p90_vol_mm",
    "p90_p10_regional_vol_mm",
)


@dataclass
class DS:
    patient_id: np.ndarray
    y: np.ndarray
    X: np.ndarray


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


def _scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _npz_paths(npz_dir: str, recursive: bool) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if recursive:
        return traj_cv._discover_npz_paths(npz_dir, True)
    return _discover_npz_files(npz_dir)


def _build_ds_trajectory(
    npz_dir: str,
    labels_csv: list[str],
    pooling: str,
    eps_mm: float,
    recursive_npz: bool,
) -> DS:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _npz_paths(npz_dir, recursive_npz)
    feats: list[np.ndarray] = []
    pids: list[str] = []
    ys: list[int] = []

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
        z = np.load(p, allow_pickle=True)
        try:
            emb, zmm, _ = traj_cv._ordered_rows(z)
        except (ValueError, KeyError, IndexError):
            continue
        if emb.shape[0] < 2:
            continue
        bl = traj_cv.trajectory_feature_blocks_extended(emb, zmm, eps_mm)
        if pooling == "p90_p10":
            v = bl["p90_p10"]
        elif pooling in ("vol_mm", "global_mean_vol_mm"):
            v = bl["global_mean_vol_mm"]
        elif pooling == "p90_vol_mm":
            v = bl["p90_vol_mm"]
        elif pooling == "regional_vol_mm":
            v = bl["regional_vol_mm"]
        elif pooling == "p90_p10_vol_mm":
            v = np.concatenate([bl["p90_p10"], bl["vol_mm"]], axis=0)
        elif pooling == "p90_p10_global_mean_vol_mm":
            v = np.concatenate([bl["p90_p10"], bl["global_mean_vol_mm"]], axis=0)
        elif pooling == "p90_p10_p90_vol_mm":
            v = np.concatenate([bl["p90_p10"], bl["p90_vol_mm"]], axis=0)
        elif pooling == "p90_p10_regional_vol_mm":
            v = np.concatenate([bl["p90_p10"], bl["regional_vol_mm"]], axis=0)
        else:
            raise ValueError(f"unknown trajectory pooling: {pooling!r}")
        if np.any(~np.isfinite(v)):
            continue
        feats.append(v.astype(np.float64))
        pids.append(pid)
        ys.append(int(row["label"]))

    if len(pids) < 5:
        raise RuntimeError(f"Too few patients after join / trajectory filter: {len(pids)}")
    return DS(
        patient_id=np.array(pids, dtype=str),
        y=np.array(ys, dtype=np.int64),
        X=np.stack(feats, axis=0),
    )


def _build_ds(
    npz_dir: str,
    labels_csv: list[str],
    pooling: str,
    eps_mm: float,
    recursive_npz: bool,
) -> DS:
    if pooling == "std":
        clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
        ds = _load_features(os.path.abspath(npz_dir), clinical, pooling="std")
        return DS(patient_id=ds.patient_ids.astype(str), y=ds.y.astype(np.int64), X=ds.X.astype(np.float64))
    return _build_ds_trajectory(npz_dir, labels_csv, pooling, eps_mm, recursive_npz)


def _grid() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        cfgs.append({"model": "logistic_l2", "C": C})
    for C in [0.01, 0.1, 1.0, 10.0]:
        cfgs.append({"model": "logistic_l1", "C": C})
    for C in [0.01, 0.1, 1.0, 10.0]:
        for r in [0.2, 0.5, 0.8]:
            cfgs.append({"model": "logistic_en", "C": C, "l1_ratio": r})
    for C in [0.01, 0.1, 1.0, 10.0]:
        cfgs.append({"model": "linear_svc", "C": C})
    # optional models appended later
    return cfgs


def _fit(kind: str, params: dict[str, Any], seed: int) -> Any:
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
    if kind == "calib_linear_svc":
        base = LinearSVC(
            max_iter=30000, class_weight="balanced", random_state=seed, dual=False, C=float(params["C"])
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if kind == "mlp_small":
        return MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            alpha=float(params["alpha"]),
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    raise ValueError(kind)


def _one_repeat_row(
    repeat_id: int,
    tr: np.ndarray,
    te: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    patient_id: np.ndarray,
    kind: str,
    cfg: dict[str, Any],
    random_state: int,
    src_name: str,
    pooling_tag: str,
    head_id: str,
    n_splits: int,
    test_size: float,
    split_random_state: int,
    write_oos: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    Xtr_raw, Xte_raw = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    pid_te = patient_id[te]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr_raw)
    Xte = sc.transform(Xte_raw)
    model = _fit(kind, cfg, seed=int(random_state))
    model.fit(Xtr, ytr)
    s_tr = _scores(model, Xtr)
    s_te = _scores(model, Xte)
    rep = {
        "latent_source": src_name,
        "pooling": pooling_tag,
        "model": head_id,
        "repeat_id": int(repeat_id),
        "train_size": int(len(tr)),
        "test_size": int(len(te)),
        "roc_auc": float(roc_auc_score(yte, s_te)),
        "pr_auc": float(average_precision_score(yte, s_te)),
        "train_roc_auc": float(roc_auc_score(ytr, s_tr)),
        "train_pr_auc": float(average_precision_score(ytr, s_tr)),
        "test_patient_ids": ";".join(pid_te.tolist()),
    }
    oos: list[dict[str, Any]] = []
    if write_oos:
        for i in range(int(len(te))):
            oos.append(
                {
                    "latent_source": src_name,
                    "pooling": pooling_tag,
                    "cluster_hist_bins": "",
                    "model": head_id,
                    "repeat_id": int(repeat_id),
                    "patient_id": str(pid_te[i]),
                    "label": int(yte[i]),
                    "prob_tts": float(s_te[i]),
                    "n_splits": int(n_splits),
                    "test_size": float(test_size),
                    "split_random_state": int(split_random_state),
                    "concat_age_sex": 0,
                    "final_rescale_after_concat": 0,
                }
            )
    return rep, oos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Repeated classifier head sweep on patient-level latent features (plain_ae / monai_ae)."
    )
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--latent_source", action="append", nargs=2, metavar=("NAME", "NPZ_DIR"), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument(
        "--pooling",
        choices=["std", *HEAD_SWEEP_TRAJECTORY_POOLINGS],
        default="std",
        help="Feature construction for X. Trajectory poolings match trajectory_volatility_classifier_cv.",
    )
    ap.add_argument(
        "--eps_mm",
        type=float,
        default=1e-3,
        help="Floor for |Δz| in mm when computing vol_mm (trajectory poolings only).",
    )
    ap.add_argument(
        "--recursive_npz",
        action="store_true",
        help="Discover *.npz recursively (trajectory poolings only; std uses one directory level).",
    )
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel repeated CV splits per head (joblib). 1 = sequential. Use -1 for all cores. "
        "Each job fits one train/test split; reduces wall time for CPU-bound sklearn.",
    )
    ap.add_argument("--include_calibrated_svm", action="store_true")
    ap.add_argument("--include_mlp", action="store_true")
    ap.add_argument("--merge_after", action="store_true", help="If set, produce repeated_summary_merged_with_fixed.csv.")
    ap.add_argument(
        "--only_head_id",
        default="",
        help=(
            "If set, run only the classifier head whose folder id matches exactly, "
            "e.g. logistic_en_C0.1_r0.2 (skips the rest of the grid)."
        ),
    )
    ap.add_argument(
        "--write_test_predictions",
        action="store_true",
        help="Write test_oos_predictions.csv (long table; same role as repeated_stratified_shuffle_eval).",
    )
    ap.add_argument(
        "--test_predictions_basename",
        default="test_oos_predictions.csv",
        help="Saved next to repeats.csv when --write_test_predictions is set.",
    )
    args = ap.parse_args()
    pooling_tag = str(args.pooling)
    n_jobs = int(args.n_jobs)
    if n_jobs != 1 and Parallel is None:
        print("[warn] joblib not installed; --n_jobs ignored (sequential).", file=sys.stderr)
        n_jobs = 1

    out_parent = Path(args.out_dir).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_parent / f"run_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid = _grid()
    if bool(args.include_calibrated_svm):
        for C in [0.1, 1.0, 10.0]:
            grid.append({"model": "calib_linear_svc", "C": C})
    if bool(args.include_mlp):
        for alpha in [1e-4, 1e-3, 1e-2]:
            grid.append({"model": "mlp_small", "alpha": alpha})

    for src_name, npz_dir in args.latent_source:
        src_name = str(src_name)
        ds = _build_ds(
            str(npz_dir),
            list(args.labels_csv),
            pooling_tag,
            float(args.eps_mm),
            bool(args.recursive_npz),
        )
        if np.unique(ds.y).size < 2:
            raise RuntimeError(f"[{src_name}] only one class present after join; cannot evaluate AUC.")

        splitter = StratifiedShuffleSplit(
            n_splits=int(args.n_splits),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
        )

        only_hid = str(args.only_head_id).strip()
        write_oos = bool(args.write_test_predictions)

        ran_any_matching_head = False
        for cfg in grid:
            kind = str(cfg["model"])
            # unique id for folder naming
            if kind in ("logistic_l2", "logistic_l1", "linear_svc", "calib_linear_svc"):
                head_id = f"{kind}_C{cfg['C']}"
            elif kind == "logistic_en":
                head_id = f"{kind}_C{cfg['C']}_r{cfg['l1_ratio']}"
            elif kind == "mlp_small":
                head_id = f"{kind}_a{cfg['alpha']}"
            else:
                head_id = kind

            if only_hid and head_id != only_hid:
                continue

            ran_any_matching_head = True
            splits = list(splitter.split(ds.X, ds.y))
            if n_jobs == 1:
                chunks = [
                    _one_repeat_row(
                        repeat_id,
                        tr,
                        te,
                        ds.X,
                        ds.y,
                        ds.patient_id,
                        kind,
                        cfg,
                        int(args.random_state),
                        src_name,
                        pooling_tag,
                        head_id,
                        int(args.n_splits),
                        float(args.test_size),
                        int(args.random_state),
                        write_oos,
                    )
                    for repeat_id, (tr, te) in enumerate(splits)
                ]
            else:
                chunks = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_one_repeat_row)(
                        repeat_id,
                        tr,
                        te,
                        ds.X,
                        ds.y,
                        ds.patient_id,
                        kind,
                        cfg,
                        int(args.random_state),
                        src_name,
                        pooling_tag,
                        head_id,
                        int(args.n_splits),
                        float(args.test_size),
                        int(args.random_state),
                        write_oos,
                    )
                    for repeat_id, (tr, te) in enumerate(splits)
                )
            rep_rows = sorted([c[0] for c in chunks], key=lambda r: int(r["repeat_id"]))
            pred_rows: list[dict[str, Any]] = []
            for c in chunks:
                pred_rows.extend(c[1])

            df_rep = pd.DataFrame(rep_rows)
            sub = out_root / src_name.replace("/", "_") / pooling_tag / head_id
            sub.mkdir(parents=True, exist_ok=True)
            df_rep.to_csv(sub / "repeats.csv", index=False)
            if write_oos:
                pd.DataFrame(pred_rows).to_csv(sub / str(args.test_predictions_basename), index=False)

            roc_s = _summary_stats(df_rep["roc_auc"].to_numpy())
            pr_s = _summary_stats(df_rep["pr_auc"].to_numpy())
            df_sum = pd.DataFrame(
                [
                    {
                        "latent_source": src_name,
                        "pooling": pooling_tag,
                        "model": head_id,
                        **{f"roc_{k}": v for k, v in roc_s.items()},
                        **{f"pr_{k}": v for k, v in pr_s.items()},
                    }
                ]
            )
            df_sum.to_csv(sub / "summary.csv", index=False)

            # hist plots
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.hist(df_rep["roc_auc"].to_numpy(), bins=20, color="tab:blue", alpha=0.85)
            ax.set_title(f"{src_name} / {pooling_tag} / {head_id} — ROC-AUC (n={args.n_splits})")
            ax.set_xlabel("ROC-AUC")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(sub / f"hist_roc_auc_{head_id}.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.hist(df_rep["pr_auc"].to_numpy(), bins=20, color="tab:green", alpha=0.85)
            ax.set_title(f"{src_name} / {pooling_tag} / {head_id} — PR-AUC (n={args.n_splits})")
            ax.set_xlabel("PR-AUC")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(sub / f"hist_pr_auc_{head_id}.png", dpi=150)
            plt.close(fig)

        if only_hid and not ran_any_matching_head:
            raise SystemExit(f"[{src_name}] --only_head_id={only_hid!r} matched no head in grid (check id / C / l1_ratio).")

    if bool(args.merge_after):
        # reuse existing merger
        import subprocess

        merger = _SCRIPTS_DIR / "merge_repeated_runs_to_merged_csv.py"
        subprocess.check_call([sys.executable, str(merger), "--out_parent", str(out_parent)])

    print("Wrote:", out_root)


if __name__ == "__main__":
    main()

