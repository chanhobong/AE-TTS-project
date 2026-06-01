#!/usr/bin/env python3
"""
Repeated StratifiedShuffleSplit evaluation (patient-level).

Protocol (as requested)
-----------------------
Use StratifiedShuffleSplit:
  - n_splits = 100
  - test_size = 0.2
  - random_state = 42

For each split:
  1) Split patients into train/test (stratified).
  2) Fit StandardScaler on TRAIN only.
  3) Transform TEST using train-fitted scaler.
  4) Train classifier on TRAIN only.
  5) Evaluate on held-out TEST only.
  6) Save repeat_id, train_size, test_size, train/test ROC-AUC & PR-AUC, and patient IDs in test split.

Models
------
- LogisticRegression(class_weight="balanced")
- LinearSVC(class_weight="balanced")
- RBF SVC(class_weight="balanced", probability=True)

Features
--------
From StageB per-patient NPZ files:
- mean / std / mean_std from embeddings
- cluster_hist from cluster_ids (mask-aware if present); optional ``--cluster_hist_k`` fixes K
  (if K < inferred #bins, only slices with id < K; if K > inferred, zero-pad).

Outputs
-------
out_dir/
  repeats.csv
  summary.csv
  hist_roc_auc.png
  hist_pr_auc.png
  compare_fixed_test_positions.csv   (optional, if --fixed_test_csv provided)
  hist_*_with_fixed.png              (optional overlay lines)
  test_oos_predictions.csv          (optional, if --write_test_predictions; long table for ensemble)
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC


def _merge_labels_and_clinical(paths: list[str]) -> pd.DataFrame:
    """Merge CSVs; require patient_id/ID, label/case, age, sex (same as other utils)."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        id_col = "patient_id" if "patient_id" in df.columns else None
        if id_col is None:
            for c in ("ID", "id"):
                if c in df.columns:
                    id_col = c
                    break
        if id_col is None:
            raise ValueError(f"No patient id column in {p}")

        lab_col = None
        for c in ("label", "case", "Label", "TTS"):
            if c in df.columns:
                lab_col = c
                break
        if lab_col is None:
            raise ValueError(f"No label/case column in {p}")

        if "age" not in df.columns:
            raise ValueError(f"Column 'age' missing in {p}")
        if "sex" not in df.columns:
            raise ValueError(f"Column 'sex' missing in {p}")

        sub = df[[id_col, lab_col, "age", "sex"]].copy()
        sub.columns = ["patient_id", "label_raw", "age", "sex"]
        sub["patient_id"] = sub["patient_id"].astype(str)

        def to_bin(v) -> int:
            if isinstance(v, str):
                s = v.lower()
                if "tts" in s or "takotsubo" in s:
                    return 1
                if v.strip().isdigit():
                    return int(v.strip())
                return 0
            try:
                return int(v)
            except Exception:
                return int(float(v))

        sub["label"] = sub["label_raw"].map(to_bin)
        sub["age"] = pd.to_numeric(sub["age"], errors="coerce")
        sub["sex"] = sub["sex"].astype(str).str.strip().str.upper()
        frames.append(sub[["patient_id", "label", "age", "sex"]])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["patient_id"], keep="last")
    return out


def _discover_npz_files(npz_dir: str) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    return [os.path.join(npz_dir, f) for f in names]


def _aggregate_mean(emb: np.ndarray) -> np.ndarray:
    return emb.mean(axis=0)


def _aggregate_std(emb: np.ndarray) -> np.ndarray:
    return emb.std(axis=0, ddof=0)


def _aggregate_mean_std(emb: np.ndarray) -> np.ndarray:
    m = emb.mean(axis=0)
    s = emb.std(axis=0, ddof=0)
    return np.concatenate([m, s], axis=0)


def _infer_k(npz_paths: list[str]) -> int:
    k_max = -1
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        if "cluster_ids" not in z.files:
            continue
        cid = np.asarray(z["cluster_ids"]).reshape(-1)
        if cid.size == 0:
            continue
        try:
            k_max = max(k_max, int(np.max(cid)))
        except Exception:
            continue
    return int(k_max + 1)


def _cluster_hist(path: str, k: int, *, drop_outside: bool = False) -> Optional[np.ndarray]:
    z = np.load(path, allow_pickle=True)
    if "cluster_ids" not in z.files:
        return None
    cid = np.asarray(z["cluster_ids"]).reshape(-1)
    if cid.size == 0 or np.any(cid < 0):
        return None
    if "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1)
        if m.shape[0] == cid.shape[0]:
            cid = cid[m.astype(bool)]
    if cid.size == 0:
        return np.zeros((k,), dtype=np.float64)
    cid = cid.astype(np.int64, copy=False)
    if drop_outside:
        inside = (cid >= 0) & (cid < k)
        cid = cid[inside]
        if cid.size == 0:
            return None
    else:
        cid = np.clip(cid, 0, k - 1)
    h = np.bincount(cid, minlength=k).astype(np.float64)
    s = float(h.sum())
    return h / s if s > 0 else h


@dataclass
class Dataset:
    patient_ids: np.ndarray
    y: np.ndarray
    X: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    cluster_hist_bins: Optional[int] = None


def _sex_to_bin(sex: str) -> float:
    """
    Encode sex as requested: M=0, F=1.
    Unknown/other values fall back to NaN (will be filtered upstream).
    """
    s = str(sex).strip().upper()
    if s.startswith("M"):
        return 0.0
    if s.startswith("F"):
        return 1.0
    return float("nan")


def _load_features(
    npz_dir: str,
    clinical: pd.DataFrame,
    pooling: str,
    *,
    cluster_hist_k: Optional[int] = None,
) -> Dataset:
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)

    pids: list[str] = []
    ys: list[int] = []
    keep_paths: list[str] = []
    ages: list[float] = []
    sexes: list[float] = []
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
        pids.append(pid)
        ys.append(int(row["label"]))
        keep_paths.append(p)
        ages.append(float(row["age"]))
        sexes.append(float(sb))

    if len(pids) < 5:
        raise RuntimeError(f"Too few patients after join: {len(pids)}")

    y = np.array(ys, dtype=np.int64)
    pid_arr = np.array(pids)
    age_arr = np.array(ages, dtype=np.float64)
    sex_arr = np.array(sexes, dtype=np.float64)

    if pooling in ("mean", "std", "mean_std"):
        agg = {"mean": _aggregate_mean, "std": _aggregate_std, "mean_std": _aggregate_mean_std}[pooling]
        feats = []
        for p in keep_paths:
            z = np.load(p, allow_pickle=True)
            emb = np.asarray(z["embeddings"], dtype=np.float64)
            feats.append(agg(emb))
        X = np.stack(feats, axis=0).astype(np.float64)
        return Dataset(patient_ids=pid_arr, y=y, X=X, age=age_arr, sex=sex_arr, cluster_hist_bins=None)

    if pooling == "cluster_hist":
        k_inf = _infer_k(keep_paths)
        if k_inf <= 0:
            raise RuntimeError("cluster_hist requested but no cluster_ids found.")
        if cluster_hist_k is not None:
            k_bins = int(cluster_hist_k)
            if k_bins < 1:
                raise ValueError("cluster_hist_k must be >= 1")
            drop_out = k_bins < k_inf
        else:
            k_bins = k_inf
            drop_out = False
        feats_c: list[np.ndarray] = []
        ok: list[bool] = []
        for p in keep_paths:
            h = _cluster_hist(p, k=k_bins, drop_outside=drop_out)
            if h is None:
                ok.append(False)
            else:
                ok.append(True)
                feats_c.append(h)
        ok_m = np.array(ok, dtype=bool)
        if ok_m.sum() < 5:
            raise RuntimeError(f"cluster_hist: too few valid patients ({ok_m.sum()}).")
        pid_arr = pid_arr[ok_m]
        y = y[ok_m]
        age_arr = age_arr[ok_m]
        sex_arr = sex_arr[ok_m]
        X = np.stack(feats_c, axis=0).astype(np.float64)
        return Dataset(
            patient_ids=pid_arr,
            y=y,
            X=X,
            age=age_arr,
            sex=sex_arr,
            cluster_hist_bins=k_bins,
        )

    raise ValueError(f"Unknown pooling: {pooling}")


def _svc_gamma_from_arg(s: str) -> float | str:
    t = str(s).strip().lower()
    if t in ("scale", "auto"):
        return t
    return float(t)


def _fit_model(
    kind: str,
    *,
    logistic_C: float = 1.0,
    logistic_penalty: str = "l2",
    logistic_solver: str = "lbfgs",
    logistic_l1_ratio: Optional[float] = None,
    svc_C: float = 1.0,
    svc_gamma: str = "scale",
):
    if kind == "logistic":
        kw: dict = dict(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            C=float(logistic_C),
            penalty=logistic_penalty,
            solver=logistic_solver,
        )
        if logistic_l1_ratio is not None:
            kw["l1_ratio"] = float(logistic_l1_ratio)
        return LogisticRegression(**kw)
    if kind == "linear_svc":
        return LinearSVC(max_iter=20000, class_weight="balanced", random_state=42, dual=False)
    if kind == "rbf_svc":
        g = _svc_gamma_from_arg(svc_gamma)
        return SVC(
            kernel="rbf",
            C=float(svc_C),
            gamma=g,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    raise ValueError(kind)


def _scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


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


def _load_fixed_test_positions(path: str) -> pd.DataFrame:
    """
    Expect a CSV containing at least:
      latent_source, pooling, model, test_roc_auc, test_pr_auc
    Works with out_kfold/test_metrics_stable_minstd.csv if you map columns accordingly.
    """
    df = pd.read_csv(path)
    # harmonize
    if "best_pooling" in df.columns and "pooling" not in df.columns:
        df = df.rename(columns={"best_pooling": "pooling"})
    if "best_model" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"best_model": "model"})
    need = {"latent_source", "pooling", "model"}
    if not need.issubset(df.columns):
        raise ValueError(f"--fixed_test_csv must contain columns {sorted(need)} (plus test_roc_auc/test_pr_auc). Got: {list(df.columns)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeated StratifiedShuffleSplit evaluation (patient-level)")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument(
        "--latent_source",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ_DIR"),
        required=True,
        help="Repeat: display_name and directory of per-patient .npz (StageB patients).",
    )
    ap.add_argument("--pooling", choices=["mean", "std", "mean_std", "cluster_hist"], required=True)
    ap.add_argument(
        "--cluster_hist_k",
        type=int,
        default=None,
        help=(
            "With --pooling cluster_hist: force histogram length K. "
            "If K is smaller than max cluster id+1 in NPZ, only slices with cluster_id < K count (renormalized). "
            "If K is larger, trailing bins are zero. Default: infer K from data."
        ),
    )
    ap.add_argument("--classifier", choices=["logistic", "linear_svc", "rbf_svc", "all"], default="all")
    ap.add_argument(
        "--concat_age_sex",
        action="store_true",
        help="Concatenate clinical features: sex(M=0,F=1) + age(min-max scaled on train only). No leakage.",
    )
    ap.add_argument(
        "--final_rescale_after_concat",
        action="store_true",
        help=(
            "If set with --concat_age_sex: build [latent_raw, age_raw, sex_bin] first, then apply a final "
            "StandardScaler fit on TRAIN only (transform TEST). This is the recommended 'global scaling' workflow. "
            "In this mode, age MinMax is skipped because final StandardScaler handles scaling."
        ),
    )
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", required=True, help="Base output directory")
    ap.add_argument(
        "--run_tag",
        default=None,
        help="Optional tag appended to out_dir (default: timestamp like 20260506_171300).",
    )
    ap.add_argument(
        "--fixed_test_csv",
        default=None,
        help="Optional: CSV containing original fixed-test AUCs to locate within the repeated-split distribution.",
    )
    ap.add_argument(
        "--write_test_predictions",
        action="store_true",
        help=(
            "Write one row per (repeat_id, test patient, classifier): OOS score used for metrics. "
            "Column prob_tts holds predict_proba[:,1] when available else decision_function (LinearSVC). "
            "Use for ensemble_oos_same_split_eval.py after filtering --classifier rows."
        ),
    )
    ap.add_argument(
        "--test_predictions_basename",
        default="test_oos_predictions.csv",
        help="Saved next to repeats.csv when --write_test_predictions is set.",
    )
    g = ap.add_argument_group("LogisticRegression (only when classifier is logistic or all)")
    g.add_argument("--logistic_C", type=float, default=1.0)
    g.add_argument(
        "--logistic_penalty",
        type=str,
        default="l2",
        choices=["l2", "l1", "elasticnet", "none"],
    )
    g.add_argument(
        "--logistic_solver",
        type=str,
        default="lbfgs",
        help="e.g. lbfgs (l2), saga (l1/elasticnet)",
    )
    g.add_argument(
        "--logistic_l1_ratio",
        type=float,
        default=None,
        help="Only for penalty=elasticnet (0<l1_ratio<1).",
    )
    g2 = ap.add_argument_group("SVC RBF (when classifier is rbf_svc or all)")
    g2.add_argument("--svc_C", type=float, default=1.0)
    g2.add_argument(
        "--svc_gamma",
        type=str,
        default="scale",
        help="scale, auto, or float as string (e.g. 0.01).",
    )
    args = ap.parse_args()

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(os.path.abspath(args.out_dir), f"run_{tag}")
    os.makedirs(out_root, exist_ok=True)
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])

    clf_kinds = ["logistic", "linear_svc", "rbf_svc"] if args.classifier == "all" else [args.classifier]

    fixed_df = _load_fixed_test_positions(args.fixed_test_csv) if args.fixed_test_csv else None
    fixed_rows: list[dict] = []

    for src_name, npz_dir in args.latent_source:
        src_name = str(src_name)
        ch_k = int(args.cluster_hist_k) if args.cluster_hist_k is not None else None
        if ch_k is not None and str(args.pooling) != "cluster_hist":
            raise SystemExit("--cluster_hist_k is only valid with --pooling cluster_hist")
        ds = _load_features(
            os.path.abspath(str(npz_dir)),
            clinical,
            pooling=str(args.pooling),
            cluster_hist_k=ch_k,
        )
        if np.unique(ds.y).size < 2:
            raise RuntimeError(f"[{src_name}] only one class present after join; cannot evaluate AUC.")

        splitter = StratifiedShuffleSplit(
            n_splits=int(args.n_splits),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
        )

        rep_rows: list[dict] = []
        pred_rows: list[dict] = []
        for repeat_id, (tr, te) in enumerate(splitter.split(ds.X, ds.y)):
            Xtr_raw, Xte_raw = ds.X[tr], ds.X[te]
            ytr, yte = ds.y[tr], ds.y[te]
            pid_te = ds.patient_ids[te]

            if bool(args.concat_age_sex) and bool(args.final_rescale_after_concat):
                # Raw concat first, then global scaling on TRAIN only.
                age_tr = ds.age[tr].reshape(-1, 1).astype(np.float64)
                age_te = ds.age[te].reshape(-1, 1).astype(np.float64)
                sex_tr = ds.sex[tr].reshape(-1, 1).astype(np.float64)
                sex_te = ds.sex[te].reshape(-1, 1).astype(np.float64)
                Xtr_cat = np.concatenate([Xtr_raw, age_tr, sex_tr], axis=1).astype(np.float64)
                Xte_cat = np.concatenate([Xte_raw, age_te, sex_te], axis=1).astype(np.float64)
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr_cat)
                Xte = sc.transform(Xte_cat)
            else:
                # Default: scale latent first, then append age/sex (age MinMax on TRAIN only).
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr_raw)
                Xte = sc.transform(Xte_raw)

                if bool(args.concat_age_sex):
                    mm = MinMaxScaler()
                    age_tr = ds.age[tr].reshape(-1, 1).astype(np.float64)
                    age_te = ds.age[te].reshape(-1, 1).astype(np.float64)
                    age_tr_s = mm.fit_transform(age_tr)
                    age_te_s = mm.transform(age_te)
                    sex_tr = ds.sex[tr].reshape(-1, 1).astype(np.float64)
                    sex_te = ds.sex[te].reshape(-1, 1).astype(np.float64)
                    Xtr = np.concatenate([Xtr, age_tr_s, sex_tr], axis=1).astype(np.float64)
                    Xte = np.concatenate([Xte, age_te_s, sex_te], axis=1).astype(np.float64)

            for kind in clf_kinds:
                if kind == "logistic":
                    model = _fit_model(
                        kind,
                        logistic_C=args.logistic_C,
                        logistic_penalty=args.logistic_penalty,
                        logistic_solver=args.logistic_solver,
                        logistic_l1_ratio=args.logistic_l1_ratio,
                    )
                elif kind == "rbf_svc":
                    model = _fit_model(kind, svc_C=float(args.svc_C), svc_gamma=str(args.svc_gamma))
                else:
                    model = _fit_model(kind)
                model.fit(Xtr, ytr)

                s_tr = _scores(model, Xtr)
                s_te = _scores(model, Xte)

                if args.write_test_predictions:
                    for i in range(int(len(te))):
                        pred_rows.append(
                            {
                                "latent_source": src_name,
                                "pooling": str(args.pooling),
                                "cluster_hist_bins": int(ds.cluster_hist_bins)
                                if ds.cluster_hist_bins is not None
                                else "",
                                "model": kind,
                                "repeat_id": int(repeat_id),
                                "patient_id": str(pid_te[i]),
                                "label": int(yte[i]),
                                "prob_tts": float(s_te[i]),
                                "n_splits": int(args.n_splits),
                                "test_size": float(args.test_size),
                                "split_random_state": int(args.random_state),
                                "concat_age_sex": int(bool(args.concat_age_sex)),
                                "final_rescale_after_concat": int(bool(args.final_rescale_after_concat)),
                            }
                        )

                rep_rows.append(
                    {
                        "latent_source": src_name,
                        "pooling": str(args.pooling),
                        "cluster_hist_bins": int(ds.cluster_hist_bins) if ds.cluster_hist_bins is not None else "",
                        "model": kind,
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
        # write per source (avoid overwrite when running multiple sources into same out_dir)
        sub = os.path.join(out_root, src_name.replace("/", "_"), str(args.pooling), str(args.classifier))
        os.makedirs(sub, exist_ok=True)
        out_rep = os.path.join(sub, "repeats.csv")
        df_rep.to_csv(out_rep, index=False)

        if args.write_test_predictions:
            out_pred = os.path.join(sub, str(args.test_predictions_basename))
            pd.DataFrame(pred_rows).to_csv(out_pred, index=False)

        # summary per model
        sum_rows: list[dict] = []
        for kind in clf_kinds:
            d = df_rep[df_rep["model"] == kind]
            roc_s = _summary_stats(d["roc_auc"].to_numpy())
            pr_s = _summary_stats(d["pr_auc"].to_numpy())
            sum_rows.append(
                {
                    "latent_source": src_name,
                    "pooling": str(args.pooling),
                    "cluster_hist_bins": int(ds.cluster_hist_bins) if ds.cluster_hist_bins is not None else "",
                    "model": kind,
                    **{f"roc_{k}": v for k, v in roc_s.items()},
                    **{f"pr_{k}": v for k, v in pr_s.items()},
                }
            )

            # plots
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.hist(d["roc_auc"].to_numpy(), bins=20, color="tab:blue", alpha=0.85)
            ax.set_title(f"{src_name} / {args.pooling} / {kind} — ROC-AUC (n={args.n_splits})")
            ax.set_xlabel("ROC-AUC")
            ax.set_ylabel("count")

            fixed_val = None
            if fixed_df is not None:
                m = (
                    (fixed_df["latent_source"].astype(str) == src_name)
                    & (fixed_df["pooling"].astype(str) == str(args.pooling))
                    & (fixed_df["model"].astype(str) == kind)
                )
                if m.any() and "test_roc_auc" in fixed_df.columns:
                    fixed_val = float(fixed_df.loc[m, "test_roc_auc"].iloc[0])
                    ax.axvline(fixed_val, color="tab:red", linewidth=2, label=f"fixed={fixed_val:.3f}")
                    ax.legend(fontsize=8)

                    # record position
                    pct = float(np.mean(d["roc_auc"].to_numpy() <= fixed_val))
                    fixed_rows.append(
                        {
                            "latent_source": src_name,
                            "pooling": str(args.pooling),
                            "model": kind,
                            "fixed_test_roc_auc": fixed_val,
                            "cdf_position": pct,
                        }
                    )

            fig.tight_layout()
            fig.savefig(os.path.join(sub, f"hist_roc_auc_{kind}.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.hist(d["pr_auc"].to_numpy(), bins=20, color="tab:green", alpha=0.85)
            ax.set_title(f"{src_name} / {args.pooling} / {kind} — PR-AUC (n={args.n_splits})")
            ax.set_xlabel("PR-AUC (Average Precision)")
            ax.set_ylabel("count")
            if fixed_df is not None:
                m = (
                    (fixed_df["latent_source"].astype(str) == src_name)
                    & (fixed_df["pooling"].astype(str) == str(args.pooling))
                    & (fixed_df["model"].astype(str) == kind)
                )
                if m.any() and "test_pr_auc" in fixed_df.columns:
                    fv = float(fixed_df.loc[m, "test_pr_auc"].iloc[0])
                    ax.axvline(fv, color="tab:red", linewidth=2, label=f"fixed={fv:.3f}")
                    ax.legend(fontsize=8)
                    pct = float(np.mean(d["pr_auc"].to_numpy() <= fv))
                    fixed_rows.append(
                        {
                            "latent_source": src_name,
                            "pooling": str(args.pooling),
                            "model": kind,
                            "fixed_test_pr_auc": fv,
                            "pr_cdf_position": pct,
                        }
                    )
            fig.tight_layout()
            fig.savefig(os.path.join(sub, f"hist_pr_auc_{kind}.png"), dpi=150)
            plt.close(fig)

        df_sum = pd.DataFrame(sum_rows)
        df_sum.to_csv(os.path.join(sub, "summary.csv"), index=False)

    if fixed_rows:
        df_fx = pd.DataFrame(fixed_rows)
        df_fx.to_csv(os.path.join(out_root, "compare_fixed_test_positions.csv"), index=False)


if __name__ == "__main__":
    main()

