#!/usr/bin/env python3
"""
Patient-level k-fold CV for StageB per-patient NPZ embeddings (plainAE / DiffAE / MONAI AE).

Goal
----
1) (Optional) Tune a configuration on train+val using stratified k-fold CV.
2) Retrain on full train+val with that configuration.
3) Evaluate once on held-out test (final).

Design (no leakage)
-------------------
- Unit is the patient (one NPZ per patient).
- CV is performed **only on train+val** (never uses test).
- For each fold:
  * Fit StandardScaler on fold-train only
  * Train model on fold-train only
  * Evaluate on fold-val only
- After selecting best config on CV, refit on all train+val, then evaluate on test once.

Features / poolings
-------------------
- mean, std, mean_std: derived from embeddings (same as latent_space_pca_umap.py)
- cluster_hist: normalized histogram of cluster_ids (uses mask if present)

Outputs
-------
- cv_metrics.csv: one row per (latent_source, pooling, model, fold) on train+val CV
- cv_summary.csv: aggregated mean/std/min/max over folds per (latent_source, pooling, model)
- selected_configs.csv: best (pooling, model) per latent_source
- test_metrics.csv: final test ROC/PR for selected config per latent_source

Example
-------
python utils/scripts/latent_kfold_cv.py \
  --split_dir data \
  --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \
  --latent_source plain_ae /Volumes/.../diff3dformer_stageB_plain_ae/patients \
  --latent_source diffae /Volumes/.../diff3dformer_stageB_diffae/patients \
  --latent_source monai_ae /Volumes/.../diff3dformer_stageB_monai_ae/patients \
  --k 5 \
  --out_dir /Volumes/.../latent_data/out_kfold
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC


# ---------------------------------------------------------------------------
# Minimal data loading (mirrors latent_space_pca_umap behavior)
# ---------------------------------------------------------------------------

def _bootstrap_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Bootstrap CIs for ROC-AUC and PR-AUC on a fixed test set.
    Returns (roc_q025,q50,q975), (pr_q025,q50,q975).
    Skips resamples that end up single-class.
    """
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


def _merge_labels_and_clinical(paths: list[str]) -> pd.DataFrame:
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

        def _to_label(v) -> int:
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

        sub["label"] = sub["label_raw"].map(_to_label)
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


def _infer_k_from_npz_paths(npz_paths: list[str]) -> int:
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


def _cluster_hist(path: str, k: int) -> Optional[np.ndarray]:
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
    cid = np.clip(cid.astype(np.int64), 0, k - 1)
    h = np.bincount(cid, minlength=k).astype(np.float64)
    s = float(h.sum())
    return h / s if s > 0 else h


@dataclass
class SourceData:
    name: str
    patient_ids: np.ndarray
    y: np.ndarray
    X_by_pooling: dict[str, np.ndarray]
    age: np.ndarray
    sex: np.ndarray


def _sex_to_bin(sex: str) -> float:
    """
    Encode sex as requested: M=0, F=1.
    Unknown/other values fall back to NaN (will be filtered).
    """
    s = str(sex).strip().upper()
    if s.startswith("M"):
        return 0.0
    if s.startswith("F"):
        return 1.0
    return float("nan")


def _load_source(name: str, npz_dir: str, clinical: pd.DataFrame, poolings: list[str]) -> SourceData:
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)

    keep_pids: list[str] = []
    keep_paths: list[str] = []
    ys: list[int] = []
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
        keep_pids.append(pid)
        keep_paths.append(p)
        ys.append(int(row["label"]))
        ages.append(float(row["age"]))
        sexes.append(float(sb))

    if len(keep_pids) < 5:
        raise RuntimeError(f"[{name}] too few patients after clinical join: {len(keep_pids)}")

    y = np.array(ys, dtype=np.int64)
    pid_arr = np.array(keep_pids)
    age_arr = np.array(ages, dtype=np.float64)
    sex_arr = np.array(sexes, dtype=np.float64)

    X_by: dict[str, np.ndarray] = {}
    # embedding-derived poolings
    for pooling in poolings:
        if pooling in ("mean", "std", "mean_std"):
            feats: list[np.ndarray] = []
            agg: Callable[[np.ndarray], np.ndarray]
            if pooling == "mean":
                agg = _aggregate_mean
            elif pooling == "std":
                agg = _aggregate_std
            else:
                agg = _aggregate_mean_std
            for p in keep_paths:
                z = np.load(p, allow_pickle=True)
                if "embeddings" not in z.files:
                    raise RuntimeError(f"[{name}] missing embeddings in {p}")
                emb = np.asarray(z["embeddings"], dtype=np.float64)
                feats.append(agg(emb))
            X_by[pooling] = np.stack(feats, axis=0).astype(np.float64)

    # cluster histogram pooling
    if "cluster_hist" in poolings:
        k = _infer_k_from_npz_paths(keep_paths)
        if k <= 0:
            print(f"[{name}] cluster_hist skipped: no cluster_ids found.")
        else:
            feats_c: list[np.ndarray] = []
            ok_mask: list[bool] = []
            for p in keep_paths:
                h = _cluster_hist(p, k=k)
                if h is None:
                    ok_mask.append(False)
                else:
                    ok_mask.append(True)
                    feats_c.append(h)
            if sum(ok_mask) < 5:
                print(f"[{name}] cluster_hist skipped: too few valid cluster_ids patients ({sum(ok_mask)}).")
            else:
                ok = np.array(ok_mask, dtype=bool)
                pid_arr = pid_arr[ok]
                y = y[ok]
                age_arr = age_arr[ok]
                sex_arr = sex_arr[ok]
                # also filter existing X_by matrices to stay aligned
                for key in list(X_by.keys()):
                    X_by[key] = X_by[key][ok]
                X_by["cluster_hist"] = np.stack(feats_c, axis=0).astype(np.float64)

    return SourceData(name=name, patient_ids=pid_arr, y=y, X_by_pooling=X_by, age=age_arr, sex=sex_arr)


def _fit_models(Xtr: np.ndarray, ytr: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    out["logistic"] = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=42, solver="lbfgs"
    ).fit(Xtr, ytr)
    out["linear_svc"] = LinearSVC(
        max_iter=20000, class_weight="balanced", random_state=42, dual=False
    ).fit(Xtr, ytr)
    out["rbf_svc"] = SVC(
        kernel="rbf", gamma="scale", class_weight="balanced", random_state=42
    ).fit(Xtr, ytr)
    return out


def _scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _safe_to_csv(df: pd.DataFrame, out_path: str, retries: int = 8, sleep_s: float = 0.25) -> None:
    """
    Write CSV robustly on network/external volumes.
    Uses a temp file + atomic replace; retries on BlockingIOError.
    """
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    last_exc: Exception | None = None
    for i in range(int(retries)):
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)
            return
        except BlockingIOError as exc:
            last_exc = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            time.sleep(sleep_s * (i + 1))
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
    raise BlockingIOError(f"Failed to write {out_path} after {retries} retries. Close any program using it.") from last_exc


def _load_split_ids(split_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = os.path.abspath(split_dir)
    train = pd.read_csv(os.path.join(sd, "train.csv"))["patient_id"].astype(str).to_numpy()
    val = pd.read_csv(os.path.join(sd, "val.csv"))["patient_id"].astype(str).to_numpy()
    test = pd.read_csv(os.path.join(sd, "test.csv"))["patient_id"].astype(str).to_numpy()
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description="k-fold CV on StageB patient NPZ latents")
    ap.add_argument(
        "--split_dir",
        default=None,
        help="Folder with train.csv, val.csv, test.csv (patient_id). If set, CV uses train+val only and final test is evaluated once.",
    )
    ap.add_argument("--labels_csv", action="append", required=True, help="CSV(s) with patient_id, label/case, age, sex")
    ap.add_argument(
        "--latent_source",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ_DIR"),
        required=True,
        help="Repeat: display_name and directory of per-patient .npz (embeddings).",
    )
    ap.add_argument("--k", type=int, default=5, help="StratifiedKFold splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--concat_age_sex",
        action="store_true",
        help="Concatenate clinical features: age(min-max scaled on train only) + sex(M=0,F=1). No leakage.",
    )
    ap.add_argument(
        "--final_rescale_after_concat",
        action="store_true",
        help=(
            "If set with --concat_age_sex: build [latent_raw, age_raw, sex_bin] then apply a final StandardScaler "
            "fit on TRAIN only (transform VAL/TEST). This is a 'global scaling' workflow; age MinMax is skipped."
        ),
    )
    ap.add_argument(
        "--poolings",
        nargs="+",
        default=["mean", "std", "mean_std", "cluster_hist"],
        help="Which feature poolings to evaluate",
    )
    ap.add_argument(
        "--select_by",
        choices=["roc_auc", "pr_auc"],
        default="roc_auc",
        help="Metric to select best config on CV (per latent_source).",
    )
    ap.add_argument(
        "--eval_stable_min_std",
        action="store_true",
        help="Additionally evaluate 2 stable configs per latent_source on test: (min roc_std) and (min pr_std), "
        "each tie-broken by higher roc_mean / pr_mean. Requires --split_dir.",
    )
    ap.add_argument(
        "--test_bootstrap_n",
        type=int,
        default=0,
        help="If >0, bootstrap ROC/PR CIs on the test set for each evaluated config (selected + stable_minstd).",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])

    # Split handling
    split_train = split_val = split_test = None
    if args.split_dir:
        split_train, split_val, split_test = _load_split_ids(args.split_dir)
        split_train = np.unique(split_train)
        split_val = np.unique(split_val)
        split_test = np.unique(split_test)
        # enforce disjointness
        if set(split_train) & set(split_val) or set(split_train) & set(split_test) or set(split_val) & set(split_test):
            raise ValueError("train/val/test patient_id overlap detected in split_dir.")

    cv_rows: list[dict] = []
    test_rows: list[dict] = []
    stable_test_rows: list[dict] = []
    selected_rows: list[dict] = []

    for name, npz_dir in args.latent_source:
        src = _load_source(str(name), os.path.abspath(str(npz_dir)), clinical, poolings=list(args.poolings))
        X_by = src.X_by_pooling
        y_all = src.y
        pid_all = src.patient_ids

        # If split_dir is provided, restrict to train+val for CV and keep test for final eval.
        if split_train is not None:
            tv_set = set(split_train) | set(split_val)
            te_set = set(split_test)
            tv_mask = np.array([p in tv_set for p in pid_all], dtype=bool)
            te_mask = np.array([p in te_set for p in pid_all], dtype=bool)
            if not np.any(tv_mask) or not np.any(te_mask):
                print(f"[{src.name}] skip: missing train+val or test patients after alignment.")
                continue
            pid_tv, y_tv = pid_all[tv_mask], y_all[tv_mask]
            pid_te, y_te = pid_all[te_mask], y_all[te_mask]
        else:
            pid_tv, y_tv = pid_all, y_all
            pid_te, y_te = np.array([], dtype=str), np.array([], dtype=np.int64)

        if np.unique(y_tv).size < 2:
            print(f"[{src.name}] skip: only one class present in train+val pool.")
            continue

        # --- CV on train+val pool ---
        skf = StratifiedKFold(n_splits=int(args.k), shuffle=True, random_state=int(args.seed))
        for pooling, X in X_by.items():
            X_tv = X[tv_mask] if split_train is not None else X
            if X_tv.shape[0] != y_tv.shape[0]:
                raise RuntimeError(f"[{src.name}] shape mismatch pooling={pooling}: X_tv={X_tv.shape} y_tv={y_tv.shape}")

            for fold, (tr, va) in enumerate(skf.split(X_tv, y_tv)):
                ytr, yva = y_tv[tr], y_tv[va]
                if np.unique(yva).size < 2:
                    continue

                if bool(args.concat_age_sex) and bool(args.final_rescale_after_concat):
                    # Raw concat -> global StandardScaler (train only)
                    age_tv = (src.age[tv_mask] if split_train is not None else src.age).astype(np.float64)
                    sex_tv = (src.sex[tv_mask] if split_train is not None else src.sex).astype(np.float64)
                    Xtr_cat = np.concatenate(
                        [X_tv[tr], age_tv[tr].reshape(-1, 1), sex_tv[tr].reshape(-1, 1)], axis=1
                    ).astype(np.float64)
                    Xva_cat = np.concatenate(
                        [X_tv[va], age_tv[va].reshape(-1, 1), sex_tv[va].reshape(-1, 1)], axis=1
                    ).astype(np.float64)
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(Xtr_cat)
                    Xva = sc.transform(Xva_cat)
                else:
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X_tv[tr])
                    Xva = sc.transform(X_tv[va])
                    if bool(args.concat_age_sex):
                        mm = MinMaxScaler()
                        age_tv = (src.age[tv_mask] if split_train is not None else src.age).astype(np.float64)
                        sex_tv = (src.sex[tv_mask] if split_train is not None else src.sex).astype(np.float64)
                        age_tr = age_tv[tr].reshape(-1, 1)
                        age_va = age_tv[va].reshape(-1, 1)
                        age_tr_s = mm.fit_transform(age_tr)
                        age_va_s = mm.transform(age_va)
                        sex_tr = sex_tv[tr].reshape(-1, 1)
                        sex_va = sex_tv[va].reshape(-1, 1)
                        Xtr = np.concatenate([Xtr, age_tr_s, sex_tr], axis=1).astype(np.float64)
                        Xva = np.concatenate([Xva, age_va_s, sex_va], axis=1).astype(np.float64)

                models = _fit_models(Xtr, ytr)
                for mname, model in models.items():
                    s = _scores(model, Xva)
                    cv_rows.append(
                        {
                            "latent_source": src.name,
                            "pooling": pooling,
                            "model": mname,
                            "k": int(args.k),
                            "fold": int(fold),
                            "n_train": int(len(tr)),
                            "n_val": int(len(va)),
                            "roc_auc": float(roc_auc_score(yva, s)),
                            "pr_auc": float(average_precision_score(yva, s)),
                        }
                    )

        # Select best config by mean CV metric
        df_cv_src = pd.DataFrame([r for r in cv_rows if r["latent_source"] == src.name])
        if df_cv_src.empty:
            print(f"[{src.name}] skip: no CV rows produced.")
            continue
        df_cv_sum = (
            df_cv_src.groupby(["pooling", "model"], as_index=False)
            .agg(roc_mean=("roc_auc", "mean"), pr_mean=("pr_auc", "mean"), n_folds=("fold", "nunique"))
        )
        metric = str(args.select_by)
        best_idx = int(df_cv_sum[metric.replace("_auc", "_mean")].idxmax()) if metric in ("roc_auc", "pr_auc") else int(df_cv_sum["roc_mean"].idxmax())
        best = df_cv_sum.iloc[best_idx]
        best_pooling = str(best["pooling"])
        best_model = str(best["model"])
        selected_rows.append(
            {
                "latent_source": src.name,
                "select_by": metric,
                "best_pooling": best_pooling,
                "best_model": best_model,
                "cv_roc_mean": float(best["roc_mean"]),
                "cv_pr_mean": float(best["pr_mean"]),
                "cv_n_folds": int(best["n_folds"]),
            }
        )

        # --- Final refit on all train+val, test once ---
        if split_train is not None:
            X_best = X_by[best_pooling]
            X_tv = X_best[tv_mask]
            X_te = X_best[te_mask]
            if bool(args.concat_age_sex) and bool(args.final_rescale_after_concat):
                age_tv = src.age[tv_mask].reshape(-1, 1).astype(np.float64)
                age_te = src.age[te_mask].reshape(-1, 1).astype(np.float64)
                sex_tv = src.sex[tv_mask].reshape(-1, 1).astype(np.float64)
                sex_te = src.sex[te_mask].reshape(-1, 1).astype(np.float64)
                Xtr_cat = np.concatenate([X_tv, age_tv, sex_tv], axis=1).astype(np.float64)
                Xte_cat = np.concatenate([X_te, age_te, sex_te], axis=1).astype(np.float64)
                sc = StandardScaler()
                Xtr_full = sc.fit_transform(Xtr_cat)
                Xte = sc.transform(Xte_cat)
            else:
                sc = StandardScaler()
                Xtr_full = sc.fit_transform(X_tv)
                Xte = sc.transform(X_te)
                if bool(args.concat_age_sex):
                    mm = MinMaxScaler()
                    age_tv = src.age[tv_mask].reshape(-1, 1).astype(np.float64)
                    age_te = src.age[te_mask].reshape(-1, 1).astype(np.float64)
                    age_tv_s = mm.fit_transform(age_tv)
                    age_te_s = mm.transform(age_te)
                    sex_tv = src.sex[tv_mask].reshape(-1, 1).astype(np.float64)
                    sex_te = src.sex[te_mask].reshape(-1, 1).astype(np.float64)
                    Xtr_full = np.concatenate([Xtr_full, age_tv_s, sex_tv], axis=1).astype(np.float64)
                    Xte = np.concatenate([Xte, age_te_s, sex_te], axis=1).astype(np.float64)
            ytr_full = y_tv
            if np.unique(y_te).size < 2:
                print(f"[{src.name}] warning: test has one class; skipping final metrics.")
            else:
                # train single model type
                models_full = _fit_models(Xtr_full, ytr_full)
                model_full = models_full[best_model]
                s = _scores(model_full, Xte)
                roc_q, pr_q = _bootstrap_roc_pr(y_te, s, n_boot=args.test_bootstrap_n, seed=args.seed) if int(args.test_bootstrap_n) > 0 else ((float("nan"),)*3, (float("nan"),)*3)
                test_rows.append(
                    {
                        "latent_source": src.name,
                        "best_pooling": best_pooling,
                        "best_model": best_model,
                        "n_trainval": int(len(ytr_full)),
                        "n_test": int(len(y_te)),
                        "test_roc_auc": float(roc_auc_score(y_te, s)),
                        "test_pr_auc": float(average_precision_score(y_te, s)),
                        "test_roc_q025": roc_q[0],
                        "test_roc_q50": roc_q[1],
                        "test_roc_q975": roc_q[2],
                        "test_pr_q025": pr_q[0],
                        "test_pr_q50": pr_q[1],
                        "test_pr_q975": pr_q[2],
                    }
                )

    if not cv_rows:
        print("No CV rows produced; check inputs.", file=sys.stderr)
        sys.exit(1)

    df_cv = pd.DataFrame(cv_rows)
    out_csv = os.path.join(args.out_dir, "cv_metrics.csv")
    _safe_to_csv(df_cv, out_csv)
    print(f"Wrote {out_csv}")

    g = df_cv.groupby(["latent_source", "pooling", "model"], as_index=False)
    df_sum = g.agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        roc_min=("roc_auc", "min"),
        roc_max=("roc_auc", "max"),
        pr_mean=("pr_auc", "mean"),
        pr_std=("pr_auc", "std"),
        pr_min=("pr_auc", "min"),
        pr_max=("pr_auc", "max"),
        n_folds=("fold", "nunique"),
    )
    sum_csv = os.path.join(args.out_dir, "cv_summary.csv")
    _safe_to_csv(df_sum, sum_csv)
    print(f"Wrote {sum_csv}")

    # Optional: evaluate "most stable" (min std) configs on test as a sanity check
    if args.eval_stable_min_std:
        if split_train is None:
            raise ValueError("--eval_stable_min_std requires --split_dir")
        # build stable config list (up to 2 per AE)
        stable_cfgs: list[dict] = []
        df_sorted_roc = df_sum.sort_values(
            ["latent_source", "roc_std", "roc_mean"], ascending=[True, True, False]
        )
        df_sorted_pr = df_sum.sort_values(
            ["latent_source", "pr_std", "pr_mean"], ascending=[True, True, False]
        )
        for src in sorted(df_sum["latent_source"].unique().tolist()):
            a = df_sorted_roc[df_sorted_roc["latent_source"] == src].head(1)
            b = df_sorted_pr[df_sorted_pr["latent_source"] == src].head(1)
            for tag, one in [("min_roc_std", a), ("min_pr_std", b)]:
                if one.empty:
                    continue
                row = one.iloc[0].to_dict()
                row["stable_type"] = tag
                stable_cfgs.append(row)

        # evaluate each stable config on test (train on train+val only)
        clinical2 = clinical  # alias
        for name, npz_dir in args.latent_source:
            src_name = str(name)
            src = _load_source(src_name, os.path.abspath(str(npz_dir)), clinical2, poolings=list(args.poolings))
            pid_all = src.patient_ids
            y_all = src.y
            X_by = src.X_by_pooling

            tv_set = set(split_train) | set(split_val)
            te_set = set(split_test)
            tv_mask = np.array([p in tv_set for p in pid_all], dtype=bool)
            te_mask = np.array([p in te_set for p in pid_all], dtype=bool)
            if not np.any(tv_mask) or not np.any(te_mask):
                continue
            y_tv = y_all[tv_mask]
            y_te = y_all[te_mask]
            if np.unique(y_te).size < 2:
                continue

            cfgs_here = [c for c in stable_cfgs if c.get("latent_source") == src_name]
            for cfg in cfgs_here:
                pooling = str(cfg["pooling"])
                model_name = str(cfg["model"])
                if pooling not in X_by:
                    continue
                X = X_by[pooling]
                X_tv = X[tv_mask]
                X_te = X[te_mask]
                if bool(args.concat_age_sex) and bool(args.final_rescale_after_concat):
                    age_tv = src.age[tv_mask].reshape(-1, 1).astype(np.float64)
                    age_te = src.age[te_mask].reshape(-1, 1).astype(np.float64)
                    sex_tv = src.sex[tv_mask].reshape(-1, 1).astype(np.float64)
                    sex_te = src.sex[te_mask].reshape(-1, 1).astype(np.float64)
                    Xtr_cat = np.concatenate([X_tv, age_tv, sex_tv], axis=1).astype(np.float64)
                    Xte_cat = np.concatenate([X_te, age_te, sex_te], axis=1).astype(np.float64)
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(Xtr_cat)
                    Xte = sc.transform(Xte_cat)
                else:
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X_tv)
                    Xte = sc.transform(X_te)
                    if bool(args.concat_age_sex):
                        mm = MinMaxScaler()
                        age_tv = src.age[tv_mask].reshape(-1, 1).astype(np.float64)
                        age_te = src.age[te_mask].reshape(-1, 1).astype(np.float64)
                        age_tv_s = mm.fit_transform(age_tv)
                        age_te_s = mm.transform(age_te)
                        sex_tv = src.sex[tv_mask].reshape(-1, 1).astype(np.float64)
                        sex_te = src.sex[te_mask].reshape(-1, 1).astype(np.float64)
                        Xtr = np.concatenate([Xtr, age_tv_s, sex_tv], axis=1).astype(np.float64)
                        Xte = np.concatenate([Xte, age_te_s, sex_te], axis=1).astype(np.float64)
                models_full = _fit_models(Xtr, y_tv)
                if model_name not in models_full:
                    continue
                s = _scores(models_full[model_name], Xte)
                roc_q, pr_q = _bootstrap_roc_pr(y_te, s, n_boot=args.test_bootstrap_n, seed=args.seed) if int(args.test_bootstrap_n) > 0 else ((float("nan"),)*3, (float("nan"),)*3)
                stable_test_rows.append(
                    {
                        "latent_source": src_name,
                        "stable_type": str(cfg["stable_type"]),
                        "pooling": pooling,
                        "model": model_name,
                        "cv_roc_mean": float(cfg["roc_mean"]),
                        "cv_roc_std": float(cfg["roc_std"]),
                        "cv_pr_mean": float(cfg["pr_mean"]),
                        "cv_pr_std": float(cfg["pr_std"]),
                        "n_trainval": int(len(y_tv)),
                        "n_test": int(len(y_te)),
                        "test_roc_auc": float(roc_auc_score(y_te, s)),
                        "test_pr_auc": float(average_precision_score(y_te, s)),
                        "test_roc_q025": roc_q[0],
                        "test_roc_q50": roc_q[1],
                        "test_roc_q975": roc_q[2],
                        "test_pr_q025": pr_q[0],
                        "test_pr_q50": pr_q[1],
                        "test_pr_q975": pr_q[2],
                    }
                )

    if selected_rows:
        df_sel = pd.DataFrame(selected_rows)
        sel_csv = os.path.join(args.out_dir, "selected_configs.csv")
        _safe_to_csv(df_sel, sel_csv)
        print(f"Wrote {sel_csv}")

    if test_rows:
        df_te = pd.DataFrame(test_rows)
        te_csv = os.path.join(args.out_dir, "test_metrics.csv")
        _safe_to_csv(df_te, te_csv)
        print(f"Wrote {te_csv}")

    if stable_test_rows:
        df_st = pd.DataFrame(stable_test_rows)
        st_csv = os.path.join(args.out_dir, "test_metrics_stable_minstd.csv")
        _safe_to_csv(df_st, st_csv)
        print(f"Wrote {st_csv}")


if __name__ == "__main__":
    main()

