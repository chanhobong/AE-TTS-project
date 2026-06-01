#!/usr/bin/env python3
"""
Clinical-only baseline: age + sex -> TTS vs Normal classification with repeated splits.

Protocol (no leakage)
---------------------
- StratifiedShuffleSplit (default n_splits=100, test_size=0.2, random_state=42)
- For each split:
  1) Split patients stratified.
  2) Fit MinMaxScaler on TRAIN age only.
  3) Transform TEST age with train-fitted scaler.
  4) Sex encoding is fixed: M=0, F=1.
  5) Train classifier on TRAIN only, evaluate on TEST only.
- Save repeats.csv + summary.csv + histograms.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC


def _sex_to_bin(sex: str) -> float:
    s = str(sex).strip().upper()
    if s.startswith("M"):
        return 0.0
    if s.startswith("F"):
        return 1.0
    return float("nan")


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
        sub["sex_bin"] = sub["sex"].map(_sex_to_bin)
        sub = sub.dropna(subset=["age", "sex_bin"])
        frames.append(sub[["patient_id", "label", "age", "sex_bin"]])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["patient_id"], keep="last")
    return out


@dataclass
class Dataset:
    patient_ids: np.ndarray
    y: np.ndarray
    age: np.ndarray
    sex: np.ndarray


def _to_dataset(clinical: pd.DataFrame) -> Dataset:
    pid = clinical["patient_id"].astype(str).to_numpy()
    y = clinical["label"].astype(int).to_numpy()
    age = clinical["age"].astype(float).to_numpy()
    sex = clinical["sex_bin"].astype(float).to_numpy()
    return Dataset(patient_ids=pid, y=y, age=age, sex=sex)


def _fit_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, solver="lbfgs")
    if kind == "linear_svc":
        return LinearSVC(max_iter=20000, class_weight="balanced", random_state=42, dual=False)
    if kind == "rbf_svc":
        return SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical-only repeated split evaluation (age+sex)")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--classifier", choices=["logistic", "linear_svc", "rbf_svc", "all"], default="all")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", default=None)
    args = ap.parse_args()

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(os.path.abspath(args.out_dir), f"run_{tag}_clinical_only")
    os.makedirs(out_root, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    ds = _to_dataset(clinical)
    if np.unique(ds.y).size < 2:
        raise RuntimeError("Only one class present after clinical merge; cannot evaluate AUC.")

    clf_kinds = ["logistic", "linear_svc", "rbf_svc"] if args.classifier == "all" else [args.classifier]
    splitter = StratifiedShuffleSplit(
        n_splits=int(args.n_splits),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    rep_rows: list[dict] = []
    for repeat_id, (tr, te) in enumerate(splitter.split(ds.age.reshape(-1, 1), ds.y)):
        ytr, yte = ds.y[tr], ds.y[te]
        pid_te = ds.patient_ids[te]

        mm = MinMaxScaler()
        age_tr = ds.age[tr].reshape(-1, 1).astype(np.float64)
        age_te = ds.age[te].reshape(-1, 1).astype(np.float64)
        age_tr_s = mm.fit_transform(age_tr)
        age_te_s = mm.transform(age_te)
        sex_tr = ds.sex[tr].reshape(-1, 1).astype(np.float64)
        sex_te = ds.sex[te].reshape(-1, 1).astype(np.float64)

        Xtr = np.concatenate([age_tr_s, sex_tr], axis=1)
        Xte = np.concatenate([age_te_s, sex_te], axis=1)

        for kind in clf_kinds:
            model = _fit_model(kind)
            model.fit(Xtr, ytr)
            s_tr = _scores(model, Xtr)
            s_te = _scores(model, Xte)
            rep_rows.append(
                {
                    "feature_set": "clinical_only(age_minmax_train, sex_M0_F1)",
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
    df_rep.to_csv(os.path.join(out_root, "repeats.csv"), index=False)

    sum_rows: list[dict] = []
    for kind in clf_kinds:
        d = df_rep[df_rep["model"] == kind]
        roc_s = _summary_stats(d["roc_auc"].to_numpy())
        pr_s = _summary_stats(d["pr_auc"].to_numpy())
        sum_rows.append(
            {
                "feature_set": "clinical_only(age_minmax_train, sex_M0_F1)",
                "model": kind,
                **{f"roc_{k}": v for k, v in roc_s.items()},
                **{f"pr_{k}": v for k, v in pr_s.items()},
            }
        )

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(d["roc_auc"].to_numpy(), bins=20, color="tab:blue", alpha=0.85)
        ax.set_title(f"clinical_only / {kind} — ROC-AUC (n={args.n_splits})")
        ax.set_xlabel("ROC-AUC")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, f"hist_roc_auc_{kind}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(d["pr_auc"].to_numpy(), bins=20, color="tab:green", alpha=0.85)
        ax.set_title(f"clinical_only / {kind} — PR-AUC (n={args.n_splits})")
        ax.set_xlabel("PR-AUC (Average Precision)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, f"hist_pr_auc_{kind}.png"), dpi=150)
        plt.close(fig)

    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_root, "summary.csv"), index=False)
    print("Wrote:", out_root)


if __name__ == "__main__":
    main()

