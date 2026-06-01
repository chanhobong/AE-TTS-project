#!/usr/bin/env python3
"""
Evaluate a single classifier head on a fixed split (train+val -> test).

Typical use-case (requested)
----------------------------
Evaluate only one setting, e.g.:
  plain_ae / std / logistic_en_C0.1_r0.2

Metrics
-------
- Accuracy
- Sensitivity / Recall (TTS=1)
- Specificity (Normal=0)
- Precision (TTS=1)
- F1 (TTS=1)
- Balanced accuracy
- ROC-AUC, PR-AUC
- Confusion matrix: TN/FP/FN/TP

Design (no leakage)
-------------------
- Patient-level split using split_dir train.csv/val.csv/test.csv
- Fit StandardScaler on train+val only
- Train model on train+val only
- Evaluate on held-out test only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import _load_features, _merge_labels_and_clinical  # noqa: E402


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


def _build_ds(npz_dir: str, labels_csv: list[str], pooling: str) -> DS:
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in labels_csv])
    ds = _load_features(os.path.abspath(npz_dir), clinical, pooling=str(pooling))
    return DS(patient_id=ds.patient_ids.astype(str), y=ds.y.astype(np.int64), X=ds.X.astype(np.float64))


def _parse_head_id(head_id: str) -> dict[str, Any]:
    """
    Supported:
      - logistic_l2_C{C}
      - logistic_l1_C{C}
      - logistic_en_C{C}_r{l1_ratio}
      - linear_svc_C{C}
      - rbf_svc_C{C} (optional)
    """
    s = str(head_id).strip()
    m = re.fullmatch(r"(logistic_l2)_C([0-9]*\.?[0-9]+)", s)
    if m:
        return {"kind": m.group(1), "C": float(m.group(2))}
    m = re.fullmatch(r"(logistic_l1)_C([0-9]*\.?[0-9]+)", s)
    if m:
        return {"kind": m.group(1), "C": float(m.group(2))}
    m = re.fullmatch(r"(logistic_en)_C([0-9]*\.?[0-9]+)_r([0-9]*\.?[0-9]+)", s)
    if m:
        return {"kind": m.group(1), "C": float(m.group(2)), "l1_ratio": float(m.group(3))}
    m = re.fullmatch(r"(linear_svc)_C([0-9]*\.?[0-9]+)", s)
    if m:
        return {"kind": m.group(1), "C": float(m.group(2))}
    m = re.fullmatch(r"(rbf_svc)_C([0-9]*\.?[0-9]+)", s)
    if m:
        return {"kind": m.group(1), "C": float(m.group(2))}
    raise ValueError(f"Unrecognized head_id format: {head_id}")


def _make_model(kind: str, params: dict[str, Any], seed: int) -> Any:
    if kind == "logistic_l2":
        return LogisticRegression(
            max_iter=12000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            penalty="l2",
            C=float(params["C"]),
        )
    if kind == "logistic_l1":
        return LogisticRegression(
            max_iter=20000,
            class_weight="balanced",
            random_state=seed,
            solver="saga",
            penalty="l1",
            C=float(params["C"]),
        )
    if kind == "logistic_en":
        return LogisticRegression(
            max_iter=20000,
            class_weight="balanced",
            random_state=seed,
            solver="saga",
            penalty="elasticnet",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
        )
    if kind == "linear_svc":
        return LinearSVC(
            max_iter=30000,
            class_weight="balanced",
            random_state=seed,
            dual=False,
            C=float(params["C"]),
        )
    if kind == "rbf_svc":
        return SVC(
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
            C=float(params["C"]),
        )
    raise ValueError(kind)


def _score(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a single head (acc/sens/f1/etc) on fixed test split.")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--latent_source", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--pooling", default="std", choices=["mean", "std", "mean_std", "cluster_hist"])
    ap.add_argument("--head_id", required=True, help="e.g. logistic_en_C0.1_r0.2")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold on score/prob for class prediction.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() / str(args.latent_source) / str(args.pooling) / str(args.head_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _build_ds(str(args.npz_dir), list(args.labels_csv), pooling=str(args.pooling))
    tr_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "train.csv"))
    va_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "val.csv"))
    te_ids = _read_ids(os.path.join(os.path.abspath(args.split_dir), "test.csv"))
    tv_set = tr_ids | va_ids

    tv_mask = np.array([p in tv_set for p in ds.patient_id], dtype=bool)
    te_mask = np.array([p in te_ids for p in ds.patient_id], dtype=bool)
    if not np.any(tv_mask) or not np.any(te_mask):
        raise RuntimeError("No aligned train+val or test patients after join.")

    X_tv, y_tv = ds.X[tv_mask], ds.y[tv_mask]
    X_te, y_te = ds.X[te_mask], ds.y[te_mask]
    pid_te = ds.patient_id[te_mask]
    if np.unique(y_tv).size < 2 or np.unique(y_te).size < 2:
        raise RuntimeError("Need both classes in train+val and test.")

    head = _parse_head_id(str(args.head_id))
    model = _make_model(str(head["kind"]), head, seed=int(args.seed))

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tv)
    Xte = sc.transform(X_te)
    model.fit(Xtr, y_tv)

    s = _score(model, Xte)
    # If score is not probability, threshold at 0 by default would be typical. User requested threshold=0.5.
    # We keep their threshold but store raw score and prob (if available) separately.
    if hasattr(model, "predict_proba"):
        prob = s
    else:
        # convert decision function to a pseudo-prob via min-max on test for logging only (NOT for metrics like AUC)
        # classification threshold still applied on raw s.
        prob = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)

    y_pred = (prob >= float(args.threshold)).astype(int) if hasattr(model, "predict_proba") else (s >= 0.0).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_te, y_pred))
    sens = float(recall_score(y_te, y_pred, pos_label=1))  # sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    prec = float(precision_score(y_te, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
    bacc = float(balanced_accuracy_score(y_te, y_pred))
    roc = float(roc_auc_score(y_te, s))
    pr = float(average_precision_score(y_te, s))

    metrics = {
        "latent_source": str(args.latent_source),
        "pooling": str(args.pooling),
        "head_id": str(args.head_id),
        "head_kind": str(head["kind"]),
        "threshold_used": float(args.threshold) if hasattr(model, "predict_proba") else 0.0,
        "n_trainval": int(len(y_tv)),
        "n_test": int(len(y_te)),
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "precision": prec,
        "f1": f1,
        "balanced_acc": bacc,
        "roc_auc": roc,
        "pr_auc": pr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "classification_metrics.csv", index=False)

    df_pred = pd.DataFrame(
        {
            "patient_id": pid_te,
            "label": y_te.astype(int),
            "score": s.astype(float),
            "prob_like": prob.astype(float),
            "pred": y_pred.astype(int),
        }
    )
    df_pred.to_csv(out_dir / "test_predictions.csv", index=False)

    print("Wrote:", out_dir)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()

