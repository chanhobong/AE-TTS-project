#!/usr/bin/env python3
"""
Project patient-level mean||std features onto logistic coefficients by block.

Training pipeline (same as latent_space_pca_umap --eval_classifiers):
  X_raw = [mean_0..D-1, std_0..D-1]  (slice-wise mean / std, ddof=0 for std)
  X_scaled = StandardScaler.fit(train).transform(X_raw)   # 2D features, elementwise
  logit decision (no intercept) = X_scaled @ coef

This script fits the same LogisticRegression on the train split (or loads saved model+scaler),
then for every patient computes:
  score_mean_block = X_scaled[0:D] @ coef[0:D]
  score_std_block  = X_scaled[D:2D] @ coef[D:2D]
  score_linear     = score_mean_block + score_std_block  (= coef @ X_scaled)
Full decision_function includes intercept; predict_proba uses full linear + intercept.

Outputs: per-patient CSV, ROC AUC table (test / val / train), overlaid histograms,
optional max ROC among **single** scaled std coordinates (sanity vs full std projection).

Example
-------
  python project_logistic_mean_std_blocks.py \\
    --split_dir /path/data \\
    --npz_dir /path/.../patients \\
    --labels_csv /path/train.csv --labels_csv /path/val.csv --labels_csv /path/test.csv \\
    --out_dir ./logistic_block_projection

Repeat ``--labels_csv`` once per file (three times). A stray token like ``/csv`` breaks argparse.

This script imports latent_mean_std_dataset_loader only (no umap / TensorFlow).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import latent_mean_std_dataset_loader as ldm  # noqa: E402

RANDOM_STATE = ldm.RANDOM_STATE


def _split_tag(tr_m: np.ndarray, va_m: np.ndarray, te_m: np.ndarray) -> np.ndarray:
    tags = np.full(tr_m.shape[0], "", dtype=object)
    tags[tr_m] = "train"
    tags[va_m] = "val"
    tags[te_m] = "test"
    return tags


def _roc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    if np.unique(y[m]).size < 2 or int(np.sum(m)) < 3:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def _hist_overlay(
    scores: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    title: str,
    xlabel: str,
    out_png: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = y[mask]
    sc = scores[mask]
    for lab, c, lbl in [(0, "tab:blue", "y=0 Normal"), (1, "tab:red", "y=1 TTS")]:
        m = ys == lab
        if np.any(m):
            ax.hist(sc[m], bins=18, alpha=0.5, color=c, label=lbl, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(out_png))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _best_single_std_dim_auc(Xs: np.ndarray, y: np.ndarray, te_m: np.ndarray, d: int) -> tuple[int, float]:
    """AUC on test using one scaled std feature at a time (coordinate j)."""
    best_j, best_a = -1, -1.0
    for j in range(d):
        col = d + j
        a = _roc_or_nan(y[te_m], Xs[te_m, col])
        if np.isfinite(a) and a > best_a:
            best_a, best_j = a, j
    return best_j, float(best_a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split_dir", required=True, help="Folder with train.csv, val.csv, test.csv")
    ap.add_argument("--npz_dir", required=True, help="Directory of per-patient .npz (embeddings)")
    ap.add_argument("--labels_csv", action="append", required=True, help="train/val/test CSVs (repeat)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--logistic_joblib",
        default=None,
        help="Optional: joblib path to fitted LogisticRegression (must match 2D mean||std features)",
    )
    ap.add_argument(
        "--scaler_joblib",
        default=None,
        help="Optional: joblib path to StandardScaler fitted on train (same order as eval pipeline)",
    )
    args = ap.parse_args()

    if (args.logistic_joblib is None) ^ (args.scaler_joblib is None):
        print("Provide both --logistic_joblib and --scaler_joblib, or neither to refit on split.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clinical = ldm._merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    train_ids, val_ids, test_ids = ldm.load_split_patient_ids(args.split_dir)
    ldm.verify_disjoint_patient_splits(train_ids, val_ids, test_ids, strict=True)

    npz_dir = os.path.abspath(args.npz_dir)
    pids, X, y, _age, _sex, tr_m, va_m, te_m = ldm.build_aligned_dataset_mean_std(
        npz_dir, clinical, train_ids, val_ids, test_ids
    )
    if X.shape[1] % 2 != 0:
        raise ValueError(f"mean||std width must be even, got {X.shape[1]}")
    d = X.shape[1] // 2
    n_tr = int(tr_m.sum())
    if n_tr < 2:
        print("Need at least 2 train patients.", file=sys.stderr)
        sys.exit(1)

    if args.logistic_joblib:
        try:
            import joblib
        except ImportError:
            print("pip install joblib for --logistic_joblib", file=sys.stderr)
            sys.exit(1)
        clf: LogisticRegression = joblib.load(args.logistic_joblib)
        scaler: StandardScaler = joblib.load(args.scaler_joblib)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[tr_m])
        y_train = y[tr_m]
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

    coef = clf.coef_.ravel()
    if coef.size != 2 * d:
        raise ValueError(f"Model coef size {coef.size} != {2 * d} (2 * D from data)")
    c_mean, c_std = coef[:d], coef[d:]
    inter = float(np.asarray(clf.intercept_).ravel()[0])

    Xs = scaler.transform(X)
    s_mean = Xs[:, :d] @ c_mean
    s_std = Xs[:, d:] @ c_std
    s_lin = Xs @ coef
    dfull = clf.decision_function(Xs)
    if not np.allclose(s_lin + inter, dfull, rtol=1e-5, atol=1e-5):
        raise RuntimeError("decision_function != X@coef + intercept (check model type)")
    prob = clf.predict_proba(Xs)[:, 1]
    all_m = tr_m | va_m | te_m

    tags = _split_tag(tr_m, va_m, te_m)
    rows = []
    for i, pid in enumerate(pids):
        rows.append(
            {
                "patient_id": pid,
                "split": str(tags[i]),
                "y_true": int(y[i]),
                "score_mean_block": float(s_mean[i]),
                "score_std_block": float(s_std[i]),
                "score_linear_no_intercept": float(s_lin[i]),
                "decision_function": float(dfull[i]),
                "prob_tts": float(prob[i]),
            }
        )
    pdf = pd.DataFrame(rows)
    pdf.to_csv(os.path.join(out_dir, "scores_per_patient.csv"), index=False)

    summ_rows: list[dict] = []
    for split_name, mask in ("train", tr_m), ("val", va_m), ("test", te_m), ("all", all_m):
        if split_name != "all" and not np.any(mask):
            continue
        msk = mask if split_name != "all" else all_m
        for cls in (0, 1):
            m = msk & (y == cls)
            if not np.any(m):
                continue
            summ_rows.append(
                {
                    "split": split_name,
                    "y_true": cls,
                    "n": int(np.sum(m)),
                    "score_std_block_mean": float(np.mean(s_std[m])),
                    "score_std_block_std": float(np.std(s_std[m], ddof=0)),
                    "score_mean_block_mean": float(np.mean(s_mean[m])),
                    "score_mean_block_std": float(np.std(s_mean[m], ddof=0)),
                }
            )
    pd.DataFrame(summ_rows).to_csv(
        os.path.join(out_dir, "score_stats_by_split_and_class.csv"), index=False
    )

    roc_rows: list[dict] = []
    for split_name, mask in ("train", tr_m), ("val", va_m), ("test", te_m):
        if not np.any(mask):
            continue
        yt, sm, ss, sl, pr = y[mask], s_mean[mask], s_std[mask], s_lin[mask], prob[mask]
        roc_rows.append(
            {
                "split": split_name,
                "score_std_block_auc": _roc_or_nan(yt, ss),
                "score_mean_block_auc": _roc_or_nan(yt, sm),
                "score_linear_noint_auc": _roc_or_nan(yt, sl),
                "prob_tts_auc": _roc_or_nan(yt, pr),
            }
        )
    pd.DataFrame(roc_rows).to_csv(os.path.join(out_dir, "roc_auc_by_split.csv"), index=False)

    best_j, best_auc = (-1, float("nan"))
    if np.any(te_m):
        best_j, best_auc = _best_single_std_dim_auc(Xs, y, te_m, d)
        with open(os.path.join(out_dir, "test_best_single_std_coord.txt"), "w", encoding="utf-8") as bf:
            bf.write(
                f"Among scaled std coordinates X_scaled[D+j] only, best dim j={best_j}, "
                f"test ROC-AUC={best_auc:.6f}\n"
                f"(monotone in coef[j]; compares full std-block projection vs best single axis)\n"
            )

    _hist_overlay(
        s_std,
        y,
        all_m,
        "All aligned patients: std-block score (scaled sigma) dot coef_std",
        "score_std_block",
        os.path.join(out_dir, "hist_all_score_std_block.png"),
    )
    _hist_overlay(
        s_mean,
        y,
        all_m,
        "All aligned patients: mean-block score (scaled mean) dot coef_mean",
        "score_mean_block",
        os.path.join(out_dir, "hist_all_score_mean_block.png"),
    )
    if np.any(te_m):
        _hist_overlay(
            s_std,
            y,
            te_m,
            "Test only: logistic std-block score (scaled sigma) dot coef_std",
            "score_std_block",
            os.path.join(out_dir, "hist_test_score_std_block.png"),
        )
        _hist_overlay(
            s_mean,
            y,
            te_m,
            "Test only: logistic mean-block score (scaled mean) dot coef_mean",
            "score_mean_block",
            os.path.join(out_dir, "hist_test_score_mean_block.png"),
        )
        _hist_overlay(
            dfull,
            y,
            te_m,
            "Test only: full decision_function (mean+std blocks + intercept)",
            "decision_function",
            os.path.join(out_dir, "hist_test_decision_function.png"),
        )

    with open(os.path.join(out_dir, "README_projection.txt"), "w", encoding="utf-8") as f:
        f.write(
            "score_std_block = (StandardScaler-transformed std block) dot coef[D:2D]\n"
            "score_mean_block = (scaled mean block) dot coef[0:D]\n"
            "Coefficients are w.r.t. scaled features (same as logistic_coef_importance CSV).\n"
            "Raw sigma dot raw coef_std without scaler is NOT the model score.\n"
        )

    print(f"Wrote {out_dir}/scores_per_patient.csv")
    print(f"Wrote {out_dir}/score_stats_by_split_and_class.csv")
    print(f"Wrote {out_dir}/roc_auc_by_split.csv")
    if np.any(te_m):
        print(f"D (latent dims) = {d}; test best single scaled std dim j={best_j} AUC={best_auc:.4f}")
    else:
        print(f"D (latent dims) = {d} (no test mask; skipped single-dim AUC)")


if __name__ == "__main__":
    main()
