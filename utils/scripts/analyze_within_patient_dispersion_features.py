#!/usr/bin/env python3
"""
Within-patient inter-slice variability decomposition (order-invariant).

Requested question
------------------
Plain AE latent에서 어떤 종류의 within-patient inter-slice variability가
가장 안정적인 TTS/Normal signal인가?

We compute, per patient, robust dispersion features across slices for each latent dim:
  - std (ddof=0)
  - IQR = q75 - q25
  - P90-P10 = q90 - q10
  - MAD = median(|x - median(x)|)

Important constraint (trajectory dynamics)
------------------------------------------
This script does NOT compute volatility / delta std / curvature because the StageB NPZs
do not contain explicit slice order metadata (only embeddings/mask/cluster_ids/patient_id).
Without reliable slice-position/order, any dynamics would be arbitrary.

Outputs
-------
out_dir/
  dispersion_dimwise_summary.csv         # per metric, dim-wise group means + diff
  dispersion_patient_scalars.csv         # per patient, per metric scalar summaries
  repeats_<metric>_<model>.csv           # repeated split rows for each metric/model
  summary_<metric>.csv                   # repeated split summary per metric (roc_mean/std/quantiles, pr_*)
  compare_metrics_summary.csv            # one table: metrics x models with roc_mean/std etc
  plots/
    violin_l2_<metric>.png
    violin_mean_<metric>.png

Evaluation protocol (no leakage)
--------------------------------
Repeated StratifiedShuffleSplit:
  - StandardScaler fit on TRAIN only
  - classifier train on TRAIN only
  - evaluate on TEST only

Note: these dispersion vectors are 512-d (same dim as embeddings), order-invariant across slices.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from repeated_stratified_shuffle_eval import _merge_labels_and_clinical  # noqa: E402


def _discover_npz_files(npz_dir: str) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    return [os.path.join(npz_dir, f) for f in names]


def _robust_mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    return np.median(np.abs(x - med), axis=0)


def _compute_dispersion(emb: np.ndarray) -> dict[str, np.ndarray]:
    # emb: (S, D)
    q10 = np.quantile(emb, 0.10, axis=0)
    q25 = np.quantile(emb, 0.25, axis=0)
    q50 = np.quantile(emb, 0.50, axis=0)
    q75 = np.quantile(emb, 0.75, axis=0)
    q90 = np.quantile(emb, 0.90, axis=0)
    out = {
        "std": emb.std(axis=0, ddof=0),
        "iqr": (q75 - q25),
        "p90_p10": (q90 - q10),
        "mad": np.median(np.abs(emb - q50), axis=0),
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


@dataclass
class DS:
    patient_id: np.ndarray
    y: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    X_by_metric: dict[str, np.ndarray]  # metric -> (n, d)


def _load_ds(npz_dir: str, clinical: pd.DataFrame) -> DS:
    clinical = clinical.set_index("patient_id", drop=False)
    paths = _discover_npz_files(npz_dir)
    pids: list[str] = []
    ys: list[int] = []
    ages: list[float] = []
    sexes: list[str] = []
    by_metric: dict[str, list[np.ndarray]] = {"std": [], "iqr": [], "p90_p10": [], "mad": []}

    for p in paths:
        pid = os.path.basename(p).replace(".npz", "")
        if pid not in clinical.index:
            continue
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            continue

        z = np.load(p, allow_pickle=True)
        if "embeddings" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)  # (S,D)
        if emb.ndim != 2 or emb.shape[0] < 2:
            continue

        # if mask exists, apply it consistently (only keep slices where mask==1)
        if "mask" in z.files:
            m = np.asarray(z["mask"]).reshape(-1)
            if m.shape[0] == emb.shape[0]:
                keep = m.astype(bool)
                if keep.sum() >= 2:
                    emb = emb[keep]

        disp = _compute_dispersion(emb)
        pids.append(pid)
        ys.append(int(row["label"]))
        ages.append(float(row["age"]))
        sexes.append(str(row["sex"]))
        for k in by_metric:
            by_metric[k].append(disp[k])

    if len(pids) < 10:
        raise RuntimeError(f"Too few patients after join: {len(pids)}")

    X_by = {k: np.stack(v, axis=0).astype(np.float64) for k, v in by_metric.items()}
    return DS(
        patient_id=np.array(pids, dtype=str),
        y=np.array(ys, dtype=np.int64),
        age=np.array(ages, dtype=np.float64),
        sex=np.array(sexes, dtype=str),
        X_by_metric=X_by,
    )


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


def _violin(out_path: Path, a0: np.ndarray, a1: np.ndarray, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    parts = ax.violinplot([a0, a1], showmeans=True, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Normal", "TTS"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompose within-patient variability features (order-invariant).")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--latent_source", default="plain_ae")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["logistic", "linear_svc"], choices=["logistic", "linear_svc", "rbf_svc"])
    ap.add_argument("--topk_dims", type=int, default=25, help="For dimwise summary highlight only (still saves full CSV).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    ds = _load_ds(os.path.abspath(args.npz_dir), clinical)

    y = ds.y
    if np.unique(y).size < 2:
        raise RuntimeError("Only one class present after join.")

    # per-patient scalar summaries per metric
    scalar_rows = []
    for metric, X in ds.X_by_metric.items():
        l2 = np.linalg.norm(X, axis=1)
        meanv = X.mean(axis=1)
        scalar_rows.append(
            pd.DataFrame(
                {
                    "latent_source": str(args.latent_source),
                    "metric": metric,
                    "patient_id": ds.patient_id,
                    "label": y,
                    "l2": l2,
                    "mean": meanv,
                    "age": ds.age,
                    "sex": ds.sex,
                }
            )
        )

        _violin(
            plots_dir / f"violin_l2_{metric}.png",
            l2[y == 0],
            l2[y == 1],
            title=f"{args.latent_source} — {metric}: ||disp||₂",
            ylabel="L2 norm (raw)",
        )
        _violin(
            plots_dir / f"violin_mean_{metric}.png",
            meanv[y == 0],
            meanv[y == 1],
            title=f"{args.latent_source} — {metric}: mean(disp)",
            ylabel="mean (raw)",
        )

    df_scal = pd.concat(scalar_rows, ignore_index=True)
    df_scal.to_csv(out_dir / "dispersion_patient_scalars.csv", index=False)

    # dim-wise summaries per metric
    dim_rows = []
    for metric, X in ds.X_by_metric.items():
        Xn = X[y == 0]
        Xt = X[y == 1]
        mean_n = Xn.mean(axis=0)
        mean_t = Xt.mean(axis=0)
        diff = mean_t - mean_n
        absdiff = np.abs(diff)
        df = pd.DataFrame(
            {
                "latent_source": str(args.latent_source),
                "metric": metric,
                "dim": np.arange(X.shape[1], dtype=int),
                "mean_normal": mean_n,
                "mean_tts": mean_t,
                "diff_tts_minus_normal": diff,
                "abs_diff": absdiff,
            }
        ).sort_values("abs_diff", ascending=False)
        dim_rows.append(df)

        k = int(min(max(args.topk_dims, 1), X.shape[1]))
        df.head(k).to_csv(out_dir / f"top{k}_dims_{metric}.csv", index=False)

    df_dim = pd.concat(dim_rows, ignore_index=True)
    df_dim.to_csv(out_dir / "dispersion_dimwise_summary.csv", index=False)

    # repeated split eval for each metric/model
    splitter = StratifiedShuffleSplit(
        n_splits=int(args.n_splits),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    compare_rows = []
    for metric, X in ds.X_by_metric.items():
        rep_rows = []
        for repeat_id, (tr, te) in enumerate(splitter.split(X, y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            ytr, yte = y[tr], y[te]
            for kind in list(args.models):
                model = _fit_model(kind)
                model.fit(Xtr, ytr)
                s_tr = _scores(model, Xtr)
                s_te = _scores(model, Xte)
                rep_rows.append(
                    {
                        "latent_source": str(args.latent_source),
                        "metric": metric,
                        "model": kind,
                        "repeat_id": int(repeat_id),
                        "train_size": int(len(tr)),
                        "test_size": int(len(te)),
                        "roc_auc": float(roc_auc_score(yte, s_te)),
                        "pr_auc": float(average_precision_score(yte, s_te)),
                        "train_roc_auc": float(roc_auc_score(ytr, s_tr)),
                        "train_pr_auc": float(average_precision_score(ytr, s_tr)),
                    }
                )

        df_rep = pd.DataFrame(rep_rows)
        for kind in list(args.models):
            df_rep[df_rep["model"] == kind].to_csv(out_dir / f"repeats_{metric}_{kind}.csv", index=False)

        # summary per model
        sum_rows = []
        for kind in list(args.models):
            d = df_rep[df_rep["model"] == kind]
            roc_s = _summary_stats(d["roc_auc"].to_numpy())
            pr_s = _summary_stats(d["pr_auc"].to_numpy())
            row = {
                "latent_source": str(args.latent_source),
                "metric": metric,
                "model": kind,
                **{f"roc_{k}": v for k, v in roc_s.items()},
                **{f"pr_{k}": v for k, v in pr_s.items()},
            }
            sum_rows.append(row)
            compare_rows.append(row)

        pd.DataFrame(sum_rows).to_csv(out_dir / f"summary_{metric}.csv", index=False)

    df_cmp = pd.DataFrame(compare_rows)
    df_cmp = df_cmp.sort_values(["model", "roc_mean", "roc_std"], ascending=[True, False, True])
    df_cmp.to_csv(out_dir / "compare_metrics_summary.csv", index=False)

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()

