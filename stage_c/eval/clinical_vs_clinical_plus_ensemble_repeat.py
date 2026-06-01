#!/usr/bin/env python3
"""
Compare **clinical-only** (age + sex) vs **clinical + fixed ensemble score** using the **same
test folds** as the latent OOS exports (``test_oos_predictions.csv`` lists test patients per
``repeat_id``; train = cohort minus that test set).

Inputs
------
- Same ``--labels_csv`` union as other repeated eval scripts (patient_id, label, age, sex).
- Two ``test_oos_predictions.csv`` from Plain and MONAI with matching ``repeat_id`` / ``patient_id``.

Cohort
------
Patients must appear in the merged OOS table (Plain ∧ MONAI inner join) **and** have valid age/sex
in the label CSVs.

Ensemble score on **test** (split ``k``)
   :math:`s = w\\,p_{\\mathrm{plain}} + (1-w)\\,p_{\\mathrm{monai}}` from the row with ``repeat_id=k``.

Ensemble score used as **train** feature (split ``k``)
   For each training patient ``pid``, mean of :math:`s` over all OOS rows with that ``pid`` and
   ``repeat_id \\neq k`` (leave-one-split-out summary of historical test scores). If none, falls back
   to the global mean :math:`s`. **Train** column is then MinMax-scaled **on train only** (same as age).

This avoids using the current split's test scores as train inputs, but still propagates information
across repeats — interpret stacker coefficients cautiously; for a cleaner train feature you would
export **train-fold** latent scores per repeat.

Outputs (under ``out_dir/run_<tag>/``)
repeats.csv, summary.csv, optional histograms.

Example
-------
  python3 utils/scripts/clinical_vs_clinical_plus_ensemble_repeat.py \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --csv_plain .../test_oos_predictions.csv --csv_monai .../test_oos_predictions.csv \\
    --filter_model_plain logistic_en_C0.1_r0.2 --filter_model_monai rbf_svc \\
    --fixed_w 0.5 --out_dir /path/latent_data/clinical_ensemble_compare
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from clinical_only_repeated_eval import (  # noqa: E402
    _merge_labels_and_clinical,
    _summary_stats,
    _to_dataset,
)


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _apply_model_filter(d: pd.DataFrame, want: str, which: str) -> pd.DataFrame:
    want = want.strip()
    if not want:
        return d
    if "model" not in d.columns:
        raise SystemExit(f"{which}: filter {want!r} but no 'model' column")
    out = d[d["model"].astype(str) == want].copy()
    if out.empty:
        raise SystemExit(f"{which}: empty after filter {want!r}")
    return out


def _coerce_repeat(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(np.int64)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _prepare_merged_oos(
    path_plain: str,
    path_monai: str,
    *,
    prob_plain: str,
    prob_monai: str,
    filter_plain: str,
    filter_monai: str,
    fixed_w: float,
) -> pd.DataFrame:
    da = _apply_model_filter(_read(path_plain), filter_plain, "csv_plain")
    db = _apply_model_filter(_read(path_monai), filter_monai, "csv_monai")
    for name, d in (("plain", da), ("monai", db)):
        for c in ("repeat_id", "patient_id", "label", prob_plain if name == "plain" else prob_monai):
            if c not in d.columns:
                raise SystemExit(f"{name} missing {c!r}; columns={list(d.columns)}")
    dp = da[["repeat_id", "patient_id", "label", prob_plain]].copy()
    dp["patient_id"] = dp["patient_id"].astype(str)
    dp["repeat_id"] = _coerce_repeat(dp["repeat_id"])
    dp = dp.rename(columns={prob_plain: "_pp"})
    dp["_pp"] = pd.to_numeric(dp["_pp"], errors="coerce")

    dm = db[["repeat_id", "patient_id", "label", prob_monai]].copy()
    dm["patient_id"] = dm["patient_id"].astype(str)
    dm["repeat_id"] = _coerce_repeat(dm["repeat_id"])
    dm = dm.rename(columns={prob_monai: "_pm"})
    dm["_pm"] = pd.to_numeric(dm["_pm"], errors="coerce")

    j = dp.merge(dm, on=["repeat_id", "patient_id"], how="inner", suffixes=("_p", "_m"))
    lp = pd.to_numeric(j["label_p"], errors="coerce")
    lm = pd.to_numeric(j["label_m"], errors="coerce")
    if not (lp == lm).all():
        raise SystemExit("Label mismatch between Plain and MONAI OOS rows.")
    j["label"] = lp.astype(int)
    j = j.drop(columns=["label_p", "label_m"])
    j = j.dropna(subset=["_pp", "_pm", "label"])
    w = float(np.clip(fixed_w, 0.0, 1.0))
    j["ens"] = w * j["_pp"].to_numpy(dtype=np.float64) + (1.0 - w) * j["_pm"].to_numpy(dtype=np.float64)
    return j


def _ens_loo_split(pid: str, k: int, j: pd.DataFrame, fallback: float) -> float:
    pid = str(pid)
    kk = int(k)
    sub = j[(j["patient_id"].astype(str) == pid) & (j["repeat_id"].astype(int) != kk)]
    if sub.empty:
        return float(fallback)
    return float(sub["ens"].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical-only vs clinical+ensemble OOS (100 matched splits).")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--csv_plain", required=True)
    ap.add_argument("--csv_monai", required=True)
    ap.add_argument("--prob_col_plain", default="prob_tts")
    ap.add_argument("--prob_col_monai", default="prob_tts")
    ap.add_argument("--filter_model_plain", default="")
    ap.add_argument("--filter_model_monai", default="")
    ap.add_argument("--fixed_w", type=float, default=0.5)
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", default=None)
    args = ap.parse_args()

    j = _prepare_merged_oos(
        args.csv_plain,
        args.csv_monai,
        prob_plain=str(args.prob_col_plain),
        prob_monai=str(args.prob_col_monai),
        filter_plain=str(args.filter_model_plain),
        filter_monai=str(args.filter_model_monai),
        fixed_w=float(args.fixed_w),
    )

    pids_latent = set(j["patient_id"].astype(str).unique())
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    clinical = clinical[clinical["patient_id"].astype(str).isin(pids_latent)].copy()
    if clinical.empty:
        raise SystemExit("No clinical rows after intersecting with OOS patient_ids.")
    clinical = clinical.sort_values("patient_id").reset_index(drop=True)
    ds = _to_dataset(clinical)
    if np.unique(ds.y).size < 2:
        raise SystemExit("Only one class in intersected cohort.")

    global_ens_mean = float(j["ens"].mean())

    split_ids = sorted(j["repeat_id"].dropna().astype(int).unique().tolist())
    if len(split_ids) != int(args.n_splits):
        print(
            f"[warn] OOS has {len(split_ids)} repeat_ids; expecting {args.n_splits}.",
            file=sys.stderr,
        )

    pid_all = np.array([str(p) for p in ds.patient_ids], dtype=object)
    n_pat = int(ds.patient_ids.shape[0])
    idx_by_pid = {str(pid_all[i]): int(i) for i in range(n_pat)}
    cohort_pid_set = set(idx_by_pid.keys())

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(os.path.abspath(args.out_dir), f"run_{tag}_clinical_vs_clinical_plus_ensemble")
    os.makedirs(out_root, exist_ok=True)

    rep_rows: list[dict[str, Any]] = []
    for repeat_id in split_ids:
        repeat_id = int(repeat_id)
        test_pids = set(j.loc[j["repeat_id"].astype(int) == repeat_id, "patient_id"].astype(str))
        if not test_pids:
            raise SystemExit(f"repeat_id={repeat_id}: empty OOS (merged Plain∧MONAI).")
        extra_te = test_pids - cohort_pid_set
        if extra_te:
            raise SystemExit(
                f"repeat_id={repeat_id}: OOS test patients missing from label/clinical cohort "
                f"(example {sorted(extra_te)[0]!r})."
            )
        te = np.array([idx_by_pid[p] for p in sorted(test_pids)], dtype=np.int64)
        tr = np.array([i for i in range(n_pat) if str(pid_all[i]) not in test_pids], dtype=np.int64)
        if len(tr) + len(te) != n_pat:
            raise SystemExit(
                f"repeat_id={repeat_id}: train ∪ test != cohort ({len(tr)} + {len(te)} vs {n_pat})."
            )

        ytr, yte = ds.y[tr], ds.y[te]
        pid_tr = ds.patient_ids[tr].astype(str)
        pid_te = ds.patient_ids[te].astype(str)

        mm_a = MinMaxScaler()
        age_tr = ds.age[tr].reshape(-1, 1).astype(np.float64)
        age_te = ds.age[te].reshape(-1, 1).astype(np.float64)
        age_tr_s = mm_a.fit_transform(age_tr)
        age_te_s = mm_a.transform(age_te)
        sex_tr = ds.sex[tr].reshape(-1, 1).astype(np.float64)
        sex_te = ds.sex[te].reshape(-1, 1).astype(np.float64)

        Xtr_clin = np.concatenate([age_tr_s, sex_tr], axis=1)
        Xte_clin = np.concatenate([age_te_s, sex_te], axis=1)

        clf0 = LogisticRegression(max_iter=8000, class_weight="balanced", random_state=42, solver="lbfgs")
        clf0.fit(Xtr_clin, ytr)
        s0_te = clf0.predict_proba(Xte_clin)[:, 1]
        roc0 = float(roc_auc_score(yte, s0_te))
        pr0 = float(average_precision_score(yte, s0_te))

        jk = j[j["repeat_id"].astype(int) == int(repeat_id)].copy()
        jk = jk.set_index("patient_id", drop=False)
        ens_te_list: list[float] = []
        for p in pid_te:
            ps = str(p)
            if ps not in jk.index:
                raise SystemExit(
                    f"repeat_id={repeat_id}: test patient {ps!r} missing from merged OOS "
                    "(cohort / export mismatch)."
                )
            ens_te_list.append(float(jk.loc[ps, "ens"]))
        ens_te = np.asarray(ens_te_list, dtype=np.float64)

        ens_tr = np.asarray(
            [_ens_loo_split(str(p), int(repeat_id), j, global_ens_mean) for p in pid_tr],
            dtype=np.float64,
        )
        mm_e = MinMaxScaler()
        ens_tr_col = mm_e.fit_transform(ens_tr.reshape(-1, 1))
        ens_te_col = mm_e.transform(ens_te.reshape(-1, 1))

        Xtr_plus = np.concatenate([age_tr_s, sex_tr, ens_tr_col], axis=1)
        Xte_plus = np.concatenate([age_te_s, sex_te, ens_te_col], axis=1)

        clf1 = LogisticRegression(max_iter=8000, class_weight="balanced", random_state=42, solver="lbfgs")
        clf1.fit(Xtr_plus, ytr)
        s1_te = clf1.predict_proba(Xte_plus)[:, 1]
        roc1 = float(roc_auc_score(yte, s1_te))
        pr1 = float(average_precision_score(yte, s1_te))

        rep_rows.append(
            {
                "repeat_id": int(repeat_id),
                "train_size": int(len(tr)),
                "test_size": int(len(te)),
                "roc_clinical_only": roc0,
                "pr_clinical_only": pr0,
                "roc_clinical_plus_ensemble_oos": roc1,
                "pr_clinical_plus_ensemble_oos": pr1,
                "delta_roc": float(roc1 - roc0),
                "delta_pr": float(pr1 - pr0),
            }
        )

    df_rep = pd.DataFrame(rep_rows)
    out_rep = os.path.join(out_root, "repeats.csv")
    df_rep.to_csv(out_rep, index=False)

    roc0s = df_rep["roc_clinical_only"].to_numpy(dtype=np.float64)
    roc1s = df_rep["roc_clinical_plus_ensemble_oos"].to_numpy(dtype=np.float64)
    pr0s = df_rep["pr_clinical_only"].to_numpy(dtype=np.float64)
    pr1s = df_rep["pr_clinical_plus_ensemble_oos"].to_numpy(dtype=np.float64)

    sum_rows = [
        {
            "method": "clinical_only_age_sex_lr",
            **_summary_stats(roc0s),
            **{f"pr_{k}": v for k, v in _summary_stats(pr0s).items()},
        },
        {
            "method": "clinical_plus_ensemble_oos_lr",
            **_summary_stats(roc1s),
            **{f"pr_{k}": v for k, v in _summary_stats(pr1s).items()},
        },
    ]
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_root, "summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(roc0s, bins=20, color="#4e79a7", alpha=0.55, label="clinical only")
    ax.hist(roc1s, bins=20, color="#59a14f", alpha=0.55, label="clinical + ens OOS")
    ax.set_title(f"ROC-AUC over splits (n={args.n_splits})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "hist_roc_auc_compare.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(pr0s, bins=20, color="#4e79a7", alpha=0.55, label="clinical only")
    ax.hist(pr1s, bins=20, color="#59a14f", alpha=0.55, label="clinical + ens OOS")
    ax.set_title(f"PR-AUC over splits (n={args.n_splits})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "hist_pr_auc_compare.png"), dpi=150)
    plt.close(fig)

    meta = {
        "n_clinical_cohort": int(len(ds.y)),
        "n_oos_unique_patients": len(pids_latent),
        "fixed_w": float(args.fixed_w),
        "note_train_ensemble": (
            "Train uses leave-one-split-out mean ensemble score over other repeats' test rows; "
            "see script docstring."
        ),
    }
    import json

    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", out_root, file=sys.stderr)


if __name__ == "__main__":
    main()
