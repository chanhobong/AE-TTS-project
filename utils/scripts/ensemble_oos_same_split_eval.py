#!/usr/bin/env python3
"""
Ensemble out-of-sample probabilities **only when Plain and MONAI rows share the same
split** (same ``repeat_id`` / ``split_id``) and the **same test ``patient_id``**.

- Inner join on ``(split_col, patient_id)``.
- Optionally verify labels match between the two files.
- **Never** mixes train rows: expects probability CSVs that list **test patients only**
  per split (or full tables that you pre-filter to test).

Reporting (addresses test-set w tuning honestly)
-----------------------------------------------
1. **plain_only** — ROC/PR from model A probabilities only.
2. **monai_only** — from model B only.
3. **ensemble_fixed** — ``w = --fixed_w`` (default 0.5), chosen **without** scanning test.
4. **ensemble_oracle** — for each split, ``w`` in ``--w_grid`` that maximises test ROC
   (exploratory / optimistic; label as such in output).
5. **ensemble_fixed + age_sex_lr** (optional ``--clinical_labels_csv``) — same fixed-w ensemble
   probability as a feature with age + sex; **train** fold uses leave-one-split-out mean ensemble
   score from other repeats (see meta); **test** uses this split's OOS ensemble prob. Reported like
   other methods in ``summary_methods.csv`` / ``per_split_metrics.csv``.

Per split, for each ``w`` in ``w_grid`` we also store ROC-AUC and PR-AUC (optional wide export).

Input CSV columns (each file)
-----------------------------
Required:
  - ``repeat_id`` or ``split_id`` (see ``--split_col``)
  - ``patient_id``
  - one probability column (``--prob_col_plain``, ``--prob_col_monai``)

Optional:
  - ``label`` (0/1). If absent in both, pass ``--label_csv`` for patient-level labels.

Example
-------
From ``repeated_stratified_shuffle_eval.py`` with ``--write_test_predictions``, split long CSV by ``model``::

  # Plain (e.g. logistic) and MONAI (e.g. rbf_svc) must use identical:
  #   --n_splits --test_size --random_state --labels_csv --pooling --concat_age_sex flags
  # and the same patient NPZ cohort ordering is implied by the same label table + NPZ dirs.

  python3 utils/scripts/ensemble_oos_same_split_eval.py \\
    --csv_plain plain_run/.../test_oos_predictions.csv \\
    --csv_monai monai_run/.../test_oos_predictions.csv \\
    --filter_model_plain logistic --filter_model_monai rbf_svc \\
    --require_split_protocol_match \\
    --prob_col_plain prob_tts --prob_col_monai prob_tts \\
    --out_dir latent_data/ensemble_same_split_eval

If each ``test_oos_predictions.csv`` contains multiple ``model`` rows (``--classifier all``),
pre-filter so each file is one model, e.g. ``df[df.model == 'logistic']`` vs ``rbf_svc``.

Legacy (pre-made prob CSVs)::

  python3 utils/scripts/ensemble_oos_same_split_eval.py \\
    --csv_plain path/plain_oos_probs.csv \\
    --csv_monai path/monai_oos_probs.csv \\
    --prob_col_plain prob_tts --prob_col_monai prob_tts \\
    --label_csv data/train.csv \\
    --out_dir latent_data/ensemble_same_split_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from clinical_only_repeated_eval import _merge_labels_and_clinical  # noqa: E402


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_w_grid(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty w_grid")
    w = np.array(vals, dtype=np.float64)
    if np.any(w < 0) or np.any(w > 1):
        raise ValueError("w_grid values must be in [0, 1]")
    return w


def _auc_pair(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _attach_label_from_csv(m: pd.DataFrame, label_path: str, id_col: str, lab_col: str) -> pd.DataFrame:
    lab = _read(label_path)
    lc = lab_col if lab_col in lab.columns else ("case" if "case" in lab.columns else lab_col)
    if id_col not in lab.columns or lc not in lab.columns:
        raise ValueError(f"{label_path}: need {id_col} and label/case; got {list(lab.columns)}")
    sub = lab[[id_col, lc]].copy()
    sub[id_col] = sub[id_col].astype(str)
    sub["label_merge"] = pd.to_numeric(sub[lc], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=["label_merge"])
    sub["label_merge"] = sub["label_merge"].astype(int)
    sub = sub[[id_col, "label_merge"]].drop_duplicates(subset=[id_col], keep="last")
    out = m.merge(sub, left_on=id_col, right_on=id_col, how="inner")
    return out


def _coerce_split_col(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(np.int64)
    return s.astype(str)


PROTOCOL_COLS = (
    "n_splits",
    "test_size",
    "split_random_state",
    "pooling",
    "cluster_hist_bins",
    "concat_age_sex",
    "final_rescale_after_concat",
)

PROTOCOL_SPLIT_DRIVER_COLS = (
    "n_splits",
    "test_size",
    "split_random_state",
    "concat_age_sex",
    "final_rescale_after_concat",
)


def _proto_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return True
        return bool(np.isclose(fa, fb, rtol=0.0, atol=1e-9))
    return str(a).strip() == str(b).strip()


def _one_protocol_val(s: pd.Series) -> Any:
    u = pd.unique(s.dropna())
    if len(u) == 0:
        return None
    if len(u) != 1:
        raise ValueError(f"non-unique values: {sorted(map(str, u.tolist()))[:12]}")
    return u[0]


def _assert_protocol_match(
    da: pd.DataFrame, db: pd.DataFrame, a_name: str, b_name: str, cols: tuple[str, ...]
) -> None:
    mismatches: list[str] = []
    for col in cols:
        in_a = col in da.columns
        in_b = col in db.columns
        if not in_a and not in_b:
            continue
        if in_a ^ in_b:
            mismatches.append(f"{col}: present only in {'plain' if in_a else 'monai'}")
            continue
        try:
            va = _one_protocol_val(da[col])
            vb = _one_protocol_val(db[col])
        except ValueError as e:
            mismatches.append(f"{col}: {e}")
            continue
        if not _proto_equal(va, vb):
            mismatches.append(f"{col}: {a_name}={va!r} vs {b_name}={vb!r}")
    if mismatches:
        raise SystemExit(
            "Split/protocol metadata mismatch between Plain and MONAI exports:\n  - "
            + "\n  - ".join(mismatches)
        )


def _apply_model_filter(d: pd.DataFrame, want: str, which: str) -> pd.DataFrame:
    want = want.strip()
    if not want:
        return d
    if "model" not in d.columns:
        raise SystemExit(f"{which}: --filter_model_*={want!r} but no 'model' column; columns={list(d.columns)}")
    out = d[d["model"].astype(str) == want].copy()
    if out.empty:
        raise SystemExit(f"{which}: no rows after model filter {want!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Same-split OOS ensemble evaluation (Plain + MONAI).")
    ap.add_argument("--csv_plain", required=True)
    ap.add_argument("--csv_monai", required=True)
    ap.add_argument("--prob_col_plain", default="prob_tts")
    ap.add_argument("--prob_col_monai", default="prob_tts")
    ap.add_argument("--split_col", default="repeat_id", help="Column name aligning splits (e.g. repeat_id, split_id)")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument(
        "--label_csv",
        default="",
        help="If probs files lack label: merge label by patient_id from this CSV (train+val+test union typical).",
    )
    ap.add_argument("--label_col", default="label")
    ap.add_argument(
        "--w_grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Weights for scanning (model A weight). Also used for oracle best-w per split.",
    )
    ap.add_argument("--fixed_w", type=float, default=0.5, help="Fixed ensemble w (no test peeking).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", default="")
    ap.add_argument(
        "--filter_model_plain",
        default="",
        help="If set and csv_plain has column 'model', keep only these rows (e.g. logistic).",
    )
    ap.add_argument(
        "--filter_model_monai",
        default="",
        help="If set and csv_monai has column 'model', keep only these rows (e.g. rbf_svc).",
    )
    ap.add_argument(
        "--require_split_protocol_match",
        action="store_true",
        help=(
            "Require matching protocol metadata between the two files when those columns exist "
            "(see --split_driver_protocol_only)."
        ),
    )
    ap.add_argument(
        "--split_driver_protocol_only",
        action="store_true",
        help=(
            "With --require_split_protocol_match: only compare n_splits, test_size, split_random_state, "
            "concat_age_sex, final_rescale_after_concat (ignore pooling / cluster_hist_bins). "
            "Use when Plain uses trajectory pooling and MONAI uses mean/std, but the same repeated split RNG."
        ),
    )
    ap.add_argument(
        "--clinical_labels_csv",
        action="append",
        default=None,
        help=(
            "Repeat for train/val/test CSVs (patient_id, age, sex, label). Restricts rows to patients with "
            "valid clinical fields, then adds summary method ensemble_fixed_w*__plus_age_sex_lr and per-split "
            "roc_ensemble_plus_age_sex_lr / pr_ensemble_plus_age_sex_lr: LogisticRegression on [ens, age, sex] "
            "with train-fold ens = mean OOS ensemble score on other splits (same as add-on clinical pipeline)."
        ),
    )
    ap.add_argument(
        "--export_clinical_stack_audit",
        action="store_true",
        help=(
            "Requires --clinical_labels_csv. Writes per_patient_clinical_stack_audit.csv (every test row) and "
            "clinical_stack_audit_per_split.csv / clinical_stack_audit_summary.csv: ensemble wrong vs ambiguous "
            "vs stack recovery at 0.5 threshold."
        ),
    )
    ap.add_argument(
        "--ensemble_ambiguous_margin",
        type=float,
        default=0.15,
        help="Audit: |prob_ensemble-0.5| < this ⇒ ensemble_ambiguous (default 0.15).",
    )
    args = ap.parse_args()

    if bool(args.export_clinical_stack_audit) and not bool(args.clinical_labels_csv):
        raise SystemExit("--export_clinical_stack_audit requires --clinical_labels_csv.")

    idc = str(args.patient_id_col)
    sc = str(args.split_col)
    cp = str(args.prob_col_plain)
    cm = str(args.prob_col_monai)

    da = _apply_model_filter(_read(args.csv_plain), str(args.filter_model_plain), "csv_plain")
    db = _apply_model_filter(_read(args.csv_monai), str(args.filter_model_monai), "csv_monai")
    if args.require_split_protocol_match:
        pcols: tuple[str, ...] = PROTOCOL_SPLIT_DRIVER_COLS if args.split_driver_protocol_only else PROTOCOL_COLS
        _assert_protocol_match(da, db, "csv_plain", "csv_monai", pcols)
    for name, d in (("csv_plain", da), ("csv_monai", db)):
        if sc not in d.columns:
            raise SystemExit(f"{name} missing split column {sc!r}; columns={list(d.columns)}")
        if idc not in d.columns:
            raise SystemExit(f"{name} missing {idc!r}")

    if cp not in da.columns:
        raise SystemExit(f"csv_plain missing {cp!r}; columns={list(da.columns)}")
    if cm not in db.columns:
        raise SystemExit(f"csv_monai missing {cm!r}; columns={list(db.columns)}")

    cols_a: list[str] = [sc, idc, cp]
    if "label" in da.columns:
        cols_a.append("label")
    a = da[cols_a].copy()
    a[idc] = a[idc].astype(str)
    a[sc] = _coerce_split_col(a[sc])
    if "label" in a.columns:
        a = a.rename(columns={"label": "label_plain"})
        a["label_plain"] = pd.to_numeric(a["label_plain"], errors="coerce")

    cols_b: list[str] = [sc, idc, cm]
    if "label" in db.columns:
        cols_b.append("label")
    b = db[cols_b].copy()
    b[idc] = b[idc].astype(str)
    b[sc] = _coerce_split_col(b[sc])
    if "label" in b.columns:
        b = b.rename(columns={"label": "label_monai"})
        b["label_monai"] = pd.to_numeric(b["label_monai"], errors="coerce")

    # If both sides use the same score column name (e.g. prob_tts), merge() would emit prob_tts_x / prob_tts_y
    # and a single rename dict would miss them — rename to disjoint keys before join.
    _pa, _pb = "__ensemble_prob_plain__", "__ensemble_prob_monai__"
    a = a.rename(columns={cp: _pa})
    b = b.rename(columns={cm: _pb})

    m = a.merge(b, on=[sc, idc], how="inner")
    if "label_plain" in m.columns and "label_monai" in m.columns:
        lp = np.asarray(m["label_plain"], dtype=np.float64)
        lm = np.asarray(m["label_monai"], dtype=np.float64)
        bad = ~(np.isfinite(lp) & np.isfinite(lm) & (lp == lm))
        if np.any(bad):
            raise SystemExit(
                f"Label mismatch for {int(np.sum(bad))} rows after join on [{sc}, {idc}]. Check inputs."
            )
        m["label"] = m["label_plain"].astype(int)
        m = m.drop(columns=["label_plain", "label_monai"])
    elif "label_plain" in m.columns:
        m["label"] = m["label_plain"].astype(int)
        m = m.drop(columns=["label_plain"])
    elif "label_monai" in m.columns:
        m["label"] = m["label_monai"].astype(int)
        m = m.drop(columns=["label_monai"])
    else:
        if not args.label_csv.strip():
            raise SystemExit("No label in probability CSVs; pass --label_csv.")
        m = _attach_label_from_csv(m, args.label_csv, idc, str(args.label_col))
        m["label"] = m["label_merge"].astype(int)
        m = m.drop(columns=["label_merge"], errors="ignore")

    m = m.rename(columns={_pa: "prob_plain", _pb: "prob_monai"})
    m["prob_plain"] = pd.to_numeric(m["prob_plain"], errors="coerce")
    m["prob_monai"] = pd.to_numeric(m["prob_monai"], errors="coerce")
    m = m.dropna(subset=["prob_plain", "prob_monai", "label"])

    w_grid = _parse_w_grid(str(args.w_grid))
    fixed_w = float(np.clip(args.fixed_w, 0.0, 1.0))

    n_joined_plain_monai = int(len(m))
    global_ens_mean: float | None = None
    do_clinical = bool(args.clinical_labels_csv)
    if do_clinical:
        clin = _merge_labels_and_clinical([os.path.abspath(p) for p in args.clinical_labels_csv])
        cm = clin[["patient_id", "age", "sex_bin"]].rename(columns={"patient_id": idc})
        m = m.merge(cm, on=idc, how="inner")
        m = m.dropna(subset=["age", "sex_bin"])
        if m.empty:
            raise SystemExit("No rows left after merging --clinical_labels_csv (need age + sex_bin).")
        m["age"] = pd.to_numeric(m["age"], errors="coerce")
        m["sex_bin"] = pd.to_numeric(m["sex_bin"], errors="coerce")
        m = m.dropna(subset=["age", "sex_bin"])
        if m.empty:
            raise SystemExit("No rows left after coercing age / sex_bin.")
        m["ens"] = fixed_w * m["prob_plain"] + (1.0 - fixed_w) * m["prob_monai"]
        global_ens_mean = float(m["ens"].mean())

    out_dir = Path(os.path.abspath(args.out_dir))
    tag = args.run_tag.strip() or None
    sub = out_dir / (f"run_{tag}" if tag else "ensemble_same_split")
    sub.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "csv_plain": os.path.abspath(args.csv_plain),
        "csv_monai": os.path.abspath(args.csv_monai),
        "filter_model_plain": str(args.filter_model_plain) or None,
        "filter_model_monai": str(args.filter_model_monai) or None,
        "require_split_protocol_match": bool(args.require_split_protocol_match),
        "split_driver_protocol_only": bool(args.split_driver_protocol_only),
        "split_col": sc,
        "n_rows_joined_plain_monai_inner": n_joined_plain_monai,
        "n_rows_joined": int(len(m)),
        "n_splits": int(m[sc].nunique()),
        "w_grid": w_grid.tolist(),
        "fixed_w": fixed_w,
        "note_oracle": "ensemble_oracle picks w per split on test ROC — optimistic / exploratory only.",
    }
    if do_clinical:
        meta["clinical_labels_csv"] = [os.path.abspath(p) for p in args.clinical_labels_csv]
        meta["ensemble_plus_age_sex_lr"] = {
            "classifier": "LogisticRegression(class_weight=balanced, lbfgs, max_iter=8000)",
            "train_ensemble_feature": (
                "Leave-one-split-out mean of fixed-w ensemble prob on other splits' OOS rows for that patient; "
                "fallback global mean if missing."
            ),
            "scaling": "MinMaxScaler on train for ensemble score and age; sex already 0/1.",
        }
    if do_clinical and bool(args.export_clinical_stack_audit):
        meta["clinical_stack_audit"] = {
            "threshold": 0.5,
            "ensemble_ambiguous_margin": float(args.ensemble_ambiguous_margin),
            "files": [
                "per_patient_clinical_stack_audit.csv",
                "clinical_stack_audit_per_split.csv",
                "clinical_stack_audit_summary.csv",
            ],
        }
    (sub / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    per_rows: list[dict[str, Any]] = []
    wid_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def _sort_key(v: Any) -> tuple:
        if isinstance(v, (int, np.integer)):
            return (0, int(v))
        if isinstance(v, float) and float(v).is_integer():
            return (0, int(v))
        return (1, str(v))

    for sid in sorted(m[sc].unique(), key=_sort_key):
        subm = m[m[sc] == sid]
        y = subm["label"].to_numpy(dtype=np.int64)
        pa = subm["prob_plain"].to_numpy(dtype=np.float64)
        pb = subm["prob_monai"].to_numpy(dtype=np.float64)

        roc_a, pr_a = _auc_pair(y, pa)
        roc_b, pr_b = _auc_pair(y, pb)
        p_fix = fixed_w * pa + (1.0 - fixed_w) * pb
        roc_fix, pr_fix = _auc_pair(y, p_fix)

        roc_clin_lr = float("nan")
        pr_clin_lr = float("nan")
        s_st_te: np.ndarray | None = None
        if do_clinical:
            assert global_ens_mean is not None
            gm = float(global_ens_mean)

            def _loo_ens(p_pid: str) -> float:
                rows = m[(m[idc].astype(str) == str(p_pid)) & (m[sc].astype(str) != str(sid))]
                if rows.empty:
                    return gm
                return float(rows["ens"].mean())

            test_pids = subm[idc].astype(str).tolist()
            gcohort = set(m[idc].astype(str).unique())
            train_pids = sorted(gcohort - set(test_pids))
            if len(set(test_pids)) != len(subm):
                raise SystemExit(
                    f"{sc}={sid!r}: duplicate test patient rows after clinical merge (check inputs)."
                )

            aux = m[[idc, "age", "sex_bin", "label"]].drop_duplicates(subset=[idc]).copy()
            aux[idc] = aux[idc].astype(str)
            aux = aux.set_index(idc)

            ytr = aux.loc[train_pids, "label"].to_numpy(dtype=np.int64)
            age_tr = aux.loc[train_pids, "age"].to_numpy(dtype=np.float64)
            sex_tr = aux.loc[train_pids, "sex_bin"].to_numpy(dtype=np.float64).reshape(-1, 1)
            ens_tr = np.asarray([_loo_ens(p) for p in train_pids], dtype=np.float64)

            yte = subm["label"].to_numpy(dtype=np.int64)
            ens_te = subm["ens"].to_numpy(dtype=np.float64)
            age_te = subm["age"].to_numpy(dtype=np.float64)
            sex_te = subm["sex_bin"].to_numpy(dtype=np.float64).reshape(-1, 1)

            if np.unique(ytr).size < 2:
                roc_clin_lr, pr_clin_lr = float("nan"), float("nan")
            else:
                mm_e = MinMaxScaler()
                mm_a = MinMaxScaler()
                ens_tr_c = mm_e.fit_transform(ens_tr.reshape(-1, 1))
                ens_te_c = mm_e.transform(ens_te.reshape(-1, 1))
                age_tr_s = mm_a.fit_transform(age_tr.reshape(-1, 1))
                age_te_s = mm_a.transform(age_te.reshape(-1, 1))
                xtr = np.concatenate([ens_tr_c, age_tr_s, sex_tr], axis=1)
                xte = np.concatenate([ens_te_c, age_te_s, sex_te], axis=1)
                clf_st = LogisticRegression(
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                )
                clf_st.fit(xtr, ytr)
                s_st_te = clf_st.predict_proba(xte)[:, 1]
                roc_clin_lr, pr_clin_lr = _auc_pair(yte, s_st_te)

            if bool(args.export_clinical_stack_audit) and s_st_te is not None:
                margin = float(args.ensemble_ambiguous_margin)
                pid_te_arr = subm[idc].astype(str).to_numpy()
                p_ens_col = f"prob_ensemble_w{fixed_w:g}"
                for i in range(len(subm)):
                    yt = int(yte[i])
                    pe = float(ens_te[i])
                    ps = float(s_st_te[i])
                    pred_e = 1 if pe >= 0.5 else 0
                    pred_s = 1 if ps >= 0.5 else 0
                    e_wrong = pred_e != yt
                    e_ambig = abs(pe - 0.5) < margin
                    s_wrong = pred_s != yt
                    closer = (yt == 1 and ps > pe) or (yt == 0 and ps < pe)
                    audit_rows.append(
                        {
                            sc: sid,
                            idc: pid_te_arr[i],
                            "label": yt,
                            "prob_plain": float(pa[i]),
                            "prob_monai": float(pb[i]),
                            p_ens_col: pe,
                            "prob_clinical_stack": ps,
                            "ensemble_pred_ge_half": int(pred_e),
                            "stack_pred_ge_half": int(pred_s),
                            "ensemble_wrong": int(e_wrong),
                            "ensemble_ambiguous": int(e_ambig),
                            "stack_wrong": int(s_wrong),
                            "wrong_or_ambiguous": int(e_wrong or e_ambig),
                            "fix_wrong_or_ambiguous": int((e_wrong or e_ambig) and (not s_wrong)),
                            "recovery_wrong_to_right": int(e_wrong and (not s_wrong)),
                            "ambiguous_only_stack_correct": int((not e_wrong) and e_ambig and (not s_wrong)),
                            "stack_prob_moves_toward_label": int(closer),
                            "regression_right_to_wrong": int((not e_wrong) and s_wrong),
                        }
                    )

        roc_w: list[float] = []
        pr_w: list[float] = []
        for w in w_grid:
            p_ens = float(w) * pa + (1.0 - float(w)) * pb
            roc_e, pr_e = _auc_pair(y, p_ens)
            roc_w.append(roc_e)
            pr_w.append(pr_e)
            wid_rows.append(
                {
                    sc: sid,
                    "w": float(w),
                    "roc_auc_ensemble": roc_e,
                    "pr_auc_ensemble": pr_e,
                }
            )

        arr_roc = np.array(roc_w, dtype=np.float64)
        best_i = int(np.nanargmax(arr_roc)) if np.any(np.isfinite(arr_roc)) else 0
        best_w = float(w_grid[best_i])
        roc_oracle = float(arr_roc[best_i]) if np.isfinite(arr_roc[best_i]) else float("nan")
        pr_oracle = float(pr_w[best_i]) if np.isfinite(pr_w[best_i]) else float("nan")

        row: dict[str, Any] = {
            sc: sid,
            "n_test": int(len(subm)),
            "roc_plain": roc_a,
            "pr_plain": pr_a,
            "roc_monai": roc_b,
            "pr_monai": pr_b,
            f"roc_ensemble_w{fixed_w:g}": roc_fix,
            f"pr_ensemble_w{fixed_w:g}": pr_fix,
            "roc_ensemble_oracle": roc_oracle,
            "pr_ensemble_oracle": pr_oracle,
            "oracle_best_w": best_w,
        }
        if do_clinical:
            row["roc_ensemble_plus_age_sex_lr"] = roc_clin_lr
            row["pr_ensemble_plus_age_sex_lr"] = pr_clin_lr
        per_rows.append(row)

    df_per = pd.DataFrame(per_rows)
    df_per.to_csv(sub / "per_split_metrics.csv", index=False)

    pd.DataFrame(wid_rows).to_csv(sub / "per_split_w_grid_metrics.csv", index=False)

    def _summ(name: str, roc_col: str, pr_col: str) -> dict[str, Any]:
        x = df_per[roc_col].to_numpy(dtype=np.float64)
        z = df_per[pr_col].to_numpy(dtype=np.float64)
        return {
            "method": name,
            "n_splits_valid_roc": int(np.sum(np.isfinite(x))),
            "roc_mean": float(np.nanmean(x)),
            "roc_std": float(np.nanstd(x, ddof=0)),
            "pr_mean": float(np.nanmean(z)),
            "pr_std": float(np.nanstd(z, ddof=0)),
        }

    summ = [
        _summ("plain_only", "roc_plain", "pr_plain"),
        _summ("monai_only", "roc_monai", "pr_monai"),
        _summ(f"ensemble_fixed_w{fixed_w:g}", f"roc_ensemble_w{fixed_w:g}", f"pr_ensemble_w{fixed_w:g}"),
    ]
    if do_clinical:
        summ.append(
            _summ(
                f"ensemble_fixed_w{fixed_w:g}_plus_age_sex_lr",
                "roc_ensemble_plus_age_sex_lr",
                "pr_ensemble_plus_age_sex_lr",
            )
        )
    summ.append(_summ("ensemble_oracle_test_w (exploratory)", "roc_ensemble_oracle", "pr_ensemble_oracle"))
    df_sum = pd.DataFrame(summ)
    df_sum.to_csv(sub / "summary_methods.csv", index=False)

    if do_clinical and bool(args.export_clinical_stack_audit) and audit_rows:
        df_a = pd.DataFrame(audit_rows)
        df_a.to_csv(sub / "per_patient_clinical_stack_audit.csv", index=False)
        margin = float(args.ensemble_ambiguous_margin)
        interesting = df_a[df_a["wrong_or_ambiguous"] == 1]
        wrong_only = df_a[df_a["ensemble_wrong"] == 1]
        ambig_only = df_a[(df_a["ensemble_ambiguous"] == 1) & (df_a["ensemble_wrong"] == 0)]
        summ_lines: list[dict[str, Any]] = [
            {
                "audit": "global",
                "threshold_pred": 0.5,
                "ensemble_ambiguous_margin": margin,
                "n_test_rows": int(len(df_a)),
                "n_ensemble_wrong": int(df_a["ensemble_wrong"].sum()),
                "n_ensemble_ambiguous": int(df_a["ensemble_ambiguous"].sum()),
                "n_wrong_or_ambiguous": int(df_a["wrong_or_ambiguous"].sum()),
                "n_recovery_wrong_to_right": int(df_a["recovery_wrong_to_right"].sum()),
                "frac_of_wrong_recovered": float(df_a["recovery_wrong_to_right"].sum() / max(int(wrong_only.shape[0]), 1)),
                "n_interesting_stack_correct": int((interesting["stack_wrong"] == 0).sum()),
                "n_fix_wrong_or_ambiguous": int(df_a["fix_wrong_or_ambiguous"].sum()),
                "frac_interesting_stack_correct": float((interesting["stack_wrong"] == 0).sum() / max(len(interesting), 1)),
                "n_ambiguous_only_then_stack_correct": int(ambig_only[ambig_only["stack_wrong"] == 0].shape[0]),
                "n_ambiguous_only_total": int(len(ambig_only)),
                "n_regression_right_to_wrong": int(df_a["regression_right_to_wrong"].sum()),
            }
        ]
        pd.DataFrame(summ_lines).to_csv(sub / "clinical_stack_audit_summary.csv", index=False)

        def _agg_audit(g: pd.DataFrame) -> pd.Series:
            wr = int(g["ensemble_wrong"].sum())
            amb = int(g["ensemble_ambiguous"].sum())
            tot = int(len(g))
            interest = g[g["wrong_or_ambiguous"] == 1]
            rec = int(g["recovery_wrong_to_right"].sum())
            return pd.Series(
                {
                    "n_test": tot,
                    "n_ensemble_wrong": wr,
                    "n_ensemble_ambiguous": amb,
                    "n_wrong_or_ambiguous": int(g["wrong_or_ambiguous"].sum()),
                    "n_recovery_wrong_to_right": rec,
                    "frac_recovery_of_wrong": float(rec / max(wr, 1)),
                    "n_interesting_stack_correct": int((interest["stack_wrong"] == 0).sum()),
                    "n_fix_wrong_or_ambiguous": int(g["fix_wrong_or_ambiguous"].sum()),
                    "frac_interesting_stack_correct": float((interest["stack_wrong"] == 0).sum() / max(len(interest), 1)),
                    "n_regression": int(g["regression_right_to_wrong"].sum()),
                }
            )

        parts_ps: list[dict[str, Any]] = []
        for sid_val, g in df_a.groupby(sc, sort=False):
            row = _agg_audit(g).to_dict()
            row[sc] = sid_val
            parts_ps.append(row)
        per_sp = pd.DataFrame(parts_ps)
        per_sp.to_csv(sub / "clinical_stack_audit_per_split.csv", index=False)
    elif do_clinical and bool(args.export_clinical_stack_audit):
        print("[warn] clinical stack audit: no rows (e.g. every split had single-class train).", file=sys.stderr)

    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(df_sum.to_string(index=False))
    print("Wrote", sub, file=sys.stderr)


if __name__ == "__main__":
    main()
