#!/usr/bin/env python3
"""
Weighted soft ensemble of two patient-level scores:

    p_ens = w * score_a + (1 - w) * score_b

Typical use: Plain AE (ElasticNet) prob + MONAI RBF-SVC prob, same patients / same eval cohort.

Inputs
------
Either:
  - one wide CSV: patient_id + score_a + score_b [+ label], or
  - two CSVs merged on patient_id (--csv_a / --csv_b).

Outputs
-------
Prints (and optionally saves JSON) ROC-AUC / PR-AUC for model A, B, and ensemble per weight.

Example
-------
  python3 utils/scripts/weighted_soft_ensemble_scores.py \\
    --csv_a plain_test_probs.csv --score_col_a prob_tts \\
    --csv_b monai_test_probs.csv --score_col_b prob_tts \\
    --label_csv data/test.csv \\
    --w_grid 0,0.25,0.5,0.75,1 \\
    --out_json latent_data/ensemble_scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _merge_two(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    id_col: str,
    col_a: str,
    col_b: str,
) -> pd.DataFrame:
    da = a[[id_col, col_a]].rename(columns={col_a: "score_a"})
    db = b[[id_col, col_b]].rename(columns={col_b: "score_b"})
    da[id_col] = da[id_col].astype(str)
    db[id_col] = db[id_col].astype(str)
    return da.merge(db, on=id_col, how="inner")


def _attach_label(m: pd.DataFrame, label_path: str, id_col: str, label_col: str) -> pd.DataFrame:
    lab = _read(label_path)
    lc = label_col
    if lc not in lab.columns and "case" in lab.columns:
        lc = "case"
    if id_col not in lab.columns or lc not in lab.columns:
        raise ValueError(f"label file needs {id_col} and {lc}; got {list(lab.columns)}")
    sub = lab[[id_col, lc]].copy()
    sub[id_col] = sub[id_col].astype(str)
    sub["label"] = pd.to_numeric(sub[lc], errors="coerce")
    sub = sub.dropna(subset=["label"])
    sub["label"] = sub["label"].astype(int)
    sub = sub[[id_col, "label"]].drop_duplicates(subset=[id_col], keep="last")
    out = m.merge(sub, on=id_col, how="inner")
    return out


def _parse_w_grid(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    if not out:
        raise ValueError("empty w_grid")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Weighted soft ensemble p = w*a + (1-w)*b")
    ap.add_argument("--csv_merged", default="", help="Wide table: patient_id + score_a + score_b")
    ap.add_argument("--csv_a", default="")
    ap.add_argument("--csv_b", default="")
    ap.add_argument("--score_col_a", default="score_a")
    ap.add_argument("--score_col_b", default="score_b")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument("--label_csv", default="", help="For AUC: patient_id + label (or case)")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--w", type=float, default=None, help="Single weight (ignored if --w_grid set)")
    ap.add_argument("--w_grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1", help="Comma-separated w values")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    idc = str(args.patient_id_col)

    if args.csv_merged.strip():
        df = _read(args.csv_merged)
        if idc not in df.columns:
            raise SystemExit(f"{idc} missing in --csv_merged; columns={list(df.columns)}")
        sa, sb = str(args.score_col_a), str(args.score_col_b)
        if sa not in df.columns or sb not in df.columns:
            raise SystemExit(f"Need columns {sa}, {sb} in merged CSV; got {list(df.columns)}")
        m = df[[idc, sa, sb]].rename(columns={sa: "score_a", sb: "score_b"}).copy()
    else:
        if not args.csv_a.strip() or not args.csv_b.strip():
            raise SystemExit("Provide --csv_merged OR both --csv_a and --csv_b")
        m = _merge_two(
            _read(args.csv_a),
            _read(args.csv_b),
            id_col=idc,
            col_a=str(args.score_col_a),
            col_b=str(args.score_col_b),
        )

    m[idc] = m[idc].astype(str)
    m["score_a"] = pd.to_numeric(m["score_a"], errors="coerce")
    m["score_b"] = pd.to_numeric(m["score_b"], errors="coerce")
    m = m.dropna(subset=["score_a", "score_b"])
    if len(m) < 5:
        raise SystemExit(f"Too few rows after merge/clean: {len(m)}")

    y: np.ndarray | None = None
    if args.label_csv.strip():
        m = _attach_label(m, args.label_csv, idc, str(args.label_col))
        y = m["label"].to_numpy(dtype=np.int64)
        if np.unique(y).size < 2:
            raise SystemExit("Labels need two classes for AUC.")

    if args.w is not None:
        ws = [float(args.w)]
    else:
        ws = _parse_w_grid(str(args.w_grid))

    rows: list[dict] = []
    xa = m["score_a"].to_numpy(dtype=np.float64)
    xb = m["score_b"].to_numpy(dtype=np.float64)

    for w in ws:
        w = float(np.clip(w, 0.0, 1.0))
        p_ens = w * xa + (1.0 - w) * xb
        rec: dict = {"w": w, "n": int(len(m))}
        if y is not None:
            rec["roc_auc_a"] = float(roc_auc_score(y, xa))
            rec["pr_auc_a"] = float(average_precision_score(y, xa))
            rec["roc_auc_b"] = float(roc_auc_score(y, xb))
            rec["pr_auc_b"] = float(average_precision_score(y, xb))
            rec["roc_auc_ensemble"] = float(roc_auc_score(y, p_ens))
            rec["pr_auc_ensemble"] = float(average_precision_score(y, p_ens))
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", 30, "display.width", 120):
        print(df_out.to_string(index=False))

    if args.out_json:
        p = Path(args.out_json).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights_scan": rows, "patient_id_col": idc, "n_matched": int(len(m))}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Wrote", p, file=sys.stderr)

    if y is not None:
        best_idx = int(np.argmax(df_out["roc_auc_ensemble"].to_numpy()))
        print(
            f"\nBest ROC-AUC ensemble: w={df_out.iloc[best_idx]['w']:.4g} "
            f"-> ROC={df_out.iloc[best_idx]['roc_auc_ensemble']:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
