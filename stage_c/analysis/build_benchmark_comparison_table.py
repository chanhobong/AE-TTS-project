#!/usr/bin/env python3
"""
Aggregate benchmark rows into a single LaTeX/Excel-friendly CSV.

Kinds (see Report/benchmark_comparison_manifest.json):

- single_row_summary_csv: heads from repeated_head_sweep_std or repeated_stratified_shuffle_eval summary (1 row CSV).
- pick_best_from_summary_csv: repeated_stratified with multiple models — pick row by maximizing pick_metric (e.g. roc_mean).
- ensemble_method: ensemble_oos_same_split_eval summary_methods + per_split_metrics (adds ROC/PR quantiles tail from splits).

Outputs columns:
  branch_id, model, representation, classifier_or_fusion,
  ROC_mean, ROC_std, ROC_p2.5, PR_mean, PR_std, notes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


OUT_COLS = [
    "branch_id",
    "model",
    "representation",
    "classifier_or_fusion",
    "ROC_mean",
    "ROC_std",
    "ROC_p2p5",
    "PR_mean",
    "PR_std",
    "PR_p2p5",
    "notes",
]


def _q025(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, 0.025))


def _row_from_shuffle_summary(r: pd.Series, notes: str = "") -> dict[str, Any]:
    roc_p = float(r.get("roc_p2p5", np.nan))
    pr_p = float(r.get("pr_p2p5", np.nan))
    return {
        "ROC_mean": float(r["roc_mean"]),
        "ROC_std": float(r["roc_std"]),
        "ROC_p2p5": roc_p,
        "PR_mean": float(r["pr_mean"]),
        "PR_std": float(r["pr_std"]),
        "PR_p2p5": pr_p,
        "notes": notes.strip(),
    }


def _ensemble_roc_pr_columns(method_key: str) -> tuple[str, str]:
    if method_key == "plain_only":
        return "roc_plain", "pr_plain"
    if method_key == "monai_only":
        return "roc_monai", "pr_monai"
    m_oracle = re.match(r"^ensemble_fixed_w([\d.eE+-]+)$", method_key)
    if m_oracle:
        w = float(m_oracle.group(1))
        # filenames use minimal float formatting in practice
        wf = ("%g" % w).replace("+", "")
        return f"roc_ensemble_w{wf}", f"pr_ensemble_w{wf}"
    if method_key.startswith("ensemble_fixed_w") and "plus_age_sex_lr" in method_key:
        return "roc_ensemble_plus_age_sex_lr", "pr_ensemble_plus_age_sex_lr"
    if "oracle" in method_key.lower():
        return "roc_ensemble_oracle", "pr_ensemble_oracle"
    raise ValueError(f"Unknown ensemble method_key={method_key!r}")


def _load_manifest(path: str) -> dict[str, Any]:
    with open(os.path.abspath(path), encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest (see Report/benchmark_comparison_manifest.json)",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Output CSV path (default: Report/benchmark_comparison_table.csv beside manifest)",
    )
    args = ap.parse_args()

    man = _load_manifest(args.manifest)
    rows_conf: list[dict[str, Any]] = man["rows"]
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    out_path = os.path.abspath(args.out_csv.strip()) if str(args.out_csv).strip() else os.path.join(
        manifest_dir, "benchmark_comparison_table.csv"
    )

    merged: list[dict[str, Any]] = []

    for spec in rows_conf:
        bid = str(spec["id"])
        model = str(spec["model"])
        rep = str(spec["representation"])
        clf = str(spec["classifier_or_fusion"])
        kind = str(spec["kind"])

        notes = ""
        def _resolve_path(p: str) -> str:
            p = os.path.expanduser(p)
            if os.path.isabs(p):
                return os.path.abspath(p)
            return os.path.abspath(os.path.join(manifest_dir, p))

        if kind == "single_row_summary_csv":
            csv_path = _resolve_path(spec["path"])
            if not os.path.isfile(csv_path):
                merged.append(dict(zip(OUT_COLS, [bid, model, rep, clf, *[""] * 6, "MISSING CSV"])))  # type: ignore
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                merged.append(dict(zip(OUT_COLS, [bid, model, rep, clf, *[""] * 6, "EMPTY CSV"])))
                continue
            r = df.iloc[0]
            row = dict(zip(OUT_COLS[:4], [bid, model, rep, clf]))
            row.update(_row_from_shuffle_summary(r))
            merged.append(row)
            continue

        if kind == "pick_best_from_summary_csv":
            csv_path = _resolve_path(spec["path"])
            metric = str(spec.get("pick_metric", "roc_mean"))
            if not os.path.isfile(csv_path):
                merged.append(
                    dict(
                        zip(
                            OUT_COLS,
                            [bid, model, rep, clf, *[""] * 6, f"MISSING {csv_path}"],
                        )
                    )
                )
                continue
            df = pd.read_csv(csv_path)
            df = df.sort_values(metric, ascending=False)
            r = df.iloc[0]
            mname = str(r.get("model", ""))
            lsrc = str(r.get("latent_source", ""))
            pool = str(r.get("pooling", ""))
            bm = float(r[metric]) if metric in r.index else float("nan")
            notes = f"pick_best:{metric}; {lsrc} / {pool} / model={mname}"
            clf_out = f"{mname} ({metric}={bm:.4f})"

            row = dict(zip(OUT_COLS[:4], [bid, model, rep, clf_out]))
            row.update(_row_from_shuffle_summary(r, notes))
            merged.append(row)
            continue

        if kind == "ensemble_method":
            smp = _resolve_path(spec["summary_methods"])
            psp = _resolve_path(spec["per_split_metrics"])
            method_key = str(spec["method_key"])
            if not (os.path.isfile(smp) and os.path.isfile(psp)):
                merged.append(dict(zip(OUT_COLS, [bid, model, rep, clf, *[""] * 6, "MISSING ensemble CSV"])))
                continue

            sums = pd.read_csv(smp)
            row_summary = sums[sums["method"].astype(str) == method_key]
            if row_summary.empty:
                merged.append(
                    dict(zip(OUT_COLS, [bid, model, rep, clf, *[""] * 6, f"No method={method_key!r}"]))
                )
                continue
            rsum = row_summary.iloc[0]

            per = pd.read_csv(psp)
            roc_col, pr_col = _ensemble_roc_pr_columns(method_key)
            if roc_col not in per.columns or pr_col not in per.columns:
                merged.append(
                    dict(
                        zip(
                            OUT_COLS,
                            [
                                bid,
                                model,
                                rep,
                                clf,
                                *[""] * 6,
                                f"per_split missing cols {roc_col}/{pr_col}",
                            ],
                        )
                    )
                )
                continue

            roc_series = pd.to_numeric(per[roc_col], errors="coerce").to_numpy()
            pr_series = pd.to_numeric(per[pr_col], errors="coerce").to_numpy()

            row = dict(zip(OUT_COLS[:4], [bid, model, rep, clf]))
            row["ROC_mean"] = float(rsum["roc_mean"])
            row["ROC_std"] = float(rsum["roc_std"])
            row["ROC_p2p5"] = _q025(roc_series)
            row["PR_mean"] = float(rsum["pr_mean"])
            row["PR_std"] = float(rsum["pr_std"])
            row["PR_p2p5"] = _q025(pr_series)
            row["notes"] = f"ensemble:{method_key} from={os.path.basename(os.path.dirname(smp))}"
            merged.append(row)
            continue

        merged.append(dict(zip(OUT_COLS, [bid, model, rep, clf, *[""] * 6, f"Unknown kind={kind!r}"])))

    df_out = pd.DataFrame.from_records(merged, columns=OUT_COLS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {len(df_out)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
