#!/usr/bin/env python3
"""
Merge run_* outputs into repeated_summary_merged_with_fixed.csv (same method as before).

It searches under out_parent/run_*:
  - all **/summary.csv to build repeated_summary_all.csv (optional)
  - all run_*/compare_fixed_test_positions.csv to build repeated_compare_fixed_all.csv (optional)
  - merged to repeated_summary_merged_with_fixed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge repeated run_* summaries into repeated_summary_merged_with_fixed.csv")
    ap.add_argument("--out_parent", required=True, help="Directory containing run_* subdirs")
    args = ap.parse_args()

    out_parent = Path(args.out_parent).resolve()
    run_dirs = sorted([p for p in out_parent.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run_* dirs under {out_parent}")

    summaries = []
    fixed_parts = []
    for run_dir in run_dirs:
        for p in run_dir.rglob("summary.csv"):
            summaries.append(pd.read_csv(p))
        fx = run_dir / "compare_fixed_test_positions.csv"
        if fx.exists():
            fixed_parts.append(pd.read_csv(fx))

    if not summaries:
        raise SystemExit("No summary.csv found under run_* dirs")
    df_sum = pd.concat(summaries, ignore_index=True)
    (out_parent / "repeated_summary_all.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")

    merged = df_sum
    if fixed_parts:
        df_fx = pd.concat(fixed_parts, ignore_index=True)
        (out_parent / "repeated_compare_fixed_all.csv").write_text(df_fx.to_csv(index=False), encoding="utf-8")
        keys = ["latent_source", "pooling", "model"]
        roc_part = (
            df_fx.dropna(subset=["fixed_test_roc_auc"])[keys + ["fixed_test_roc_auc", "cdf_position"]]
            .drop_duplicates(keys)
        )
        pr_part = (
            df_fx.dropna(subset=["fixed_test_pr_auc"])[keys + ["fixed_test_pr_auc", "pr_cdf_position"]]
            .drop_duplicates(keys)
        )
        df_fixed = roc_part.merge(pr_part, on=keys, how="outer")
        merged = df_sum.merge(df_fixed, on=keys, how="left")

    front = ["latent_source", "model", "pooling"]
    cols = list(merged.columns)
    ordered = [c for c in front if c in cols] + [c for c in cols if c not in front]
    merged = merged[ordered]

    out_csv = out_parent / "repeated_summary_merged_with_fixed.csv"
    merged.to_csv(out_csv, index=False)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()

