#!/usr/bin/env python3
"""
Summarize an out_new* run directory into concise per-model reports.

Inputs (expected in out_new_dir root):
  - classifier_metrics_mean_std_pooling.csv
  - diagnostics_pooling_ablation.csv
  - diagnostics_label_shuffle_logistic.csv

Outputs:
  - run_summary.md (overview + per-latent_source best configs)
  - per_source/<latent_source>_summary.md
  - summary_tables/best_by_source.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SourceSummary:
    latent_source: str
    best_test_roc: float
    best_test_pr: float
    best_pooling: str
    best_model: str
    best_val_roc: float | None
    best_val_pr: float | None
    logistic_shuffle_p: float | None


def _best_row(df: pd.DataFrame, group_key: str, metric: str) -> pd.DataFrame:
    # pick max(metric) per group_key; tie-break by test_pr_auc if available
    if "test_pr_auc" in df.columns and metric != "test_pr_auc":
        d = df.sort_values([group_key, metric, "test_pr_auc"], ascending=[True, False, False])
    else:
        d = df.sort_values([group_key, metric], ascending=[True, False])
    return d.groupby(group_key, as_index=False).head(1)


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize out_new run directory")
    ap.add_argument("--out_new_dir", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.out_new_dir)
    m_path = os.path.join(root, "classifier_metrics_mean_std_pooling.csv")
    a_path = os.path.join(root, "diagnostics_pooling_ablation.csv")
    s_path = os.path.join(root, "diagnostics_label_shuffle_logistic.csv")
    for p in (m_path, a_path, s_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    df_m = pd.read_csv(m_path)
    df_a = pd.read_csv(a_path)
    df_s = pd.read_csv(s_path)

    # Best config per source from pooling ablation (test ROC)
    best = _best_row(df_a, "latent_source", "test_roc_auc").copy()
    best = best.rename(columns={"pooling": "best_pooling", "model": "best_model"})
    best = best.rename(columns={"test_roc_auc": "best_test_roc", "test_pr_auc": "best_test_pr"})

    # Join val/test from classifier_metrics for the chosen model (mean_std pooling there)
    # We'll just attach the logistic label-shuffle p-value as a sanity check.
    p_map = {}
    if not df_s.empty:
        for _, r in df_s.iterrows():
            p_map[str(r["latent_source"])] = float(r.get("p_value_one_sided", np.nan))

    rows: list[SourceSummary] = []
    for _, r in best.iterrows():
        src = str(r["latent_source"])
        rows.append(
            SourceSummary(
                latent_source=src,
                best_test_roc=float(r["best_test_roc"]),
                best_test_pr=float(r["best_test_pr"]),
                best_pooling=str(r["best_pooling"]),
                best_model=str(r["best_model"]),
                best_val_roc=None,
                best_val_pr=None,
                logistic_shuffle_p=p_map.get(src, None),
            )
        )

    # write tables
    os.makedirs(os.path.join(root, "summary_tables"), exist_ok=True)
    best_csv = os.path.join(root, "summary_tables", "best_by_source.csv")
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(best_csv, index=False)

    # per-source markdown
    per_dir = os.path.join(root, "per_source")
    os.makedirs(per_dir, exist_ok=True)
    for r in rows:
        md = []
        md.append(f"## {r.latent_source}\n")
        md.append(f"- **Best (by test ROC)**: `{r.best_pooling}` + `{r.best_model}`\n")
        md.append(f"- **Test ROC / PR**: {_fmt(r.best_test_roc)} / {_fmt(r.best_test_pr)}\n")
        md.append(f"- **Logistic label-shuffle p** (diagnostics): {_fmt(r.logistic_shuffle_p)}\n")
        md.append("\n### Pooling ablation (test)\n")
        sub = df_a[df_a["latent_source"].astype(str) == r.latent_source].copy()
        if sub.empty:
            md.append("_No rows found._\n")
        else:
            sub = sub.sort_values(["test_roc_auc", "test_pr_auc"], ascending=[False, False])
            md.append("`pooling, model, test_roc_auc, test_pr_auc`\n\n")
            for _, rr in sub.iterrows():
                md.append(
                    f"- `{rr['pooling']}`, `{rr['model']}`, {float(rr['test_roc_auc']):.3f}, {float(rr['test_pr_auc']):.3f}\n"
                )
        out_path = os.path.join(per_dir, f"{r.latent_source}_summary.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(md))

    # run summary markdown
    run_md = []
    run_md.append(f"## out_new summary\n\n")
    run_md.append(f"- Directory: `{root}`\n")
    run_md.append(f"- Sources: {', '.join(sorted({r.latent_source for r in rows}))}\n\n")
    run_md.append("### Best per source (by test ROC, from pooling ablation)\n\n")
    for r in sorted(rows, key=lambda x: x.best_test_roc, reverse=True):
        run_md.append(
            f"- **{r.latent_source}**: `{r.best_pooling}` + `{r.best_model}` → "
            f"test ROC/PR={_fmt(r.best_test_roc)}/{_fmt(r.best_test_pr)}; shuffle p={_fmt(r.logistic_shuffle_p)}\n"
        )
    run_md.append(f"\nFiles written:\n- `summary_tables/best_by_source.csv`\n- `per_source/*_summary.md`\n")
    with open(os.path.join(root, "run_summary.md"), "w", encoding="utf-8") as f:
        f.write("".join(run_md))

    print(f"Wrote: {best_csv}")
    print(f"Wrote: {os.path.join(root, 'run_summary.md')}")
    print(f"Wrote: {per_dir}/*_summary.md")


if __name__ == "__main__":
    main()

