#!/usr/bin/env python3
"""
Box/violin plots of **per-split** ROC-AUC and PR-AUC from ensemble_oos_same_split_eval.py output.

Reads ``per_split_metrics.csv`` (paired across the same repeat_id for Plain, MONAI, fixed ensemble).

Optional: include oracle (exploratory) as a fourth series — off by default for main-text figures.

Example
-------
  python3 utils/scripts/plot_ensemble_per_split_auc_distributions.py \\
    --per_split_csv latent_data/ensemble_out/run_tag/per_split_metrics.csv \\
    --fixed_w 0.5 \\
    --out_png Report/Draft/figures/fig_ensemble_auc_distributions.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensemble_roc_col(df: pd.DataFrame, fixed_w: float) -> str:
    """Match roc_ensemble_w{fixed_w} allowing float formatting in column names."""
    candidates = [c for c in df.columns if c.startswith("roc_ensemble_w") and "oracle" not in c.lower()]
    if not candidates:
        raise SystemExit(f"No roc_ensemble_w* column in CSV; got {list(df.columns)}")
    want = f"roc_ensemble_w{fixed_w:g}"
    if want in df.columns:
        return want
    for c in candidates:
        m = re.match(r"roc_ensemble_w(.+)$", c)
        if m:
            try:
                if np.isclose(float(m.group(1)), fixed_w, rtol=0, atol=1e-6):
                    return c
            except ValueError:
                continue
    raise SystemExit(f"Could not find ensemble ROC column for fixed_w={fixed_w!r}; have {candidates}")


def _ensemble_pr_col(df: pd.DataFrame, fixed_w: float) -> str:
    roc_c = _ensemble_roc_col(df, fixed_w)
    pr_c = roc_c.replace("roc_", "pr_")
    if pr_c not in df.columns:
        raise SystemExit(f"Missing {pr_c!r} (paired with {roc_c!r}); columns={list(df.columns)}")
    return pr_c


def _boxplot_compat(ax, arrays, names, **kw):
    try:
        return ax.boxplot(arrays, tick_labels=names, **kw)
    except TypeError:
        return ax.boxplot(arrays, labels=names, **kw)


def _ensure_per_split_metrics(df: pd.DataFrame, path: str) -> None:
    need = ("roc_plain", "pr_plain", "roc_monai", "pr_monai")
    if all(c in df.columns for c in need):
        return
    p = Path(path).resolve()
    hint = p.parent / "per_split_metrics.csv"
    if "method" in df.columns and "roc_mean" in df.columns and "roc_std" in df.columns:
        raise SystemExit(
            "You passed summary_methods.csv (one row per method).\n"
            "This script needs per_split_metrics.csv (one row per repeat_id / split).\n"
            f"Use e.g.:\n  {hint}\n"
            "(same folder as ensemble_oos_same_split_eval output)."
        )
    missing = [c for c in need if c not in df.columns]
    raise SystemExit(f"Missing columns {missing!r}; got {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Violin/box per-split AUC distributions (Plain / MONAI / ensemble).")
    ap.add_argument(
        "--per_split_csv",
        required=True,
        help="ensemble_oos_same_split_eval.py output: per_split_metrics.csv (NOT summary_methods.csv).",
    )
    ap.add_argument("--fixed_w", type=float, default=0.5)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--kind",
        choices=("box", "violin"),
        default="box",
        help="box (default) or violin.",
    )
    ap.add_argument("--include_oracle", action="store_true", help="Add exploratory oracle AUC series.")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    df = _read(args.per_split_csv)
    _ensure_per_split_metrics(df, args.per_split_csv)

    roc_e = _ensemble_roc_col(df, float(args.fixed_w))
    pr_e = _ensemble_pr_col(df, float(args.fixed_w))

    series_roc: list[tuple[str, np.ndarray]] = [
        ("Plain", df["roc_plain"].to_numpy(dtype=np.float64)),
        ("MONAI", df["roc_monai"].to_numpy(dtype=np.float64)),
        (rf"Ensemble $w$={args.fixed_w:g}", df[roc_e].to_numpy(dtype=np.float64)),
    ]
    series_pr: list[tuple[str, np.ndarray]] = [
        ("Plain", df["pr_plain"].to_numpy(dtype=np.float64)),
        ("MONAI", df["pr_monai"].to_numpy(dtype=np.float64)),
        (rf"Ensemble $w$={args.fixed_w:g}", df[pr_e].to_numpy(dtype=np.float64)),
    ]

    if args.include_oracle:
        if "roc_ensemble_oracle" in df.columns and "pr_ensemble_oracle" in df.columns:
            series_roc.append(("Oracle $w$/split (exploratory)", df["roc_ensemble_oracle"].to_numpy(dtype=np.float64)))
            series_pr.append(("Oracle $w$/split (exploratory)", df["pr_ensemble_oracle"].to_numpy(dtype=np.float64)))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)

    def _plot_panel(ax: plt.Axes, data: list[tuple[str, np.ndarray]], ylabel: str) -> None:
        names = [t[0] for t in data]
        arrays = [np.asarray(v[np.isfinite(v)], dtype=np.float64) for _, v in data]
        cols = ["#4e79a7", "#f28e2b", "#59a14f", "#bab0ab"]
        if args.kind == "violin":
            parts = ax.violinplot(
                arrays,
                positions=range(len(data)),
                showmeans=True,
                showmedians=True,
                widths=0.72,
            )
            for i, b in enumerate(parts["bodies"]):
                b.set_facecolor(cols[i % len(cols)])
                b.set_alpha(0.82)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=14, ha="right")
        else:
            bp = _boxplot_compat(ax, arrays, names, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(cols[i % len(cols)])
                patch.set_alpha(0.78)
            ax.tick_params(axis="x", rotation=14)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.35)

    _plot_panel(axes[0], series_roc, "ROC-AUC (per split)")
    _plot_panel(axes[1], series_pr, "PR-AUC (per split)")
    tit = args.title.strip() or "Same-split evaluation: distribution over repeated train/test partitions"
    fig.suptitle(tit, fontsize=11)

    out = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
