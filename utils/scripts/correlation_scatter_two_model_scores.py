#!/usr/bin/env python3
"""
Pearson r (and optional Spearman) between two model prediction scores + scatter plot.

Expects two tables with ``patient_id`` and one numeric score column each, inner-joined.
Use the **same evaluation cohort** (e.g. same repeat's test patients with both models' scores);
do not mix scores from incompatible splits.

Example:
  python3 utils/scripts/correlation_scatter_two_model_scores.py \\
    --csv_a plain_ae_test_scores.csv --score_col_a prob_tts \\
    --csv_b monai_rbf_test_scores.csv --score_col_b prob_tts \\
    --label_csv data/test.csv \\
    --out_png Report/Draft/figures/scatter_plain_vs_monai_scores.png \\
    --out_stats latent_data/score_correlation_plain_monai.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None  # type: ignore[misc, assignment]
    spearmanr = None  # type: ignore[misc, assignment]


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _merge_scores(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    id_col: str,
    col_a: str,
    col_b: str,
) -> pd.DataFrame:
    if id_col not in a.columns or id_col not in b.columns:
        raise ValueError(f"Need {id_col} in both CSVs")
    if col_a not in a.columns:
        raise ValueError(f"{col_a!r} not in CSV A columns: {list(a.columns)}")
    if col_b not in b.columns:
        raise ValueError(f"{col_b!r} not in CSV B columns: {list(b.columns)}")
    da = a[[id_col, col_a]].rename(columns={col_a: "score_a"})
    db = b[[id_col, col_b]].rename(columns={col_b: "score_b"})
    da[id_col] = da[id_col].astype(str)
    db[id_col] = db[id_col].astype(str)
    return da.merge(db, on=id_col, how="inner")


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlation + scatter for two model score columns.")
    ap.add_argument("--csv_a", required=True, help="CSV with patient_id + model A score")
    ap.add_argument("--csv_b", required=True, help="CSV with patient_id + model B score")
    ap.add_argument("--score_col_a", required=True)
    ap.add_argument("--score_col_b", required=True)
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument(
        "--label_csv",
        default="",
        help="Optional: CSV with patient_id and label (or case) for coloring (0=Normal, 1=TTS).",
    )
    ap.add_argument("--label_col", default="label", help="Or 'case' if 0/1")
    ap.add_argument("--out_png", default="", help="If set, write scatter plot PNG.")
    ap.add_argument("--out_stats", default="", help="If set, write JSON with r, n, etc.")
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Model A score")
    ap.add_argument("--ylabel", default="Model B score")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    da = _read(args.csv_a)
    db = _read(args.csv_b)
    m = _merge_scores(
        da,
        db,
        id_col=str(args.patient_id_col),
        col_a=str(args.score_col_a),
        col_b=str(args.score_col_b),
    )
    m["score_a"] = pd.to_numeric(m["score_a"], errors="coerce")
    m["score_b"] = pd.to_numeric(m["score_b"], errors="coerce")
    m = m.dropna(subset=["score_a", "score_b"])
    if len(m) < 3:
        raise SystemExit(f"Too few matched rows with finite scores: {len(m)}")

    x = m["score_a"].to_numpy(dtype=np.float64)
    y = m["score_b"].to_numpy(dtype=np.float64)

    stats: dict = {"n_patients": int(len(m))}
    if pearsonr is not None:
        pr, pp = pearsonr(x, y)
        stats["pearson_r"] = float(pr)
        stats["pearson_p_two_sided"] = float(pp)
    else:
        c = np.corrcoef(x, y)[0, 1]
        stats["pearson_r"] = float(c)
        stats["pearson_p_two_sided"] = None
        stats["note"] = "install scipy for p-value"

    if spearmanr is not None:
        sr, sp = spearmanr(x, y)
        stats["spearman_rho"] = float(sr)
        stats["spearman_p_two_sided"] = float(sp)

    print(json.dumps(stats, indent=2))

    if args.out_stats:
        p = Path(args.out_stats).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("Wrote", p, file=sys.stderr)

    if args.out_png:
        if args.label_csv.strip():
            lab_df = _read(args.label_csv)
            idc = str(args.patient_id_col)
            lc = str(args.label_col)
            if lc not in lab_df.columns and "case" in lab_df.columns:
                lc = "case"
            if idc not in lab_df.columns or lc not in lab_df.columns:
                raise SystemExit(f"label_csv needs {idc} and label/case; got {list(lab_df.columns)}")
            sub = lab_df[[idc, lc]].copy()
            sub[idc] = sub[idc].astype(str)
            sub["_lab"] = pd.to_numeric(sub[lc], errors="coerce").astype("float64")
            sub = sub[[idc, "_lab"]]
            m = m.merge(sub, left_on=args.patient_id_col, right_on=idc, how="left")
            if idc != args.patient_id_col and idc in m.columns:
                m = m.drop(columns=[idc], errors="ignore")

        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        if args.label_csv.strip() and "_lab" in m.columns:
            pal = {0: ("Normal", "#4C72B0"), 1: ("TTS", "#DD8452")}
            for lab in (0, 1):
                sl = m[m["_lab"] == float(lab)]
                if sl.empty:
                    continue
                name, col = pal[lab]
                ax.scatter(
                    sl["score_a"],
                    sl["score_b"],
                    s=36,
                    alpha=0.78,
                    c=col,
                    label=name,
                    edgecolors="white",
                    linewidths=0.35,
                )
            ax.legend(loc="best", framealpha=0.9)
        else:
            ax.scatter(x, y, s=36, alpha=0.72, c="#3d5a80", edgecolors="white", linewidths=0.35)
        ax.set_xlabel(str(args.xlabel))
        ax.set_ylabel(str(args.ylabel))
        r_txt = f"Pearson r = {stats['pearson_r']:.3f}\nn = {stats['n_patients']}"
        ax.text(
            0.04,
            0.96,
            r_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
        t = str(args.title).strip()
        if t:
            ax.set_title(t)
        ax.grid(True, alpha=0.28, linestyle=":")
        fig.tight_layout()
        outp = Path(args.out_png).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=int(args.dpi))
        plt.close(fig)
        print("Wrote", outp, file=sys.stderr)


if __name__ == "__main__":
    main()
