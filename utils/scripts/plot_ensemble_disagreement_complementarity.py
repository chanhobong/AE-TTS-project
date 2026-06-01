#!/usr/bin/env python3
"""
Complementarity of Plain vs MONAI **test** predictions on the same splits.

Joins two ``test_oos_predictions.csv`` (or equivalent) on ``(repeat_id, patient_id)``,
applies optional ``--filter_model_*``, builds binary predictions at ``--threshold``,
then summarises the 2×2 table of **correctness** (relative to labels).

Outputs
-------
1. **PNG** — Discordant-rate boxplots + mean stacked-style bar of the four outcome cells.
2. **CSV** — One row per ``repeat_id`` with counts/fractions + McNemar p when scipy is available.

McNemar tests paired **error** patterns per split. Splits reuse patients — treat p-values as heuristic.

Example
-------
  python3 utils/scripts/plot_ensemble_disagreement_complementarity.py \\
    --csv_plain path/plain_oos.csv --csv_monai path/monai_oos.csv \\
    --filter_model_plain logistic_en_C0.1_r0.2 --filter_model_monai rbf_svc \\
    --fixed_w 0.5 --out_png Report/Draft/figures/fig_ensemble_disagreement.png \\
    --out_csv latent_data/ensemble_disagreement_per_split.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mcnemar
except ImportError:  # pragma: no cover
    mcnemar = None  # type: ignore[misc, assignment]


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
        raise SystemExit(f"{which}: empty after model filter {want!r}")
    return out


def _coerce_split_col(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(np.int64)
    return s.astype(str)


def _boxplot_compat(ax, arrays, names, **kw):
    try:
        return ax.boxplot(arrays, tick_labels=names, **kw)
    except TypeError:
        return ax.boxplot(arrays, labels=names, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Disagreement / complementarity plots for paired OOS predictions.")
    ap.add_argument("--csv_plain", required=True)
    ap.add_argument("--csv_monai", required=True)
    ap.add_argument("--prob_col_plain", default="prob_tts")
    ap.add_argument("--prob_col_monai", default="prob_tts")
    ap.add_argument("--split_col", default="repeat_id")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument("--filter_model_plain", default="")
    ap.add_argument("--filter_model_monai", default="")
    ap.add_argument("--fixed_w", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_stats_json", default="")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    sc = str(args.split_col)
    idc = str(args.patient_id_col)
    cp = str(args.prob_col_plain)
    cm = str(args.prob_col_monai)
    fw = float(np.clip(args.fixed_w, 0.0, 1.0))
    thr = float(args.threshold)

    da = _apply_model_filter(_read(args.csv_plain), str(args.filter_model_plain), "csv_plain")
    db = _apply_model_filter(_read(args.csv_monai), str(args.filter_model_monai), "csv_monai")

    for name, d in (("csv_plain", da), ("csv_monai", db)):
        if sc not in d.columns or idc not in d.columns:
            raise SystemExit(f"{name} needs {sc!r} and {idc!r}; columns={list(d.columns)}")
    if cp not in da.columns or cm not in db.columns:
        raise SystemExit("Missing prob columns")

    cols_a = [sc, idc, cp, "label"]
    if "label" not in da.columns:
        raise SystemExit("csv_plain must contain 'label'.")
    a = da[cols_a].copy()
    a[idc] = a[idc].astype(str)
    a[sc] = _coerce_split_col(a[sc])
    a = a.rename(columns={"label": "label_plain", cp: "_pa"})
    a["label_plain"] = pd.to_numeric(a["label_plain"], errors="coerce").astype("Int64")

    if "label" not in db.columns:
        raise SystemExit("csv_monai must contain 'label'.")
    b = db[[sc, idc, cm, "label"]].copy()
    b[idc] = b[idc].astype(str)
    b[sc] = _coerce_split_col(b[sc])
    b = b.rename(columns={"label": "label_monai", cm: "_pb"})
    b["label_monai"] = pd.to_numeric(b["label_monai"], errors="coerce").astype("Int64")

    m = a.merge(b, on=[sc, idc], how="inner")
    if not (m["label_plain"] == m["label_monai"]).all():
        raise SystemExit("Label mismatch after join")
    m["y"] = m["label_plain"].astype(int)
    m = m.drop(columns=["label_plain", "label_monai"])

    m["_pa"] = pd.to_numeric(m["_pa"], errors="coerce")
    m["_pb"] = pd.to_numeric(m["_pb"], errors="coerce")
    m = m.dropna(subset=["_pa", "_pb", "y"])

    rows: list[dict[str, Any]] = []
    split_ids = sorted(m[sc].unique(), key=lambda v: (str(type(v)), str(v)))

    disc_plain_right_monai_wrong: list[float] = []
    disc_plain_wrong_monai_right: list[float] = []
    fr_both_ok: list[float] = []
    fr_p_only: list[float] = []
    fr_m_only: list[float] = []
    fr_both_bad: list[float] = []
    mcnemar_p: list[float] = []
    fr_ens_correct: list[float] = []

    for sid in split_ids:
        sub = m[m[sc] == sid]
        y = sub["y"].to_numpy(dtype=np.int64)
        pa = sub["_pa"].to_numpy(dtype=np.float64)
        pb = sub["_pb"].to_numpy(dtype=np.float64)
        pe = fw * pa + (1.0 - fw) * pb
        pp = (pa >= thr).astype(np.int64)
        pm = (pb >= thr).astype(np.int64)
        peb = (pe >= thr).astype(np.int64)

        cp_ = pp == y
        cm_ = pm == y

        both_ok = cp_ & cm_
        p_only = cp_ & ~cm_
        m_only = ~cp_ & cm_
        both_bad = ~cp_ & ~cm_

        n = int(len(sub))
        n_both_ok = int(np.sum(both_ok))
        n_p_only = int(np.sum(p_only))
        n_m_only = int(np.sum(m_only))
        n_both_bad = int(np.sum(both_bad))

        disc_plain_right_monai_wrong.append(float(n_p_only) / n if n else float("nan"))
        disc_plain_wrong_monai_right.append(float(n_m_only) / n if n else float("nan"))
        fr_both_ok.append(float(n_both_ok) / n if n else float("nan"))
        fr_p_only.append(float(n_p_only) / n if n else float("nan"))
        fr_m_only.append(float(n_m_only) / n if n else float("nan"))
        fr_both_bad.append(float(n_both_bad) / n if n else float("nan"))
        fr_ens_correct.append(float(np.mean(peb == y)) if n else float("nan"))

        row_stat: dict[str, Any] = {
            sc: sid,
            "n_test": n,
            "frac_both_correct": fr_both_ok[-1],
            "frac_plain_only_correct": fr_p_only[-1],
            "frac_monai_only_correct": fr_m_only[-1],
            "frac_both_wrong": fr_both_bad[-1],
            "frac_ensemble_correct": fr_ens_correct[-1],
            "mcnemar_p": float("nan"),
        }

        tab = np.array([[n_both_ok, n_p_only], [n_m_only, n_both_bad]], dtype=np.float64)
        if mcnemar is not None and n_p_only + n_m_only > 0:
            try:
                res = mcnemar(tab, exact=False, correction=True)
                row_stat["mcnemar_p"] = float(res.pvalue)
                mcnemar_p.append(float(res.pvalue))
            except Exception:
                pass
        rows.append(row_stat)

    df_rows = pd.DataFrame(rows)
    if args.out_csv.strip():
        pth = Path(args.out_csv).resolve()
        pth.parent.mkdir(parents=True, exist_ok=True)
        df_rows.to_csv(pth, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)

    ax0 = axes[0]
    data_bp = [
        np.asarray(disc_plain_right_monai_wrong, dtype=np.float64),
        np.asarray(disc_plain_wrong_monai_right, dtype=np.float64),
    ]
    bp = _boxplot_compat(ax0, data_bp, ["Plain correct,\nMONAI wrong", "Plain wrong,\nMONAI correct"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#4e79a7", "#f28e2b"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)
    ax0.set_ylabel("Fraction of test patients (per split)")
    ax0.set_title("Discordant correctness")
    ax0.grid(True, axis="y", alpha=0.35)

    ax1 = axes[1]
    means = [
        float(np.nanmean(fr_both_ok)),
        float(np.nanmean(fr_p_only)),
        float(np.nanmean(fr_m_only)),
        float(np.nanmean(fr_both_bad)),
    ]
    labels = ["Both correct", "Plain only", "MONAI only", "Both wrong"]
    cols_bar = ["#59a14f", "#4e79a7", "#f28e2b", "#bab0ab"]
    ax1.bar(np.arange(len(means)), means, color=cols_bar, edgecolor="0.25", linewidth=0.6)
    ax1.set_xticks(np.arange(len(means)))
    ax1.set_xticklabels(labels, rotation=14, ha="right")
    ax1.set_ylabel("Mean fraction over splits")
    ax1.set_title("Paired error pattern (Plain vs MONAI)")
    ymax = max(means) + 0.12 if means else 1.0
    ax1.set_ylim(0.0, min(1.05, ymax))
    ax1.grid(True, axis="y", alpha=0.35)

    tit = args.title.strip() or rf"Same-split OOS (paired); ensemble $w$={fw:g}, threshold={thr:g}"
    fig.suptitle(tit, fontsize=11)

    out_png = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_splits": int(len(split_ids)),
        "mean_frac_both_correct": float(np.nanmean(fr_both_ok)),
        "mean_frac_plain_only_correct": float(np.nanmean(fr_p_only)),
        "mean_frac_monai_only_correct": float(np.nanmean(fr_m_only)),
        "mean_frac_both_wrong": float(np.nanmean(fr_both_bad)),
        "mean_frac_ensemble_correct": float(np.nanmean(fr_ens_correct)),
        "mcnemar_p_median_over_splits": float(np.nanmedian(mcnemar_p)) if mcnemar_p else None,
        "note": "McNemar is per-split; repeated splits are not independent experiments.",
    }
    if args.out_stats_json.strip():
        jp = Path(args.out_stats_json).resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Wrote", out_png, file=sys.stderr)
    if args.out_csv.strip():
        print("Wrote", args.out_csv, file=sys.stderr)


if __name__ == "__main__":
    main()
