#!/usr/bin/env python3
"""
Plot **ROC curves** for Plain, MONAI, and fixed-weight soft ensemble on the **same test patients**
for one chosen ``repeat_id`` (or a representative split chosen by ``--repeat_pick``).

Uses the same joined OOS tables as ``plot_ensemble_disagreement_complementarity.py``.

Example
-------
  python3 utils/scripts/plot_ensemble_roc_curves_compare.py \\
    --csv_plain .../plain/.../test_oos_predictions.csv \\
    --csv_monai .../monai/.../test_oos_predictions.csv \\
    --filter_model_plain logistic_en_C0.1_r0.2 --filter_model_monai rbf_svc \\
    --fixed_w 0.5 --repeat_pick median_plain \\
    --out_png Report/Draft/figures/fig_ensemble_roc_compare.png

  # Specific split:
  python3 utils/scripts/plot_ensemble_roc_curves_compare.py \\
    --csv_plain ... --csv_monai ... --repeat_id 0 --out_png ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve, roc_auc_score


def _json_native(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    return x


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


def _join_oos(
    da: pd.DataFrame,
    db: pd.DataFrame,
    *,
    sc: str,
    idc: str,
    cp: str,
    cm: str,
) -> pd.DataFrame:
    cols_a = [sc, idc, cp, "label"]
    a = da[cols_a].copy()
    a[idc] = a[idc].astype(str)
    a[sc] = _coerce_split_col(a[sc])
    a = a.rename(columns={"label": "label_plain", cp: "_pa"})

    b = db[[sc, idc, cm, "label"]].copy()
    b[idc] = b[idc].astype(str)
    b[sc] = _coerce_split_col(b[sc])
    b = b.rename(columns={"label": "label_monai", cm: "_pb"})

    m = a.merge(b, on=[sc, idc], how="inner")
    if not (m["label_plain"] == m["label_monai"]).all():
        raise SystemExit("Label mismatch after join")
    m["y"] = m["label_plain"].astype(int)
    m = m.drop(columns=["label_plain", "label_monai"])
    m["_pa"] = pd.to_numeric(m["_pa"], errors="coerce")
    m["_pb"] = pd.to_numeric(m["_pb"], errors="coerce")
    m = m.dropna(subset=["_pa", "_pb", "y"])
    return m


def _sort_key_split(v: Any) -> tuple:
    if isinstance(v, (int, np.integer)):
        return (0, int(v))
    if isinstance(v, float) and float(v).is_integer():
        return (0, int(v))
    return (1, str(v))


def _pick_repeat_id(
    m: pd.DataFrame,
    sc: str,
    mode: str,
    fw: float,
) -> Any:
    split_ids = sorted(m[sc].unique(), key=_sort_key_split)
    if mode == "first":
        return split_ids[0]

    scores: list[tuple[Any, float]] = []
    for sid in split_ids:
        sub = m[m[sc] == sid]
        y = sub["y"].to_numpy(dtype=np.int64)
        if len(np.unique(y)) < 2:
            continue
        pa = sub["_pa"].to_numpy(dtype=np.float64)
        pb = sub["_pb"].to_numpy(dtype=np.float64)
        pe = fw * pa + (1.0 - fw) * pb
        if mode == "median_plain":
            s = pa
        elif mode == "median_monai":
            s = pb
        elif mode == "median_ensemble":
            s = pe
        else:
            raise SystemExit(f"Unknown --repeat_pick {mode!r}")
        scores.append((sid, float(roc_auc_score(y, s))))
    if not scores:
        raise SystemExit("No split with two classes for AUC-based pick.")
    scores.sort(key=lambda x: x[1])
    return scores[len(scores) // 2][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="ROC curve comparison: Plain vs MONAI vs ensemble (one split).")
    ap.add_argument("--csv_plain", required=True)
    ap.add_argument("--csv_monai", required=True)
    ap.add_argument("--prob_col_plain", default="prob_tts")
    ap.add_argument("--prob_col_monai", default="prob_tts")
    ap.add_argument("--split_col", default="repeat_id")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument("--filter_model_plain", default="")
    ap.add_argument("--filter_model_monai", default="")
    ap.add_argument("--fixed_w", type=float, default=0.5)
    ap.add_argument("--repeat_id", default="", help="Use this split explicitly (integer or string).")
    ap.add_argument(
        "--repeat_pick",
        choices=("first", "median_plain", "median_monai", "median_ensemble"),
        default="median_plain",
        help="Chooses one split when --repeat_id is omitted.",
    )
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="", help="Optional: AUCs and chosen repeat_id.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    sc = str(args.split_col)
    idc = str(args.patient_id_col)
    cp = str(args.prob_col_plain)
    cm = str(args.prob_col_monai)
    fw = float(np.clip(args.fixed_w, 0.0, 1.0))

    da = _apply_model_filter(_read(args.csv_plain), str(args.filter_model_plain), "csv_plain")
    db = _apply_model_filter(_read(args.csv_monai), str(args.filter_model_monai), "csv_monai")
    if "label" not in da.columns or "label" not in db.columns:
        raise SystemExit("Both CSVs need a 'label' column.")

    m = _join_oos(da, db, sc=sc, idc=idc, cp=cp, cm=cm)

    if str(args.repeat_id).strip():
        sid_raw = str(args.repeat_id).strip()
        try:
            chosen = int(sid_raw)
        except ValueError:
            chosen = sid_raw
        uniq = m[sc].unique()
        if not any(np.asarray(chosen == u).item() for u in uniq):
            raise SystemExit(f"repeat_id {chosen!r} not in joined data.")
        sid = chosen
    else:
        sid = _pick_repeat_id(m, sc, str(args.repeat_pick), fw)

    sub = m[m[sc] == sid]
    y = sub["y"].to_numpy(dtype=np.int64)
    pa = sub["_pa"].to_numpy(dtype=np.float64)
    pb = sub["_pb"].to_numpy(dtype=np.float64)
    pe = fw * pa + (1.0 - fw) * pb

    if len(np.unique(y)) < 2:
        raise SystemExit(f"Split {sid!r} has only one class in test; cannot draw ROC.")

    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)

    def _curves() -> None:
        for arr, name, c in (
            (pa, "Plain", "#4e79a7"),
            (pb, "MONAI", "#f28e2b"),
            (pe, rf"Ensemble ($w$={fw:g})", "#59a14f"),
        ):
            fpr, tpr, _ = roc_curve(y, arr)
            a = float(sk_auc(fpr, tpr))
            ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC = {a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.35)

    _curves()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal")

    tit = args.title.strip() or f"ROC (test only), {sc}={sid} — paired OOS"
    ax.set_title(tit, fontsize=11)

    out_png = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)

    meta = {
        str(sc): _json_native(sid),
        "n_test": int(len(sub)),
        "fixed_w": float(fw),
        "roc_auc_plain": float(roc_auc_score(y, pa)),
        "roc_auc_monai": float(roc_auc_score(y, pb)),
        "roc_auc_ensemble": float(roc_auc_score(y, pe)),
        "repeat_pick": str(args.repeat_pick) if not str(args.repeat_id).strip() else None,
    }
    if args.out_json.strip():
        p = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print("Wrote", out_png, file=sys.stderr)
    print(json.dumps(meta, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
