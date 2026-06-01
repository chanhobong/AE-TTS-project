#!/usr/bin/env python3
"""
Paired **per-split** inference: does fixed ensemble beat Plain or MONAI across the same splits?

From ``per_split_metrics.csv`` (``ensemble_oos_same_split_eval.py``):

* ``delta_roc = roc_ensemble - roc_ref`` per split (ref = plain or monai)
* ``delta_pr`` likewise
* Mean delta, **95% bootstrap CI** (resample splits), **Wilcoxon signed-rank** p-value

Optional: two ``test_oos_predictions.csv`` paths add

* Pearson **r** (mean ± bootstrap across splits; pooled r diagnostic)
* Four-way correctness fractions with bootstrap CI on split means

Example
-------
  python3 utils/scripts/analyze_ensemble_paired_splits.py \\
    --per_split_csv ensemble_out/run_tag/per_split_metrics.csv \\
    --fixed_w 0.5 \\
    --out_json latent_data/ensemble_paired_inference.json
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

try:
    from scipy.stats import pearsonr, wilcoxon
except ImportError:  # pragma: no cover
    pearsonr = None  # type: ignore[misc, assignment]
    wilcoxon = None  # type: ignore[misc, assignment]


def _read(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_per_split(df: pd.DataFrame) -> None:
    need = ("roc_plain", "pr_plain", "roc_monai", "pr_monai")
    if not all(c in df.columns for c in need):
        raise SystemExit(f"Expected columns including {need}; got {list(df.columns)}")


def _ensemble_roc_col(df: pd.DataFrame, fixed_w: float) -> str:
    candidates = [c for c in df.columns if c.startswith("roc_ensemble_w") and "oracle" not in c.lower()]
    if not candidates:
        raise SystemExit(f"No roc_ensemble_w* column; have {list(df.columns)}")
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
    raise SystemExit(f"No ensemble ROC column for fixed_w={fixed_w!r}; candidates={candidates}")


def _ensemble_pr_col(df: pd.DataFrame, fixed_w: float) -> str:
    r = _ensemble_roc_col(df, fixed_w)
    p = r.replace("roc_", "pr_")
    if p not in df.columns:
        raise SystemExit(f"Missing {p!r}")
    return p


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(x))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(x[idx]))
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1.0 - alpha / 2))
    return mu, lo, hi


def _wilcoxon_report(d: np.ndarray, alternative: str) -> dict[str, Any]:
    out: dict[str, Any] = {"wilcoxon_statistic": None, "wilcoxon_p": None, "n_nonzero_pairs": int(np.sum(d != 0))}
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        out["note"] = "Too few finite differences for Wilcoxon"
        return out
    if wilcoxon is None:
        out["note"] = "scipy not installed"
        return out
    try:
        res = wilcoxon(d, zero_method="wilcox", alternative=alternative, mode="auto")
        out["wilcoxon_statistic"] = float(res.statistic)
        out["wilcoxon_p"] = float(res.pvalue)
    except ValueError as e:
        out["wilcoxon_error"] = str(e)
    return out


def _json_native(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _json_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_native(v) for v in x]
    return x


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


def _coerce_split_col(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(np.int64)
    return s.astype(str)


def _join_oos(da: pd.DataFrame, db: pd.DataFrame, sc: str, idc: str, cp: str, cm: str) -> pd.DataFrame:
    a = da[[sc, idc, cp, "label"]].copy()
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
    return m.dropna(subset=["_pa", "_pb", "y"])


def _sort_key_split(v: Any) -> tuple:
    if isinstance(v, (int, np.integer)):
        return (0, int(v))
    if isinstance(v, float) and float(v).is_integer():
        return (0, int(v))
    return (1, str(v))


def _analyze_oos_block(
    m: pd.DataFrame,
    sc: str,
    thr: float,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    split_ids = sorted(m[sc].unique(), key=_sort_key_split)
    rs: list[float] = []
    fr_bc: list[float] = []
    fr_po: list[float] = []
    fr_mo: list[float] = []
    fr_bw: list[float] = []
    p_all_a: list[float] = []
    p_all_b: list[float] = []

    for sid in split_ids:
        sub = m[m[sc] == sid]
        pa = sub["_pa"].to_numpy(dtype=np.float64)
        pb = sub["_pb"].to_numpy(dtype=np.float64)
        y = sub["y"].to_numpy(dtype=np.int64)
        if pearsonr is not None and len(pa) > 2:
            r, _ = pearsonr(pa, pb)
            if np.isfinite(r):
                rs.append(float(r))
        p_all_a.extend(float(x) for x in pa)
        p_all_b.extend(float(x) for x in pb)

        pp = (pa >= thr).astype(np.int64)
        pm = (pb >= thr).astype(np.int64)
        cp = pp == y
        cm = pm == y
        n = len(sub)
        if n == 0:
            continue
        fr_bc.append(float(np.mean(cp & cm)))
        fr_po.append(float(np.mean(cp & ~cm)))
        fr_mo.append(float(np.mean(~cp & cm)))
        fr_bw.append(float(np.mean(~cp & ~cm)))

    pooled_r: Any = None
    if pearsonr is not None and len(p_all_a) > 2:
        pooled_r = float(pearsonr(np.asarray(p_all_a), np.asarray(p_all_b))[0])

    def _ci(xs: list[float]) -> tuple[float, float, float]:
        return _bootstrap_mean_ci(np.asarray(xs, dtype=np.float64), n_boot, alpha, seed)

    mu_r, lo_r, hi_r = _ci(rs) if rs else (float("nan"), float("nan"), float("nan"))

    def _pack(xs: list[float]) -> dict[str, Any]:
        mu, lo, hi = _ci(xs)
        return {"mean_over_splits": mu, "ci95_low": lo, "ci95_high": hi}

    return {
        "pearson_r_between_scores": {
            "mean_per_split": mu_r,
            "std_per_split": float(np.std(rs, ddof=0)) if rs else float("nan"),
            "ci95_mean_r": {"low": lo_r, "high": hi_r},
            "n_splits_with_r": len(rs),
            "pooled_all_rows": pooled_r,
            "note_pooled": "Pooled r inflates effective n (same patients in many splits). Prefer mean per-split r.",
        },
        "accuracy_2x2_cell_fractions": {
            "both_correct": _pack(fr_bc),
            "plain_only_correct": _pack(fr_po),
            "monai_only_correct": _pack(fr_mo),
            "both_wrong": _pack(fr_bw),
        },
        "threshold_for_binary_accuracy": thr,
    }


def _block_vs_ref(
    roc_e: np.ndarray,
    pr_e: np.ndarray,
    roc_r: np.ndarray,
    pr_r: np.ndarray,
    *,
    ref_name: str,
    n_boot: int,
    alpha: float,
    seed: int,
    wilcoxon_alt: str,
) -> dict[str, Any]:
    d_roc = roc_e - roc_r
    d_pr = pr_e - pr_r
    roc_mu, roc_lo, roc_hi = _bootstrap_mean_ci(d_roc, n_boot, alpha, seed)
    pr_mu, pr_lo, pr_hi = _bootstrap_mean_ci(d_pr, n_boot, alpha, seed)
    wr = _wilcoxon_report(d_roc, wilcoxon_alt)
    wp = _wilcoxon_report(d_pr, wilcoxon_alt)
    return {
        "reference": ref_name,
        "delta_roc": {
            "mean": roc_mu,
            "ci95_low": roc_lo,
            "ci95_high": roc_hi,
            "wilcoxon_statistic": wr.get("wilcoxon_statistic"),
            "wilcoxon_p": wr.get("wilcoxon_p"),
            "n_nonzero_pairs": wr.get("n_nonzero_pairs"),
            **({k: wr[k] for k in ("note", "wilcoxon_error") if k in wr}),
        },
        "delta_pr": {
            "mean": pr_mu,
            "ci95_low": pr_lo,
            "ci95_high": pr_hi,
            "wilcoxon_statistic": wp.get("wilcoxon_statistic"),
            "wilcoxon_p": wp.get("wilcoxon_p"),
            "n_nonzero_pairs": wp.get("n_nonzero_pairs"),
            **({k: wp[k] for k in ("note", "wilcoxon_error") if k in wp}),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired per-split ensemble vs plain/monai + optional OOS numbers.")
    ap.add_argument("--per_split_csv", required=True)
    ap.add_argument("--fixed_w", type=float, default=0.5)
    ap.add_argument("--n_bootstrap", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument(
        "--wilcoxon_alternative",
        choices=("greater", "less", "two-sided"),
        default="greater",
        help="Paired ensemble−ref: 'greater' => ensemble AUC tends larger.",
    )
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--csv_plain", default="")
    ap.add_argument("--csv_monai", default="")
    ap.add_argument("--prob_col_plain", default="prob_tts")
    ap.add_argument("--prob_col_monai", default="prob_tts")
    ap.add_argument("--split_col", default="repeat_id")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument("--filter_model_plain", default="")
    ap.add_argument("--filter_model_monai", default="")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = _read(args.per_split_csv)
    _ensure_per_split(df)

    fw = float(np.clip(args.fixed_w, 0.0, 1.0))
    roc_e_col = _ensemble_roc_col(df, fw)
    pr_e_col = _ensemble_pr_col(df, fw)

    roc_e = df[roc_e_col].to_numpy(dtype=np.float64)
    pr_e = df[pr_e_col].to_numpy(dtype=np.float64)
    roc_p = df["roc_plain"].to_numpy(dtype=np.float64)
    pr_p = df["pr_plain"].to_numpy(dtype=np.float64)
    roc_m = df["roc_monai"].to_numpy(dtype=np.float64)
    pr_m = df["pr_monai"].to_numpy(dtype=np.float64)

    n_boot = max(100, int(args.n_bootstrap))
    seed = int(args.bootstrap_seed)
    alpha = float(args.alpha)
    alt = str(args.wilcoxon_alternative)

    summary: dict[str, Any] = {
        "n_splits": int(len(df)),
        "fixed_w": fw,
        "ensemble_roc_col": roc_e_col,
        "ensemble_pr_col": pr_e_col,
        "vs_plain": _block_vs_ref(
            roc_e, pr_e, roc_p, pr_p, ref_name="plain", n_boot=n_boot, alpha=alpha, seed=seed, wilcoxon_alt=alt
        ),
        "vs_monai": _block_vs_ref(
            roc_e, pr_e, roc_m, pr_m, ref_name="monai", n_boot=n_boot, alpha=alpha, seed=seed, wilcoxon_alt=alt
        ),
        "notes": {
            "wilcoxon": "Signed-rank on paired per-split differences (ensemble − ref).",
            "bootstrap": "Resamples splits with replacement; CI targets mean(delta) over splits.",
        },
    }

    if str(args.csv_plain).strip() and str(args.csv_monai).strip():
        da = _apply_model_filter(_read(args.csv_plain), str(args.filter_model_plain), "csv_plain")
        db = _apply_model_filter(_read(args.csv_monai), str(args.filter_model_monai), "csv_monai")
        sc = str(args.split_col)
        idc = str(args.patient_id_col)
        mm = _join_oos(da, db, sc=sc, idc=idc, cp=str(args.prob_col_plain), cm=str(args.prob_col_monai))
        summary["oos_scores"] = _analyze_oos_block(
            mm,
            sc=sc,
            thr=float(args.threshold),
            n_boot=n_boot,
            alpha=alpha,
            seed=seed + 1,
        )

    summary = _json_native(summary)

    if args.out_json.strip():
        p = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.out_csv.strip():
        rows = []
        for ref_key in ("vs_plain", "vs_monai"):
            block = summary[ref_key]
            ref = block["reference"]
            for metric in ("delta_roc", "delta_pr"):
                d = block[metric]
                rows.append(
                    {
                        "reference": ref,
                        "metric": metric,
                        "mean_delta": d["mean"],
                        "ci95_low": d["ci95_low"],
                        "ci95_high": d["ci95_high"],
                        "wilcoxon_p": d.get("wilcoxon_p"),
                    }
                )
        pd.DataFrame(rows).to_csv(os.path.abspath(args.out_csv), index=False)

    print(json.dumps(summary, indent=2))
    if args.out_json.strip():
        print("Wrote", args.out_json, file=sys.stderr)


if __name__ == "__main__":
    main()
