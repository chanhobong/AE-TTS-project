#!/usr/bin/env python3
"""
Per-patient slice statistics on selected latent dims: mean, std, and detrended volatility.

Volatility (detrended wobble index), on valid rows in NPZ order:
    Volatility = (1 / (N-1)) * sum_t | z[t,j] - z[t-1,j] |

Group summaries (y_true 0 vs 1) compare average of those per-patient scalars — e.g. whether
mean-of-means differs less than mean-of-stds. Includes Mann-Whitney U / t-test p-values and
Pearson / Spearman / point-biserial vs y_true for std and volatility.

Example
-------
  python analyze_latent_mean_std_volatility.py \\
    --npz_dir .../patients \\
    --labels_csv /path/Combined_Labels.csv \\
    --dims 63 \\
    --out_dir ./latent_mean_std_vol

`--labels_csv` matches other utils: patient id column (patient_id / ID), label (case / label / …).

NPZ stem usually matches `patient_id`. Normal cohort NPZ stems often end with `_<age><F|M>` (e.g. `LBH_22392393_71F`)
while the table lists the base ID only. Then: try stripping that suffix and look up again; if still missing and
`--infer_normal_from_stem` matches `--normal_stem_regex`, label is set to 0 (normal), same convention as
`plot_latent_slice_profile.py --normal_stem_regex`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, pearsonr, pointbiserialr, spearmanr, ttest_ind
except ImportError:
    mannwhitneyu = pearsonr = pointbiserialr = spearmanr = ttest_ind = None  # type: ignore


def _merge_labels(paths: list[str]) -> dict[str, int]:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        id_col = "patient_id" if "patient_id" in df.columns else None
        if id_col is None:
            for c in ("ID", "id"):
                if c in df.columns:
                    id_col = c
                    break
        if id_col is None:
            raise ValueError(f"No patient id column in {p}")
        lab_col = None
        for c in ("label", "case", "Label", "TTS"):
            if c in df.columns:
                lab_col = c
                break
        if lab_col is None:
            raise ValueError(f"No label column in {p}")
        sub = df[[id_col, lab_col]].copy()
        sub.columns = ["patient_id", "raw_label"]
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["patient_id"] = out["patient_id"].astype(str)

    def to_bin(v) -> int:
        if isinstance(v, str):
            s = v.lower()
            if "tts" in s or "takotsubo" in s:
                return 1
            if v.strip().isdigit():
                return int(v.strip())
            return 0
        return int(v)

    out["y_true"] = out["raw_label"].map(to_bin)
    out = out.drop_duplicates(subset=["patient_id"], keep="last")
    return dict(zip(out["patient_id"], out["y_true"]))


def _strip_age_sex_suffix(stem: str) -> str:
    """Remove trailing _12F / _71M style token (one underscore + digits + F or M)."""
    return re.sub(r"_\d+[FM]$", "", stem, flags=re.IGNORECASE)


def _resolve_y_for_npz_stem(
    stem: str,
    labels: dict[str, int],
    normal_pat: re.Pattern[str],
    infer_normal_from_stem: bool,
) -> tuple[Optional[int], str]:
    """
    Returns (y_true, source). source documents how the label was chosen (for CSV column label_source).
    """
    if stem in labels:
        return int(labels[stem]), "labels_csv_exact"
    base = _strip_age_sex_suffix(stem)
    if base != stem and base in labels:
        return int(labels[base]), "labels_csv_stripped_suffix"
    if infer_normal_from_stem and normal_pat.match(stem):
        return 0, "inferred_normal_stem_regex"
    return None, "unmatched"


def _slice_mask(z) -> Optional[np.ndarray]:
    if "mask" not in z.files:
        return None
    return np.asarray(z["mask"]).reshape(-1)


def _patient_scalar_features(
    npz_path: str,
    dims: list[int],
    respect_mask: bool,
    std_ddof: int,
) -> Optional[dict]:
    z = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in z.files:
        return None
    emb = np.asarray(z["embeddings"], dtype=np.float64)
    if emb.ndim != 2:
        return None
    d_tot = emb.shape[1]
    if any(j < 0 or j >= d_tot for j in dims):
        return None
    n = emb.shape[0]
    valid = np.ones(n, dtype=bool)
    m = _slice_mask(z)
    if respect_mask and m is not None and m.shape[0] == n:
        valid = m.astype(bool)
    row: dict = {}
    for j in dims:
        y = emb[valid, j]
        k = int(y.size)
        if k == 0:
            row[f"dim_{j}_slice_mean"] = np.nan
            row[f"dim_{j}_slice_std"] = np.nan
            row[f"dim_{j}_volatility"] = np.nan
            row[f"dim_{j}_n_valid_slices"] = 0
            continue
        row[f"dim_{j}_slice_mean"] = float(np.mean(y))
        row[f"dim_{j}_slice_std"] = float(np.std(y, ddof=std_ddof))
        row[f"dim_{j}_n_valid_slices"] = k
        if k < 2:
            row[f"dim_{j}_volatility"] = np.nan
        else:
            row[f"dim_{j}_volatility"] = float(np.mean(np.abs(np.diff(y))))
    return row


def _pct_diff_tts_vs_normal(normal_mu: float, tts_mu: float) -> float:
    if not np.isfinite(normal_mu) or not np.isfinite(tts_mu):
        return float("nan")
    denom = abs(float(normal_mu))
    if denom < 1e-15:
        return float("nan")
    return 100.0 * (float(tts_mu) - float(normal_mu)) / denom


def _corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    yv = np.asarray(y, dtype=int)
    xv = np.asarray(x, dtype=np.float64)
    n0, n1 = int(np.sum(yv == 0)), int(np.sum(yv == 1))
    out: dict = {
        "n_patients": int(len(yv)),
        "y_n_class0": n0,
        "y_n_class1": n1,
        "note": "",
    }
    if pearsonr is None:
        raise RuntimeError("scipy required; pip install scipy")
    if len(yv) < 3 or n0 == 0 or n1 == 0:
        out["note"] = "single_class_or_too_few"
    elif np.nanstd(xv) < 1e-14:
        out["note"] = "x_constant"
    if out["note"]:
        for k in (
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "pointbiserial_r",
            "pointbiserial_p",
        ):
            out[k] = np.nan
        return out
    m = np.isfinite(xv) & np.isfinite(yv.astype(float))
    if int(np.sum(m)) < 3:
        out["note"] = "too_few_finite_pairs"
        for k in (
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "pointbiserial_r",
            "pointbiserial_p",
        ):
            out[k] = np.nan
        return out
    xv2 = xv[m]
    yv2 = yv[m]
    pr = pearsonr(xv2, yv2.astype(np.float64))
    sr = spearmanr(xv2, yv2)
    pbr = pointbiserialr(yv2.astype(np.float64), xv2)
    out["pearson_r"] = float(pr.statistic)
    out["pearson_p"] = float(pr.pvalue)
    out["spearman_r"] = float(sr.statistic)
    out["spearman_p"] = float(sr.pvalue)
    out["pointbiserial_r"] = float(pbr.correlation)
    out["pointbiserial_p"] = float(pbr.pvalue)
    return out


def _two_sample_pvals(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mannwhitney_p_two_sided, ttest_p_two_sided); nan if scipy missing or insufficient."""
    if mannwhitneyu is None or ttest_ind is None:
        return float("nan"), float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    try:
        u = mannwhitneyu(a, b, alternative="two-sided")
        mw_p = float(u.pvalue)
    except ValueError:
        mw_p = float("nan")
    try:
        tt = ttest_ind(a, b, equal_var=False)
        tt_p = float(tt.pvalue)
    except ValueError:
        tt_p = float("nan")
    return mw_p, tt_p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument(
        "--dims",
        type=int,
        nargs="+",
        required=True,
        help="Latent dim indices j (same as z[:, j] in NPZ embeddings).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--respect_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NPZ mask row when present (default: true).",
    )
    ap.add_argument(
        "--std_ddof",
        type=int,
        default=0,
        help="ddof for slice std over valid rows (0 matches many training diagnostics).",
    )
    ap.add_argument(
        "--infer_normal_from_stem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If stem matches --normal_stem_regex and CSV has no row, use y_true=0 (normal). Default: on.",
    )
    ap.add_argument(
        "--normal_stem_regex",
        default=r".*_\d+[FM]$",
        help=r"Regex on NPZ stem (no .npz); default: *_<digits>F or M e.g. LBH_22392393_71F",
    )
    args = ap.parse_args()
    npz_dir = os.path.abspath(args.npz_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    labels = _merge_labels([os.path.abspath(p) for p in args.labels_csv])
    dims = sorted(set(args.dims))
    normal_pat = re.compile(args.normal_stem_regex, re.IGNORECASE)

    rows: list[dict] = []
    skipped: list[str] = []
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz"):
            continue
        pid = fn.replace(".npz", "")
        y_res, src = _resolve_y_for_npz_stem(
            pid, labels, normal_pat, args.infer_normal_from_stem
        )
        if y_res is None:
            skipped.append(f"no_label:{pid}")
            continue
        path = os.path.join(npz_dir, fn)
        feats = _patient_scalar_features(path, dims, args.respect_mask, args.std_ddof)
        if feats is None:
            skipped.append(f"bad_npz:{pid}")
            continue
        feats["patient_id"] = pid
        feats["y_true"] = int(y_res)
        feats["label_source"] = src
        rows.append(feats)

    if not rows:
        print("No aligned NPZ rows; check npz_dir and labels.", file=sys.stderr)
        sys.exit(1)

    feat_df = pd.DataFrame(rows)
    per_path = os.path.join(out_dir, "per_patient_features.csv")
    feat_df.to_csv(per_path, index=False)

    summary_rows: list[dict] = []
    test_rows: list[dict] = []
    corr_rows: list[dict] = []

    y = feat_df["y_true"].to_numpy(dtype=int)
    for j in dims:
        mean_col = f"dim_{j}_slice_mean"
        std_col = f"dim_{j}_slice_std"
        vol_col = f"dim_{j}_volatility"
        for feat_name, col in (
            ("slice_mean", mean_col),
            ("slice_std", std_col),
            ("volatility_absdiff_mean", vol_col),
        ):
            v = feat_df[col].to_numpy(dtype=np.float64)
            n0 = feat_df.loc[feat_df["y_true"] == 0, col]
            n1 = feat_df.loc[feat_df["y_true"] == 1, col]
            mu0 = float(np.nanmean(n0))
            mu1 = float(np.nanmean(n1))
            summary_rows.append(
                {
                    "latent_dim": j,
                    "feature": feat_name,
                    "normal_n": int(feat_df["y_true"].eq(0).sum()),
                    "tts_n": int(feat_df["y_true"].eq(1).sum()),
                    "normal_avg": mu0,
                    "tts_avg": mu1,
                    "pct_diff_tts_vs_normal": _pct_diff_tts_vs_normal(mu0, mu1),
                }
            )
            a = n0.to_numpy(dtype=np.float64)
            b = n1.to_numpy(dtype=np.float64)
            mw_p, tt_p = _two_sample_pvals(a, b)
            test_rows.append(
                {
                    "latent_dim": j,
                    "feature": feat_name,
                    "mannwhitney_p_two_sided": mw_p,
                    "welch_ttest_p_two_sided": tt_p,
                }
            )

        for feat_name, col in (("slice_std", std_col), ("volatility_absdiff_mean", vol_col)):
            cdict = _corr_block(feat_df[col].to_numpy(dtype=np.float64), y)
            corr_rows.append(
                {
                    "latent_dim": j,
                    "feature": feat_name,
                    **cdict,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "summary_normal_vs_tts.csv"), index=False
    )
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, "group_tests.csv"), index=False)
    pd.DataFrame(corr_rows).to_csv(
        os.path.join(out_dir, "correlation_vs_y_true.csv"), index=False
    )

    skip_path = os.path.join(out_dir, "skipped_npz.txt")
    with open(skip_path, "w", encoding="utf-8") as f:
        f.write("\n".join(skipped) if skipped else "(none)")

    print(f"Wrote {per_path}")
    print(f"Wrote {out_dir}/summary_normal_vs_tts.csv")
    print(f"Wrote {out_dir}/group_tests.csv")
    print(f"Wrote {out_dir}/correlation_vs_y_true.csv")
    print(f"Wrote {skip_path} ({len(skipped)} lines)")


if __name__ == "__main__":
    main()
