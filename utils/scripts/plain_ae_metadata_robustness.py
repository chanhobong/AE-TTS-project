#!/usr/bin/env python3
"""
Plain AE metadata / robustness analysis (patient-level).

Primary input is a wide or mergeable table with at least:
  patient_id, label, age, sex, and one continuous ``primary_score_col``
(e.g. per-slice P90−P10 L2 from your own ``patient_recon_summary``/latent table,
   L1 reconstruction error per patient, or ElasticNet decision values).

Analyses
--------
  - Global AUC of label vs primary score + bootstrap CI (ranks raw score as-is).
  - Subgroup AUC: all / age < cutoff / age ≥ cutoff; male / female.
  - Confounding (statsmodels if installed): OLS score ~ age + sex (+ optional site via ``--site_col``);
    logit label ~ score + age + sex (+ optional site).
  - Optional: pick repeats.csv row whose test roc_auc is closest to median;
    plot normals-only age vs score on that repeat's test_patient_ids.

Dependencies: numpy, pandas, scikit-learn, matplotlib.
Optional: statsmodels (OLS/logit p-values, ORs); scipy (normal-sex score test).

Examples
--------
  python3 utils/scripts/plain_ae_metadata_robustness.py \\
    --scores_csv outputs/patient_scores.csv \\
    --demographics_csv data/train_val_demo.csv \\
    --out_dir outputs/metad_rob \\
    --primary_score_col latent_p90_p10_l2

  # With site as fixed effect: add --site_col site (column must exist in demographics).

  python3 utils/scripts/plain_ae_metadata_robustness.py \\
    --merged_csv outputs/all_in_one.csv \\
    --repeats_csv path/to/repeats.csv \\
    --out_dir outputs/metad_rob \\
    --primary_score_col reconstruction_l1_mean
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required for plots.") from e

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.formula.api as smf

    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from statsmodels.regression.mixed_linear_model import MixedLM

    HAS_MIXED = True
except ImportError:
    HAS_MIXED = False

try:
    from scipy.stats import mannwhitneyu, ttest_ind

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _read_table(path: str) -> pd.DataFrame:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise SystemExit(
            f"File not found: {path}\n"
            "  Fix --scores_csv / --demographics_csv / --merged_csv path. "
            "If it lives on an external drive, mount it (e.g. /Volumes/...) first "
            "or copy the CSV next to the repo and use that path."
        )
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def merge_scores_and_demographics(
    scores_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    *,
    site_col: Optional[str],
) -> pd.DataFrame:
    """Inner join on patient_id; demographics supply age/sex/(site), scores supply label/scores."""
    s = scores_df.copy()
    d = demo_df.copy()
    if "patient_id" not in s.columns:
        raise ValueError("scores table must contain patient_id")
    if "patient_id" not in d.columns:
        raise ValueError("demographics table must contain patient_id")
    s["patient_id"] = s["patient_id"].astype(str)
    d["patient_id"] = d["patient_id"].astype(str)

    keep_demo = ["patient_id", "age", "sex"]
    if site_col and site_col in d.columns:
        keep_demo.append(site_col)
    missing = [c for c in keep_demo if c not in d.columns]
    if missing:
        raise ValueError(f"demographics missing columns: {missing}")

    sub = d[keep_demo].copy()
    sub["age"] = pd.to_numeric(sub["age"], errors="coerce")
    sub["sex"] = sub["sex"].astype(str).str.strip().str.upper()

    out = s.merge(sub, on="patient_id", how="inner")
    out = out.dropna(subset=["age"])
    return out


def _sex_male_indicator(sex: pd.Series) -> pd.Series:
    s = sex.astype(str).str.strip().str.upper()
    male = s.isin(("M", "MALE", "1", "남"))
    female = s.isin(("F", "FEMALE", "2", "여"))
    unk = ~(male | female)
    out = male.astype(float)
    out = out.where(~unk, other=np.nan)
    return out


def bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return (auc, q025, q975) from case resampling on patients (rows)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    base = roc_auc_score(y, s)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], s[idx])))
    if not aucs:
        return float(base), float("nan"), float("nan")
    q = np.quantile(aucs, [0.025, 0.975])
    return float(base), float(q[0]), float(q[1])


def subgroup_auc_table(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    age_col: str,
    age_cutoff: float,
    min_per_class: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def one(name: str, sub: pd.DataFrame) -> None:
        y = sub[label_col].to_numpy(dtype=np.int64)
        sc = sub[score_col].to_numpy(dtype=np.float64)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        if n0 < min_per_class or n1 < min_per_class:
            rows.append(
                {
                    "subgroup": name,
                    "n": int(len(sub)),
                    "n_normal": n0,
                    "n_tts": n1,
                    "roc_auc": np.nan,
                    "boot_q025": np.nan,
                    "boot_q975": np.nan,
                    "note": "skipped_low_n",
                }
            )
            return
        auc, lo, hi = bootstrap_auc(y, sc, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "subgroup": name,
                "n": int(len(sub)),
                "n_normal": n0,
                "n_tts": n1,
                "roc_auc": auc,
                "boot_q025": lo,
                "boot_q975": hi,
                "note": "",
            }
        )

    one("all", df)
    young = df[df[age_col] < age_cutoff]
    old = df[df[age_col] >= age_cutoff]
    one(f"age_lt_{age_cutoff:g}", young)
    one(f"age_ge_{age_cutoff:g}", old)
    return pd.DataFrame(rows)


def sex_subgroup_aucs(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    sex_col: str,
    min_per_class: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tag, mask in (
        ("female", df[sex_col].astype(str).str.upper().isin(("F", "FEMALE", "2", "여"))),
        ("male", df[sex_col].astype(str).str.upper().isin(("M", "MALE", "1", "남"))),
    ):
        sub = df.loc[mask]
        y = sub[label_col].to_numpy(dtype=np.int64)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        if n0 < min_per_class or n1 < min_per_class:
            rows.append(
                {
                    "sex_subgroup": tag,
                    "n": int(len(sub)),
                    "n_normal": n0,
                    "n_tts": n1,
                    "roc_auc": np.nan,
                    "boot_q025": np.nan,
                    "boot_q975": np.nan,
                    "note": "skipped_low_n",
                }
            )
            continue
        sc = sub[score_col].to_numpy(dtype=np.float64)
        auc, lo, hi = bootstrap_auc(y, sc, n_boot=n_boot, seed=seed + hash(tag) % 10000)
        rows.append(
            {
                "sex_subgroup": tag,
                "n": int(len(sub)),
                "n_normal": n0,
                "n_tts": n1,
                "roc_auc": auc,
                "boot_q025": lo,
                "boot_q975": hi,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def normals_sex_primary_score_test(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    sex_col: str,
) -> dict[str, Any]:
    """Normal (label==0) only: compare score between male and female."""
    sub = df[df[label_col] == 0].copy()
    sx = sub[sex_col].astype(str).str.strip().str.upper()
    m_mask = sx.isin(("M", "MALE", "1", "남"))
    f_mask = sx.isin(("F", "FEMALE", "2", "여"))
    x_m = sub.loc[m_mask, score_col].dropna().to_numpy(dtype=np.float64)
    x_f = sub.loc[f_mask, score_col].dropna().to_numpy(dtype=np.float64)
    out: dict[str, Any] = {
        "n_normal_male": int(x_m.size),
        "n_normal_female": int(x_f.size),
        "mean_male": float(np.mean(x_m)) if x_m.size else float("nan"),
        "mean_female": float(np.mean(x_f)) if x_f.size else float("nan"),
    }
    if not HAS_SCIPY or x_m.size < 2 or x_f.size < 2:
        out["note"] = "insufficient_n_or_no_scipy"
        return out
    tt = ttest_ind(x_m, x_f, equal_var=False)
    mw = mannwhitneyu(x_m, x_f, alternative="two-sided")
    out["ttest_statistic"] = float(tt.statistic)
    out["ttest_pvalue"] = float(tt.pvalue)
    out["mannwhitney_statistic"] = float(mw.statistic)
    out["mannwhitney_pvalue"] = float(mw.pvalue)
    return out


def run_confound_models(
    df: pd.DataFrame,
    *,
    primary_score: str,
    label_col: str,
    site_col: Optional[str],
) -> dict[str, Any]:
    """OLS for score ~ confounders; logit for label ~ score + confounders."""
    d = df.copy()
    d["sex_male"] = _sex_male_indicator(d["sex"])
    d = d.dropna(subset=["sex_male"])
    result: dict[str, Any] = {"has_statsmodels": HAS_SM}

    site_term = f" + C({site_col})" if site_col and site_col in d.columns and d[site_col].nunique() > 1 else ""

    if HAS_SM:
        fo_score = f"{primary_score} ~ age + sex_male{site_term}"
        try:
            fit_o = smf.ols(fo_score, data=d).fit()
            result["ols_formula"] = fo_score
            result["ols_summary"] = fit_o.summary().as_text()
            result["ols_params"] = fit_o.params.to_dict()
            result["ols_pvalues"] = fit_o.pvalues.to_dict()
        except Exception as ex:
            result["ols_error"] = repr(ex)

        fo_log = f"{label_col} ~ {primary_score} + age + sex_male{site_term}"
        try:
            fit_l = smf.logit(fo_log, data=d).fit(disp=False, maxiter=200)
            result["logit_formula"] = fo_log
            result["logit_summary"] = fit_l.summary().as_text()
            result["logit_params"] = fit_l.params.to_dict()
            result["logit_pvalues"] = fit_l.pvalues.to_dict()
            result["logit_or"] = np.exp(fit_l.params).to_dict()
        except Exception as ex:
            result["logit_error"] = repr(ex)

        dm = d.dropna(subset=[site_col]) if site_col and site_col in d.columns else d
        if (
            HAS_MIXED
            and site_col
            and site_col in dm.columns
            and dm[site_col].nunique() > 1
            and len(dm) >= 10
        ):
            try:
                exog = pd.DataFrame({"Intercept": 1.0, "age": dm["age"].astype(float), "sex_male": dm["sex_male"]})
                md = MixedLM(dm[primary_score].astype(float), exog, groups=dm[site_col])
                mfit = md.fit(reml=False, disp=False)
                result["mixedlm_summary"] = mfit.summary().as_text()
            except Exception as ex:
                result["mixedlm_error"] = repr(ex)
    else:
        Xs = pd.get_dummies(
            d[["age", "sex_male"]].assign(**{site_col: d[site_col]} if site_col and site_col in d else {}),
            columns=[site_col] if site_col and site_col in d.columns else [],
            drop_first=True,
        )
        y_score = d[primary_score].to_numpy(dtype=np.float64)
        lr = LinearRegression().fit(Xs, y_score)
        result["sklearn_ols_r2"] = float(lr.score(Xs, y_score))
        result["sklearn_ols_coef"] = dict(zip(Xs.columns, lr.coef_.tolist()))

        Xl = pd.concat([d[[primary_score]], Xs], axis=1)
        yl = d[label_col].to_numpy(dtype=np.int64)
        log = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)
        sc = StandardScaler()
        Xls = sc.fit_transform(Xl)
        log.fit(Xls, yl)
        result["sklearn_logit_note"] = "Install statsmodels for OR/SE; sklearn fallback fitted on scaled X."

    return result


def pick_median_repeat_row(
    repeats: pd.DataFrame,
    *,
    metric_col: str = "roc_auc",
    id_col: str = "repeat_id",
) -> pd.Series:
    if metric_col not in repeats.columns:
        raise ValueError(f"repeats CSV missing {metric_col}")
    med = repeats[metric_col].median()
    tmp = repeats.assign(_abs_diff=(repeats[metric_col] - med).abs())
    tmp = tmp.sort_values(["_abs_diff", id_col], ascending=[True, True])
    return tmp.iloc[0]


def parse_test_ids(row: pd.Series) -> list[str]:
    for col in ("test_patient_ids", "test_patients"):
        if col in row.index and isinstance(row[col], str) and row[col].strip():
            return [x.strip() for x in row[col].split(";") if x.strip()]
    raise ValueError("repeats row: need test_patient_ids (semicolon-separated)")


def plot_normals_age_vs_score(
    df: pd.DataFrame,
    *,
    age_col: str,
    score_col: str,
    label_col: str,
    out_path: str,
    title: str,
) -> None:
    sub = df[df[label_col] == 0]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(sub[age_col], sub[score_col], alpha=0.72, edgecolors="none", s=38, c="#1f77b4")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(score_col)
    ax.set_title(title)
    fig.tight_layout()
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_merged_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, Optional[str]]:
    site_col = args.site_col.strip() or None

    if args.merged_csv:
        df = _read_table(args.merged_csv)
        if "patient_id" not in df.columns:
            raise ValueError("merged_csv must contain patient_id")
        df["patient_id"] = df["patient_id"].astype(str)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df.dropna(subset=["age"])
        if site_col and site_col not in df.columns:
            print(f"[warn] site column {site_col!r} not in merged table; analyses drop site.", file=sys.stderr)
            site_col = None
        return df, site_col

    if not args.scores_csv or not args.demographics_csv:
        raise SystemExit("Provide --merged_csv OR both --scores_csv and --demographics_csv")

    scores = _read_table(args.scores_csv)
    demo = _read_table(args.demographics_csv)
    if site_col and site_col not in demo.columns:
        print(f"[warn] site column {site_col!r} not in demographics; site omitted.", file=sys.stderr)
        site_col = None

    df = merge_scores_and_demographics(scores, demo, site_col=site_col)
    return df, site_col


def main() -> None:
    p = argparse.ArgumentParser(description="Plain AE patient-level metadata / robustness analysis")
    p.add_argument("--merged_csv", type=str, default="", help="Wide table with scores + age + sex + label")
    p.add_argument("--scores_csv", type=str, default="", help="Per-patient scores (must include patient_id, label)")
    p.add_argument("--demographics_csv", type=str, default="", help="patient_id, age, sex, optional site")
    p.add_argument(
        "--extra_scores_csv",
        type=str,
        default="",
        help="Optional CSV with patient_id + extra columns merged left onto scores",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--primary_score_col", type=str, default="latent_p90_p10_l2")
    p.add_argument(
        "--site_col",
        type=str,
        default="",
        help="Hospital/site column name for OLS/logit (default: empty = omit site). Example: --site_col site",
    )
    p.add_argument("--age_cutoff", type=float, default=60.0)
    p.add_argument("--min_class_n", type=int, default=5, help="Min per-class count for subgroup AUC")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--repeats_csv",
        type=str,
        default="",
        help="repeats.csv: pick row nearest median metric; use test_patient_ids for plot subset",
    )
    p.add_argument(
        "--repeats_metric_col",
        type=str,
        default="roc_auc",
        help="Column in repeats_csv to match median (default: test roc_auc)",
    )
    p.add_argument(
        "--skip_normal_sex_test",
        action="store_true",
        help="Do not write normals_sex_primary_score_test.json (Welch t + Mann–Whitney)",
    )
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df, site_col = build_merged_frame(args)
    if args.extra_scores_csv.strip():
        extra = _read_table(args.extra_scores_csv)
        if "patient_id" not in extra.columns:
            raise SystemExit("extra_scores_csv must contain patient_id")
        extra["patient_id"] = extra["patient_id"].astype(str)
        df["patient_id"] = df["patient_id"].astype(str)
        df = df.merge(extra, on="patient_id", how="left", suffixes=("", "_extra"))

    label_col = args.label_col
    score_col = args.primary_score_col

    if score_col not in df.columns:
        raise SystemExit(f"primary score column missing: {score_col!r}. Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise SystemExit(f"label column missing: {label_col!r}")

    da = df[np.isfinite(df[score_col].to_numpy(dtype=np.float64))].copy()

    y_all = da[label_col].to_numpy(dtype=np.int64)
    s_all = da[score_col].to_numpy(dtype=np.float64)
    auc, lo, hi = bootstrap_auc(y_all, s_all, n_boot=int(args.n_bootstrap), seed=int(args.random_state))
    pd.DataFrame([{"roc_auc": auc, "boot_q025": lo, "boot_q975": hi, "n": len(da)}]).to_csv(
        os.path.join(out_dir, "auc_global_primary_score.csv"), index=False
    )

    subgroup_auc_table(
        da,
        label_col=label_col,
        score_col=score_col,
        age_col="age",
        age_cutoff=float(args.age_cutoff),
        min_per_class=int(args.min_class_n),
        n_boot=int(args.n_bootstrap),
        seed=int(args.random_state),
    ).to_csv(os.path.join(out_dir, "auc_by_age_subgroup.csv"), index=False)

    sex_subgroup_aucs(
        da,
        label_col=label_col,
        score_col=score_col,
        sex_col="sex",
        min_per_class=int(args.min_class_n),
        n_boot=int(args.n_bootstrap),
        seed=int(args.random_state),
    ).to_csv(os.path.join(out_dir, "auc_by_sex_subgroup.csv"), index=False)

    if not args.skip_normal_sex_test:
        ns = normals_sex_primary_score_test(
            da, label_col=label_col, score_col=score_col, sex_col="sex"
        )
        with open(os.path.join(out_dir, "normals_sex_primary_score_test.json"), "w", encoding="utf-8") as f:
            json.dump(ns, f, indent=2, default=str)

    conf = run_confound_models(
        da,
        primary_score=score_col,
        label_col=label_col,
        site_col=site_col,
    )
    with open(os.path.join(out_dir, "confounder_models.json"), "w", encoding="utf-8") as f:
        slim = {k: v for k, v in conf.items() if not k.endswith("_summary")}
        json.dump(slim, f, indent=2, default=str)
    if "ols_summary" in conf:
        with open(os.path.join(out_dir, "ols_confounders.txt"), "w", encoding="utf-8") as f:
            f.write(conf["ols_summary"])
    if "logit_summary" in conf:
        with open(os.path.join(out_dir, "logit_adjusted.txt"), "w", encoding="utf-8") as f:
            f.write(conf["logit_summary"])
    if "mixedlm_summary" in conf:
        with open(os.path.join(out_dir, "mixedlm_score_random_site.txt"), "w", encoding="utf-8") as f:
            f.write(conf["mixedlm_summary"])

    if not HAS_SM:
        print(
            "[note] statsmodels not installed; confounder_models.json has sklearn fallback only. "
            "pip install statsmodels for full OLS/logit OR and p-values.",
            file=sys.stderr,
        )

    if args.repeats_csv:
        rep = _read_table(args.repeats_csv)
        row = pick_median_repeat_row(rep, metric_col=args.repeats_metric_col)
        repeat_id = int(row.get("repeat_id", -1))
        test_ids = set(parse_test_ids(row))
        d_test = da[da["patient_id"].isin(test_ids)].copy()
        meta = {
            "repeat_id": repeat_id,
            "target_median_metric": float(rep[args.repeats_metric_col].median()),
            "this_repeat_metric": float(row[args.repeats_metric_col]),
            "n_test_patients_listed": len(test_ids),
            "n_in_merged_after_join": int(len(d_test)),
        }
        with open(os.path.join(out_dir, "median_repeat_selection.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        title = f"Normals: age vs {score_col}\n(median-{args.repeats_metric_col} repeat {repeat_id})"
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", score_col)
        plot_normals_age_vs_score(
            d_test,
            age_col="age",
            score_col=score_col,
            label_col=label_col,
            out_path=os.path.join(out_dir, f"scatter_normals_age_vs_{slug}_median_repeat.png"),
            title=title,
        )
        d_test.to_csv(os.path.join(out_dir, "table_median_repeat_test_cohort.csv"), index=False)

    da.to_csv(os.path.join(out_dir, "analysis_table_merged.csv"), index=False)
    print(f"Wrote outputs under {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
