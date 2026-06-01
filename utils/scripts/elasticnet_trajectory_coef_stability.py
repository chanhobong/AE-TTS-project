#!/usr/bin/env python3
"""
ElasticNet coefficient stability / sparsity for trajectory pooling features
(p90_p10, vol_mm, p90_p10_vol_mm, std, …) using the same build protocol as
``repeated_head_sweep_std.py``.

The original head sweep only saves AUC summaries; this script **re-fits** the same
StratifiedShuffleSplit + StandardScaler + LogisticRegression(saga, elasticnet)
and records ``coef_`` in **scaled feature space** (as in the sweep).

Outputs (under ``--out_dir``)
-----------------------------
- ``coef_repeat_matrix.npy`` — (n_splits, D) stacked ``coef_`` rows
- ``repeat_metadata.csv`` — repeat_id, n_nonzero_coef, train_n, test_n, roc_auc_te, pr_auc_te (optional)
- ``sparsity_summary.csv`` — nnz per repeat + distribution stats
- ``feature_blocks.csv`` — explicit block ↔ global feature index ranges (**must match** ``--pooling`` / head sweep)
- ``order_annotations.csv`` — ``profile_order`` vs ``trajectory_order`` (see cohort PNG vs coef / peak / cluster)
- ``dim_selection_frequency.csv`` — ``feature_index`` (same as coef vector index), ``block``, ``latent_dim``, ``nonzero_freq``,
  mean_coef, std_coef, mean_abs_coef (``nonzero_freq`` uses ``|coef| > --nnz_tol``)
- ``stable_top_dims.csv`` — subset with ``nonzero_freq >= --stable_freq_thr`` sorted by ``mean_abs_coef``
- ``group_diff_stable_dims.csv`` — Welch summary on **raw** patient features ``X`` (pre-scaler)
  for stable dims (needs scipy)
- ``peak_stack_position_stable_dims.csv`` — per stable latent dim: argmax |z[t,j]| along **trajectory-ordered**
  slices; group mean position in [0,1] + optional MWU p-value
- ``cluster_slice_eta2_stable_dims.csv`` — when ``cluster_ids`` exist in NPZ: one-way η² of z[t,j] across
  slice clusters (trajectory order), mean per class

Notes
-----
* **Same feature basis as the sweep:** pass ``--pooling`` exactly as in ``repeated_head_sweep_std.py`` for the head you
  interpret (``p90_p10`` only vs ``p90_p10_vol_mm`` concat → different ``D`` and coef indexing).
* Feature **block** for ``p90_p10_vol_mm``: indices ``0..D-1`` → ``p90_p10`` latent 0..D-1;
  ``D..2D-1`` → ``vol_mm`` latent 0..D-1 (see ``feature_blocks.csv``).
* **Reconstruction / latent intervention (item 6):** StageB ``*.npz`` holds per-slice embeddings only.
  Volumetric decode needs the trained encoder/decoder weights and preprocessing pipeline
  (see ``models/ae.py``, ``utils/analysis/latent_interpolation_analysis.py``). Not run here.

Example
-------
  python elasticnet_trajectory_coef_stability.py \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --npz_dir /path/to/diff3dformer_stageB_plain_ae_slice_meta/patients \\
    --pooling p90_p10_vol_mm \\
    --C 0.1 --l1_ratio 0.2 \\
    --n_splits 100 --random_state 42 \\
    --out_dir ./coef_stability_en \\
    --plot_top_k 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from repeated_head_sweep_std import (  # noqa: E402
    HEAD_SWEEP_TRAJECTORY_POOLINGS,
    _build_ds,
    _scores,
    _fit,
)
import trajectory_volatility_classifier_cv as traj_cv  # noqa: E402

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _infer_d_latent(pooling: str, d_feat: int) -> int:
    if pooling in ("p90_p10_vol_mm", "p90_p10_global_mean_vol_mm", "p90_p10_p90_vol_mm"):
        if d_feat % 2 != 0:
            raise SystemExit(f"Expected even D for {pooling}, got {d_feat}")
        return d_feat // 2
    if pooling == "regional_vol_mm":
        if d_feat % 3 != 0:
            raise SystemExit(f"Expected D divisible by 3 for regional_vol_mm, got {d_feat}")
        return d_feat // 3
    if pooling == "p90_p10_regional_vol_mm":
        if d_feat % 4 != 0:
            raise SystemExit(f"Expected D = 4×D_latent for p90_p10_regional_vol_mm, got {d_feat}")
        return d_feat // 4
    return d_feat


def _feature_meta(feature_index: int, pooling: str, d_latent: int) -> tuple[str, int]:
    """Return (block_name, latent_dim_index_within_z)."""
    if pooling == "p90_p10_vol_mm":
        if feature_index < d_latent:
            return "p90_p10", int(feature_index)
        return "vol_mm", int(feature_index - d_latent)
    if pooling == "p90_p10_global_mean_vol_mm":
        if feature_index < d_latent:
            return "p90_p10", int(feature_index)
        return "global_mean_vol_mm", int(feature_index - d_latent)
    if pooling == "p90_p10_p90_vol_mm":
        if feature_index < d_latent:
            return "p90_p10", int(feature_index)
        return "p90_vol_mm", int(feature_index - d_latent)
    if pooling == "regional_vol_mm":
        third = feature_index // d_latent
        return f"regional_third{int(third)}", int(feature_index % d_latent)
    if pooling == "p90_p10_regional_vol_mm":
        if feature_index < d_latent:
            return "p90_p10", int(feature_index)
        kk = feature_index - d_latent
        third = kk // d_latent
        return f"regional_third{int(third)}", int(kk % d_latent)
    if pooling in ("p90_p10", "vol_mm", "std", "global_mean_vol_mm", "p90_vol_mm"):
        return str(pooling), int(feature_index)
    return "feature", int(feature_index)


def _write_feature_blocks_csv(path: Path, pooling: str, d_feat: int, d_latent: int) -> None:
    """One row per contiguous block in the coef / X vector (global feature_index)."""
    rows: list[dict[str, object]] = []

    def row(block: str, imin: int, imax: int, lmin: int, lmax: int, desc: str) -> None:
        rows.append(
            {
                "block": block,
                "feature_index_min": imin,
                "feature_index_max": imax,
                "latent_dim_min": lmin,
                "latent_dim_max": lmax,
                "n_block_features": imax - imin + 1,
                "description": desc,
            }
        )

    if pooling == "p90_p10_vol_mm":
        row("p90_p10", 0, d_latent - 1, 0, d_latent - 1, "P90−P10 global variability")
        row("vol_mm", d_latent, d_feat - 1, 0, d_latent - 1, "Global mean transition rate |dz|/dmm")
    elif pooling == "p90_p10_global_mean_vol_mm":
        row("p90_p10", 0, d_latent - 1, 0, d_latent - 1, "P90−P10 global variability")
        row("global_mean_vol_mm", d_latent, d_feat - 1, 0, d_latent - 1, "Same as vol_mm (mean transition rate)")
    elif pooling == "p90_p10_p90_vol_mm":
        row("p90_p10", 0, d_latent - 1, 0, d_latent - 1, "P90−P10 global variability")
        row("p90_vol_mm", d_latent, d_feat - 1, 0, d_latent - 1, "Per-dim 90th pct of transition rates")
    elif pooling == "regional_vol_mm":
        for t in range(3):
            lo, hi = t * d_latent, (t + 1) * d_latent - 1
            row(f"regional_third{t}", lo, hi, 0, d_latent - 1, f"Basal/mid/apical third {t}; mean transition rate in region")
    elif pooling == "p90_p10_regional_vol_mm":
        row("p90_p10", 0, d_latent - 1, 0, d_latent - 1, "P90−P10 global variability")
        for t in range(3):
            lo = d_latent + t * d_latent
            hi = lo + d_latent - 1
            row(f"regional_third{t}", lo, hi, 0, d_latent - 1, f"Regional third {t} transition rates")
    elif pooling in ("p90_p10", "vol_mm", "std", "global_mean_vol_mm", "p90_vol_mm"):
        row(pooling, 0, d_feat - 1, 0, d_latent - 1, f"Single-block {pooling}")
    else:
        row(pooling, 0, d_feat - 1, 0, d_latent - 1, "Verify layout manually")
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_order_annotations_csv(path: Path) -> None:
    """Slice-profile plot order vs trajectory feature construction order."""
    rows = [
        {
            "key": "profile_order",
            "value": "original_npz_order",
            "applies_to": "cohort_profile_latent_*.png (plot_latent_slice_profile._plot_cohort_bands)",
            "detail": "Valid rows = mask==1 & finite emb; order = ascending NPZ row index among those rows; "
            "then resampled to profile_n_grid. Not re-sorted by slice_z_mm.",
        },
        {
            "key": "trajectory_order",
            "value": "sorted_by_slice_z_mm",
            "applies_to": "patient X; coef[k]; peak_stack_position_*.csv; cluster_slice_eta2_*.csv",
            "detail": "Sort key (slice_z_mm, slice_index); if no usable mm, slice_index only. "
            "See trajectory_volatility_classifier_cv._ordered_stack.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _nonzero_mask(coef: np.ndarray, tol: float) -> np.ndarray:
    c = np.asarray(coef, dtype=np.float64).ravel()
    thr = max(float(tol), 1e-15 * float(np.max(np.abs(c))) if c.size else 0.0)
    return np.abs(c) > thr


def _eta2_oneway_slice(y: np.ndarray, g: np.ndarray) -> float:
    """η² for slice-level values y with group labels g (same length)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.int64).reshape(-1)
    if y.size < 2 or y.shape != g.shape:
        return float("nan")
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    if y.size < 2:
        return float("nan")
    uniq = np.unique(g)
    if uniq.size < 2:
        return float("nan")
    grand_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - grand_mean) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for k in uniq:
        m = g == k
        nk = float(np.sum(m))
        if nk < 1:
            continue
        mk = float(np.mean(y[m]))
        ss_between += nk * (mk - grand_mean) ** 2
    return float(np.clip(ss_between / ss_tot, 0.0, 1.0))


def _cluster_eta2_dim_npz(npz_path: str, dim_j: int) -> float:
    z = np.load(npz_path, allow_pickle=True)
    emb, _, _, cid = traj_cv._ordered_stack(z)
    if emb.shape[0] < 2 or cid is None or dim_j < 0 or dim_j >= emb.shape[1]:
        return float("nan")
    return _eta2_oneway_slice(emb[:, dim_j], cid)


def _rel_peak_pos_ordered_npz(npz_path: str, latent_j: int) -> float:
    """Argmax |z[t,j]| position normalized to [0,1] along trajectory-ordered rows."""
    z = np.load(npz_path, allow_pickle=True)
    emb, _, _ = traj_cv._ordered_rows(z)
    if emb.shape[0] == 0 or latent_j >= emb.shape[1] or latent_j < 0:
        return float("nan")
    col = np.abs(emb[:, latent_j])
    ix = int(np.argmax(col))
    denom = max(1, emb.shape[0] - 1)
    return float(ix / denom)


def main() -> None:
    ap = argparse.ArgumentParser(description="ElasticNet coef stability for trajectory pooling features.")
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument(
        "--pooling",
        choices=["std", *HEAD_SWEEP_TRAJECTORY_POOLINGS],
        required=True,
        help="Must match repeated_head_sweep_std --pooling for the head you interpret (defines X columns / coef layout).",
    )
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--l1_ratio", type=float, required=True)
    ap.add_argument("--eps_mm", type=float, default=1e-3)
    ap.add_argument("--recursive_npz", action="store_true")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--nnz_tol", type=float, default=1e-6, help="|coef| count as nonzero if above this (and rel floor).")
    ap.add_argument(
        "--stable_freq_thr",
        type=float,
        default=0.5,
        help="Mark dim as stable if nonzero in at least this fraction of repeats.",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--plot_top_k", type=int, default=0, help="If >0, save cohort latent profiles for top-K stable dims.")
    ap.add_argument("--profile_n_grid", type=int, default=48)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _build_ds(
        str(args.npz_dir),
        list(args.labels_csv),
        str(args.pooling),
        float(args.eps_mm),
        bool(args.recursive_npz),
    )
    if np.unique(ds.y).size < 2:
        raise SystemExit("Only one class after join; cannot fit.")

    d_feat = int(ds.X.shape[1])
    d_latent = _infer_d_latent(str(args.pooling), d_feat)

    _write_feature_blocks_csv(out_dir / "feature_blocks.csv", str(args.pooling), d_feat, d_latent)
    _write_order_annotations_csv(out_dir / "order_annotations.csv")

    splitter = StratifiedShuffleSplit(
        n_splits=int(args.n_splits),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    cfg = {"model": "logistic_en", "C": float(args.C), "l1_ratio": float(args.l1_ratio)}
    coef_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for repeat_id, (tr, te) in enumerate(splitter.split(ds.X, ds.y)):
        Xtr_raw, Xte_raw = ds.X[tr], ds.X[te]
        ytr, yte = ds.y[tr], ds.y[te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_raw)
        Xte = sc.transform(Xte_raw)
        model = _fit("logistic_en", cfg, seed=int(args.random_state))
        model.fit(Xtr, ytr)
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        if coef.shape[0] != d_feat:
            raise RuntimeError(f"coef dim {coef.shape[0]} != D {d_feat}")
        coef_rows.append(coef.copy())
        s_te = _scores(model, Xte)
        s_tr = _scores(model, Xtr)
        nnz = int(np.sum(_nonzero_mask(coef, float(args.nnz_tol))))
        meta_rows.append(
            {
                "repeat_id": int(repeat_id),
                "n_nonzero_coef": nnz,
                "train_size": int(len(tr)),
                "test_size": int(len(te)),
                "roc_auc_te": float(roc_auc_score(yte, s_te)),
                "pr_auc_te": float(average_precision_score(yte, s_te)),
                "roc_auc_tr": float(roc_auc_score(ytr, s_tr)),
                "pr_auc_tr": float(average_precision_score(ytr, s_tr)),
            }
        )

    coef_mat = np.stack(coef_rows, axis=0)
    np.save(out_dir / "coef_repeat_matrix.npy", coef_mat)
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(out_dir / "repeat_metadata.csv", index=False)

    nnz_all = df_meta["n_nonzero_coef"].to_numpy(dtype=np.float64)
    pd.DataFrame(
        [
            {
                "n_splits": int(args.n_splits),
                "nnz_mean": float(np.mean(nnz_all)),
                "nnz_std": float(np.std(nnz_all, ddof=0)),
                "nnz_median": float(np.median(nnz_all)),
                "nnz_min": int(np.min(nnz_all)),
                "nnz_max": int(np.max(nnz_all)),
                "D_features": d_feat,
                "pooling": str(args.pooling),
            }
        ]
    ).to_csv(out_dir / "sparsity_summary.csv", index=False)

    tol = float(args.nnz_tol)
    hits = np.stack([_nonzero_mask(coef_mat[i], tol) for i in range(coef_mat.shape[0])], axis=0)
    freq = np.mean(hits.astype(np.float64), axis=0)
    mean_c = np.mean(coef_mat, axis=0)
    std_c = np.std(coef_mat, axis=0, ddof=0)
    mean_abs = np.mean(np.abs(coef_mat), axis=0)

    dim_rows = []
    for k in range(d_feat):
        blk, lj = _feature_meta(k, str(args.pooling), d_latent)
        dim_rows.append(
            {
                "coef_vector_index": k,
                "feature_index": k,
                "block": blk,
                "latent_dim": lj,
                "nonzero_freq": float(freq[k]),
                "mean_coef": float(mean_c[k]),
                "std_coef": float(std_c[k]),
                "mean_abs_coef": float(mean_abs[k]),
            }
        )
    df_dim = pd.DataFrame(dim_rows).sort_values("mean_abs_coef", ascending=False)
    df_dim.to_csv(out_dir / "dim_selection_frequency.csv", index=False)

    thr = float(args.stable_freq_thr)
    stable = df_dim[df_dim["nonzero_freq"] >= thr].copy()
    stable.to_csv(out_dir / "stable_top_dims.csv", index=False)

    # Raw-feature group differences (Welch) on full cohort for stable dims
    X_raw = np.asarray(ds.X, dtype=np.float64)
    y_all = np.asarray(ds.y, dtype=np.int64)
    g0 = y_all == 0
    g1 = y_all == 1
    gd_rows: list[dict] = []
    if scipy_stats is not None and not stable.empty:
        for _, r in stable.iterrows():
            k = int(r["feature_index"])
            a, b = X_raw[g0, k], X_raw[g1, k]
            t_stat, p_welch = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            gd_rows.append(
                {
                    "coef_vector_index": k,
                    "feature_index": k,
                    "block": r["block"],
                    "latent_dim": int(r["latent_dim"]),
                    "mean_class0": float(np.nanmean(a)),
                    "std_class0": float(np.nanstd(a, ddof=0)),
                    "n_class0": int(np.sum(g0)),
                    "mean_class1": float(np.nanmean(b)),
                    "std_class1": float(np.nanstd(b, ddof=0)),
                    "n_class1": int(np.sum(g1)),
                    "welch_t": float(t_stat),
                    "welch_p": float(p_welch),
                }
            )
    elif stable.empty:
        pass
    elif scipy_stats is None:
        print("[warn] scipy not installed; skip group_diff_stable_dims.csv", file=sys.stderr)
    if gd_rows:
        pd.DataFrame(gd_rows).to_csv(out_dir / "group_diff_stable_dims.csv", index=False)

    # Peak stack position (trajectory order) for stable latent dims
    npz_dir = os.path.abspath(str(args.npz_dir))
    pid_to_y = {str(p): int(ly) for p, ly in zip(ds.patient_id, ds.y)}
    pos_rows: list[dict] = []
    if not stable.empty:
        for _, r in stable.iterrows():
            blk = str(r["block"])
            lj = int(r["latent_dim"])
            poss0: list[float] = []
            poss1: list[float] = []
            for fn in sorted(os.listdir(npz_dir)):
                if not fn.endswith(".npz"):
                    continue
                pid = fn.replace(".npz", "")
                if pid not in pid_to_y:
                    continue
                ppos = _rel_peak_pos_ordered_npz(os.path.join(npz_dir, fn), lj)
                if not np.isfinite(ppos):
                    continue
                if pid_to_y[pid] == 0:
                    poss0.append(ppos)
                else:
                    poss1.append(ppos)
            rec: dict = {
                "profile_order": "original_npz_order",
                "trajectory_order": "sorted_by_slice_z_mm",
                "block": blk,
                "latent_dim": lj,
                "mean_peak_pos_class0": float(np.mean(poss0)) if poss0 else float("nan"),
                "mean_peak_pos_class1": float(np.mean(poss1)) if poss1 else float("nan"),
                "n_class0": len(poss0),
                "n_class1": len(poss1),
            }
            if scipy_stats is not None and len(poss0) >= 3 and len(poss1) >= 3:
                _, p_mw = scipy_stats.mannwhitneyu(poss0, poss1, alternative="two-sided")
                rec["mannwhitney_p_peakpos"] = float(p_mw)
            pos_rows.append(rec)
    if pos_rows:
        pd.DataFrame(pos_rows).to_csv(out_dir / "peak_stack_position_stable_dims.csv", index=False)

    has_cluster = False
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(npz_dir, fn)
        with np.load(path, allow_pickle=True) as z:
            if "cluster_ids" not in z.files:
                continue
            _, _, _, cid = traj_cv._ordered_stack(z)
            if cid is not None and np.unique(cid).size >= 2:
                has_cluster = True
                break

    cl_rows: list[dict] = []
    if has_cluster and not stable.empty:
        for _, r in stable.iterrows():
            blk = str(r["block"])
            lj = int(r["latent_dim"])
            e0: list[float] = []
            e1: list[float] = []
            for fn in sorted(os.listdir(npz_dir)):
                if not fn.endswith(".npz"):
                    continue
                pid = fn.replace(".npz", "")
                if pid not in pid_to_y:
                    continue
                p = os.path.join(npz_dir, fn)
                e2 = _cluster_eta2_dim_npz(p, lj)
                if not np.isfinite(e2):
                    continue
                if pid_to_y[pid] == 0:
                    e0.append(e2)
                else:
                    e1.append(e2)
            rec2: dict = {
                "profile_order": "n/a_for_this_metric",
                "trajectory_order": "sorted_by_slice_z_mm",
                "block": blk,
                "latent_dim": lj,
                "mean_eta2_slice_clusters_class0": float(np.mean(e0)) if e0 else float("nan"),
                "mean_eta2_slice_clusters_class1": float(np.mean(e1)) if e1 else float("nan"),
                "n_class0": len(e0),
                "n_class1": len(e1),
            }
            if scipy_stats is not None and len(e0) >= 3 and len(e1) >= 3:
                _, p_mw = scipy_stats.mannwhitneyu(e0, e1, alternative="two-sided")
                rec2["mannwhitney_p_eta2"] = float(p_mw)
            cl_rows.append(rec2)
    if cl_rows:
        pd.DataFrame(cl_rows).to_csv(out_dir / "cluster_slice_eta2_stable_dims.csv", index=False)

    # Cohort latent profiles: raw z[t, latent_j] (NPZ row order + mask; see plot_latent_slice_profile).
    if int(args.plot_top_k) > 0 and not stable.empty:
        try:
            from plot_latent_slice_profile import _plot_cohort_bands
        except ImportError:
            _plot_cohort_bands = None  # type: ignore
        if _plot_cohort_bands:
            lab = {str(p): int(v) for p, v in zip(ds.patient_id, ds.y)}
            top = stable.head(int(args.plot_top_k))
            for _, r in top.iterrows():
                lj = int(r["latent_dim"])
                title = (
                    f"Stable dim (ElasticNet) block={r['block']} latent_j={lj} "
                    f"(freq={r['nonzero_freq']:.2f})\n"
                    f"profile_order = original_npz_order (cohort plot)\n"
                    f"trajectory_order = sorted_by_slice_z_mm (coef / X features)"
                )
                out_png = str(out_dir / f"cohort_profile_latent_{r['block']}_dim{lj:03d}.png")
                _plot_cohort_bands(
                    npz_dir,
                    lab,
                    lj,
                    respect_mask=True,
                    n_grid=int(args.profile_n_grid),
                    n_std=1.0,
                    use_sem=False,
                    title=title,
                    out_png=out_png,
                    firstdiff=False,
                )
    elif int(args.plot_top_k) > 0 and stable.empty:
        print("[info] plot_top_k>0 but no stable dims at this threshold; skip plots", file=sys.stderr)

    print(f"Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
