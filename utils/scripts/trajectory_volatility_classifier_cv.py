#!/usr/bin/env python3
"""
Ordered-slice trajectory features + CV + Logistic (L2).

Default CV: ``StratifiedShuffleSplit`` (repeated stratified hold-out), e.g. 100×20% test.
Optional: ``--cv_mode kfold --k_folds 5`` for stratified K-fold.

Uses rich NPZ (e.g. diff3dformer_stageB_plain_ae_slice_meta):
  embeddings (T, D), mask (T,), slice_z_mm (T,), optional slice_index (T,).

Per patient:
  1) Keep rows with mask==1 & finite slice_z_mm.
  2) Sort by slice_z_mm, tie-break slice_index (then row order).
  3) std_j = std along T (ddof=0)
     p90p10_j = P90 - P10 along T
     vol_j = mean_t |z[t+1,j] - z[t,j]|   (axis 0 over consecutive slices)
     vol_mm_j = mean_t |dz| / max(delta_mm_t, eps)   (physical mm between neighbours)

Feature sets (concatenated, no within-patient standardization before patient matrix):
  std_only, vol_only, std_vol, p90_p10_only, p90_p10_vol, p90_p10_std,
  vol_mm_only, std_vol_mm, p90_p10_vol_mm, p90_p10_vol_mm_std

Evaluation: per fold/split, StandardScaler fit on train, LogisticRegression(L2, C=...),
ROC-AUC and PR-AUC on train and test.

Outputs (same schema as ``repeated_stratified_shuffle_eval.py``)

- ``repeats.csv`` — one row per split × (feature_set, C): repeat_id, train/test sizes,
  train_roc_auc, train_pr_auc, roc_auc, pr_auc, test_patient_ids
- ``summary.csv`` — per (latent_source, pooling, model): roc_mean/std/median/iqr/min/max/p2p5/p97p5, pr_*
- ``repeated_summary_all.csv`` — identical to ``summary.csv`` (merge script compatibility)
- ``cv_summary.csv`` — compact extras: cv_mode, feature_dim, blocks, roc_auc_p2_5, pr_auc_p2_5

Example
-------
  python trajectory_volatility_classifier_cv.py \\
    --npz_dir /Volumes/.../diff3dformer_stageB_plain_ae_slice_meta \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --split_dir data \\
    --n_shuffle_splits 100 --test_size 0.2 --c_values 0.1 1.0 \\
    --out_dir ./traj_vol_cv_out
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import latent_mean_std_dataset_loader as ldm  # noqa: E402

RANDOM_STATE = 42


def _discover_npz_paths(npz_dir: str, recursive: bool) -> list[str]:
    """Sorted .npz paths; default one directory level only (matches other scripts)."""
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    if not recursive:
        return ldm.discover_npz_files(npz_dir)
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))


def _npz_patient_id(z, stem: str) -> str:
    if "patient_id" not in z.files:
        return stem
    v = z["patient_id"]
    try:
        v = v.item()
    except (AttributeError, ValueError):
        v = np.asarray(v).reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    return str(v)


def _ordered_stack(z) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
    """
    Returns (embeddings_ordered, z_mm_ordered, D_latent, cluster_ids_ordered or None).
    Same row ordering/mask rules as historical _ordered_rows; cluster_ids aligned if present.
    """
    emb = np.asarray(z["embeddings"], dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {emb.shape}")
    T, d = emb.shape
    mask = np.ones(T, dtype=bool)
    if "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1)
        if m.shape[0] == T:
            mask = m.astype(bool)
    if "slice_z_mm" in z.files:
        zmm = np.asarray(z["slice_z_mm"], dtype=np.float64).reshape(-1)
        has_mm = True
    else:
        zmm = np.zeros(T, dtype=np.float64)
        has_mm = False
    if "slice_index" in z.files:
        sidx = np.asarray(z["slice_index"], dtype=np.int64).reshape(-1)
    else:
        sidx = np.arange(T, dtype=np.int64)
    if zmm.shape[0] != T or sidx.shape[0] != T:
        raise ValueError("slice_z_mm / slice_index length must match embeddings rows")
    mm_ok = np.isfinite(zmm) if has_mm else np.ones(T, dtype=bool)
    valid = mask & mm_ok & np.isfinite(emb).all(axis=1)
    w = np.flatnonzero(valid)
    if w.size == 0:
        return emb[:0], zmm[:0], d, None
    if has_mm and np.any(np.isfinite(zmm[w])):
        order = w[np.lexsort((sidx[w], zmm[w]))]
    else:
        order = w[np.argsort(sidx[w], kind="mergesort")]
    emb_o = emb[order]
    zmm_o = zmm[order]
    cids_o: np.ndarray | None = None
    if "cluster_ids" in z.files:
        cid = np.asarray(z["cluster_ids"]).reshape(-1)
        if cid.shape[0] == T:
            cids_o = cid[order]
    return emb_o, zmm_o, d, cids_o


def _ordered_rows(z) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (embeddings_ordered (T', D), z_mm_ordered (T'), D_latent).
    Skips np.inf rows in z_mm. Uses mask if present.
    """
    emb, zmm, d, _ = _ordered_stack(z)
    return emb, zmm, d


def _per_patient_feats(
    emb: np.ndarray,
    z_mm: np.ndarray,
    eps_mm: float,
) -> dict[str, np.ndarray]:
    """Return 1D feature blocks (each length D). May contain nan vol if T'<2."""
    if emb.shape[0] == 0:
        d = emb.shape[1] if emb.ndim == 2 else 0
        nanv = np.full(d, np.nan, dtype=np.float64)
        return {
            "std": nanv.copy(),
            "p90_p10": nanv.copy(),
            "vol": nanv.copy(),
            "vol_mm": nanv.copy(),
        }
    d = emb.shape[1]
    t = emb.shape[0]
    std = np.std(emb, axis=0, ddof=0)
    p90p10 = np.percentile(emb, 90.0, axis=0) - np.percentile(emb, 10.0, axis=0)
    if t < 2:
        vol = np.full(d, np.nan, dtype=np.float64)
        vol_mm = np.full(d, np.nan, dtype=np.float64)
    else:
        dz = emb[1:] - emb[:-1]
        vol = np.mean(np.abs(dz), axis=0)
        dmm = np.abs(z_mm[1:] - z_mm[:-1])
        dmm = np.maximum(dmm, float(eps_mm))
        vol_mm = np.mean(np.abs(dz) / dmm[:, np.newaxis], axis=0)
    return {"std": std, "p90_p10": p90p10, "vol": vol, "vol_mm": vol_mm}


def _slice_positions_normalized(z_mm: np.ndarray, t: int) -> np.ndarray:
    """pos_norm[t] in [0, 1] for each ordered slice row (basal→apical style along z_mm)."""
    z = np.asarray(z_mm, dtype=np.float64).reshape(-1)
    if z.size != t:
        raise ValueError(f"z_mm length {z.size} != T {t}")
    finite = np.isfinite(z)
    if finite.any():
        lo = float(np.nanmin(z))
        hi = float(np.nanmax(z))
        if hi - lo > 1e-9:
            out = (z - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0)
            idx = np.arange(t, dtype=np.float64)
            out[~finite] = idx[~finite] / max(1.0, float(t - 1))
            return out
    if t <= 1:
        return np.zeros(max(0, t), dtype=np.float64)
    return np.linspace(0.0, 1.0, t, dtype=np.float64)


def _transition_rates_vol_mm(
    emb: np.ndarray,
    z_mm: np.ndarray,
    eps_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per edge t -> t+1: r[t, j] = |dz| / max(dmm, eps).
    pos_mid[t] = (pos_norm[t] + pos_norm[t+1]) / 2 for third labeling.
    """
    t, d = emb.shape
    if t < 2:
        return np.zeros((0, d), dtype=np.float64), np.zeros(0, dtype=np.float64)
    pos_norm = _slice_positions_normalized(z_mm, t)
    pos_mid = 0.5 * (pos_norm[:-1] + pos_norm[1:])
    dz = emb[1:] - emb[:-1]
    dmm = np.maximum(np.abs(z_mm[1:] - z_mm[:-1]), float(eps_mm))
    r = np.abs(dz) / dmm[:, np.newaxis]
    return r, pos_mid


def _regional_mean_vol_mm(r: np.ndarray, pos_mid: np.ndarray) -> np.ndarray:
    """Basal / mid / apical thirds on pos_mid; concat (3*D,)."""
    if r.ndim != 2:
        raise ValueError(r.shape)
    t_edge, d = r.shape
    if t_edge == 0:
        return np.full(3 * d, np.nan, dtype=np.float64)
    blocks: list[np.ndarray] = []
    masks = (
        pos_mid < (1.0 / 3.0),
        (pos_mid >= (1.0 / 3.0)) & (pos_mid < (2.0 / 3.0)),
        pos_mid >= (2.0 / 3.0),
    )
    for m in masks:
        if not np.any(m):
            blocks.append(np.full(d, np.nan, dtype=np.float64))
        else:
            blocks.append(np.mean(r[m], axis=0))
    return np.concatenate(blocks, axis=0)


def _p90_vol_mm_from_r(r: np.ndarray) -> np.ndarray:
    """Dim-wise 90th percentile of |dz|/dmm over transitions."""
    if r.shape[0] == 0:
        return np.full(r.shape[1], np.nan, dtype=np.float64)
    return np.percentile(r, 90.0, axis=0).astype(np.float64)


def trajectory_feature_blocks_extended(
    emb: np.ndarray,
    z_mm: np.ndarray,
    eps_mm: float,
) -> dict[str, np.ndarray]:
    """
    Patient-level blocks: std, p90_p10, vol, vol_mm (global mean |dz|/dmm),
    global_mean_vol_mm (same as vol_mm), p90_vol_mm, regional_vol_mm (3*D concat).
    """
    fd = _per_patient_feats(emb, z_mm, eps_mm)
    d = fd["vol_mm"].shape[0]
    r, pos_mid = _transition_rates_vol_mm(emb, z_mm, eps_mm)
    if emb.shape[0] < 2:
        reg = np.full(3 * d, np.nan, dtype=np.float64)
        p90_vm = np.full(d, np.nan, dtype=np.float64)
    else:
        reg = _regional_mean_vol_mm(r, pos_mid)
        p90_vm = _p90_vol_mm_from_r(r)
    out = {**fd, "global_mean_vol_mm": fd["vol_mm"].copy(), "p90_vol_mm": p90_vm, "regional_vol_mm": reg}
    return out


FEATURE_SET_SPECS: dict[str, tuple[str, ...]] = {
    "std_only": ("std",),
    "vol_only": ("vol",),
    "std_vol": ("std", "vol"),
    "p90_p10_only": ("p90_p10",),
    "p90_p10_vol": ("p90_p10", "vol"),
    "p90_p10_std": ("p90_p10", "std"),
    "vol_mm_only": ("vol_mm",),
    "global_mean_vol_mm_only": ("global_mean_vol_mm",),
    "p90_vol_mm_only": ("p90_vol_mm",),
    "regional_vol_mm_only": ("regional_vol_mm",),
    "std_vol_mm": ("std", "vol_mm"),
    "p90_p10_vol_mm": ("p90_p10", "vol_mm"),
    "p90_p10_global_mean_vol_mm": ("p90_p10", "global_mean_vol_mm"),
    "p90_p10_p90_vol_mm": ("p90_p10", "p90_vol_mm"),
    "p90_p10_regional_vol_mm": ("p90_p10", "regional_vol_mm"),
    "p90_p10_vol_mm_std": ("p90_p10", "vol_mm", "std"),
}


def _build_X_y(
    feats_by_pid: dict[str, dict[str, np.ndarray]],
    pid_order: list[str],
    y_by_pid: dict[str, int],
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate blocks; drop patients with nan in any used block."""
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    kept: list[str] = []
    for pid in pid_order:
        if pid not in y_by_pid or pid not in feats_by_pid:
            continue
        blocks = []
        ok = True
        for k in keys:
            v = feats_by_pid[pid][k]
            if np.any(~np.isfinite(v)):
                ok = False
                break
            blocks.append(v)
        if not ok:
            continue
        rows_X.append(np.concatenate(blocks, axis=0))
        rows_y.append(int(y_by_pid[pid]))
        kept.append(pid)
    if not rows_X:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64), []
    X = np.stack(rows_X, axis=0)
    y = np.asarray(rows_y, dtype=np.int64)
    return X, y, kept


def _one_fold_fit_score(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    c: float,
    random_state: int,
) -> tuple[float, float, float, float]:
    """train_roc, train_pr, test_roc, test_pr."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr_idx])
    Xte = scaler.transform(X[te_idx])
    clf = LogisticRegression(
        penalty="l2",
        C=float(c),
        max_iter=8000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(Xtr, y[tr_idx])
    s_tr = clf.predict_proba(Xtr)[:, 1]
    s_te = clf.predict_proba(Xte)[:, 1]
    return (
        float(roc_auc_score(y[tr_idx], s_tr)),
        float(average_precision_score(y[tr_idx], s_tr)),
        float(roc_auc_score(y[te_idx], s_te)),
        float(average_precision_score(y[te_idx], s_te)),
    )


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    """Same as repeated_stratified_shuffle_eval._summary_stats."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p2p5": float("nan"),
            "p97p5": float("nan"),
        }
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    q025, q975 = np.quantile(x, [0.025, 0.975])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p2p5": float(q025),
        "p97p5": float(q975),
    }


def _collect_repeats(
    X: np.ndarray,
    y: np.ndarray,
    pids: list[str],
    *,
    latent_source: str,
    pooling: str,
    model: str,
    cv_mode: str,
    k_folds: int,
    n_shuffle_splits: int,
    test_size: float,
    c: float,
    random_state: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Returns (repeat_rows, test_roc array, test_pr array)."""
    if X.shape[0] < 4 or np.unique(y).size < 2:
        return [], np.array([]), np.array([])
    pids_arr = np.asarray(pids, dtype=object)
    rows: list[dict] = []
    rocs: list[float] = []
    prs: list[float] = []
    rid = 0
    if cv_mode == "kfold":
        k = int(k_folds)
        if X.shape[0] < k:
            return [], np.array([]), np.array([])
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = skf.split(X, y)
    elif cv_mode == "shuffle":
        sss = StratifiedShuffleSplit(
            n_splits=int(n_shuffle_splits),
            test_size=test_size,
            random_state=random_state,
        )
        splits = sss.split(X, y)
    else:
        raise ValueError(f"cv_mode must be 'kfold' or 'shuffle', got {cv_mode!r}")

    for tr_idx, te_idx in splits:
        if tr_idx.size < 2 or te_idx.size < 2:
            continue
        tr_roc, tr_pr, te_roc, te_pr = _one_fold_fit_score(
            X, y, tr_idx=tr_idx, te_idx=te_idx, c=c, random_state=random_state
        )
        rocs.append(te_roc)
        prs.append(te_pr)
        te_pids = ";".join(str(pids_arr[i]) for i in te_idx)
        rows.append(
            {
                "latent_source": latent_source,
                "pooling": pooling,
                "model": model,
                "repeat_id": int(rid),
                "train_size": int(tr_idx.size),
                "test_size": int(te_idx.size),
                "roc_auc": te_roc,
                "pr_auc": te_pr,
                "train_roc_auc": tr_roc,
                "train_pr_auc": tr_pr,
                "test_patient_ids": te_pids,
            }
        )
        rid += 1
    return rows, np.asarray(rocs, dtype=np.float64), np.asarray(prs, dtype=np.float64)


def _order_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = ["latent_source", "model", "pooling"]
    cols = list(df.columns)
    ordered = [c for c in front if c in cols] + [c for c in cols if c not in front]
    return df[ordered]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument(
        "--split_dir",
        default=None,
        help="If set, only patients listed in train.csv|val.csv|test.csv (merged) are used.",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--cv_mode",
        choices=("kfold", "shuffle"),
        default="shuffle",
        help="shuffle = StratifiedShuffleSplit (default); kfold = StratifiedKFold.",
    )
    ap.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Used when --cv_mode kfold (same style as latent_kfold_cv.py, etc.).",
    )
    ap.add_argument(
        "--n_shuffle_splits",
        "--n_splits",
        type=int,
        default=100,
        dest="n_shuffle_splits",
        help="StratifiedShuffleSplit repeat count (--cv_mode shuffle). Alias: --n_splits.",
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Used only when --cv_mode shuffle.",
    )
    ap.add_argument("--c_values", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument(
        "--latent_source",
        default=None,
        help="Row key latent_source (default: basename of --npz_dir).",
    )
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE)
    ap.add_argument(
        "--eps_mm",
        type=float,
        default=1e-3,
        help="Floor for delta_mm (mm) to avoid divide-by-zero.",
    )
    ap.add_argument(
        "--recursive_npz",
        action="store_true",
        help="Find *.npz recursively under --npz_dir (default: only that directory, non-recursive).",
    )
    ap.add_argument(
        "--feature_sets",
        default="all",
        help="Comma-separated keys from FEATURE_SET_SPECS or 'all' (default).",
    )
    args = ap.parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clinical = ldm._merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    clinical_ix = clinical.set_index("patient_id", drop=False)
    y_map = clinical_ix["label"].astype(int).to_dict()

    allowed: Optional[set[str]] = None
    if args.split_dir:
        tr, va, te = ldm.load_split_patient_ids(args.split_dir)
        allowed = set(tr) | set(va) | set(te)
        print(f"[filter] split_dir union size = {len(allowed)}", file=sys.stderr)

    feats_by_pid: dict[str, dict[str, np.ndarray]] = {}
    pid_order: list[str] = []
    skipped: list[str] = []

    npz_paths = _discover_npz_paths(npz_dir, args.recursive_npz)
    print(
        f"[npz] dir={npz_dir!r}  n_npz_files={len(npz_paths)}  "
        f"recursive={bool(args.recursive_npz)}",
        file=sys.stderr,
    )
    if not npz_paths:
        print(
            "[npz] No .npz found. Use the folder that **directly** contains patient_*.npz, "
            "or pass --recursive_npz if they live in subfolders.",
            file=sys.stderr,
        )

    for path in npz_paths:
        stem = os.path.basename(path).replace(".npz", "")
        z = np.load(path, allow_pickle=True)
        pid = _npz_patient_id(z, stem)
        if allowed is not None and pid not in allowed:
            skipped.append(f"not_in_split:{pid}")
            continue
        if pid not in y_map:
            skipped.append(f"no_label:{pid}")
            continue
        try:
            emb, zmm, _d = _ordered_rows(z)
        except ValueError as e:
            skipped.append(f"bad_npz:{pid}:{e}")
            continue
        if emb.shape[0] < 2:
            skipped.append(f"too_few_slices:{pid}:{emb.shape[0]}")
            continue
        bl = trajectory_feature_blocks_extended(emb, zmm, args.eps_mm)
        if np.any(~np.isfinite(bl["vol_mm"])) or np.any(~np.isfinite(bl["p90_p10"])):
            skipped.append(f"nan_core:{pid}")
            continue
        feats_by_pid[pid] = bl
        pid_order.append(pid)

    pid_order = sorted(set(pid_order))
    y_vals = np.asarray([y_map[p] for p in pid_order], dtype=np.int64)
    if not pid_order and npz_paths:
        print(
            "[data] 0 patients after filters — check skipped_npz.txt (split CSV IDs vs NPZ stem/patient_id, labels_csv).",
            file=sys.stderr,
        )
    print(
        f"[data] patients with full trajectory feats: {len(pid_order)}  "
        f"(y=0: {int(np.sum(y_vals==0))}, y=1: {int(np.sum(y_vals==1))})",
        file=sys.stderr,
    )
    skip_path = os.path.join(out_dir, "skipped_npz.txt")
    with open(skip_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(skipped) if skipped else "(none)")

    latent_source = args.latent_source or os.path.basename(npz_dir.rstrip("/"))

    if args.feature_sets.strip().lower() == "all":
        fs_names = list(FEATURE_SET_SPECS.keys())
    else:
        fs_names = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
        for n in fs_names:
            if n not in FEATURE_SET_SPECS:
                print(f"Unknown feature set {n!r}. Choices: {list(FEATURE_SET_SPECS)}", file=sys.stderr)
                sys.exit(1)

    all_repeat_rows: list[dict] = []
    summary_rows: list[dict] = []
    cv_rows: list[dict] = []
    y_by_pid = {p: int(y_map[p]) for p in pid_order}

    for fs in fs_names:
        keys = FEATURE_SET_SPECS[fs]
        X, y, kept = _build_X_y(feats_by_pid, pid_order, y_by_pid, keys)
        n_pat = X.shape[0]
        dim_f = X.shape[1] if n_pat else 0
        for c in args.c_values:
            model = f"logistic_c{c}"
            rep_rows, rocs, prs = _collect_repeats(
                X,
                y,
                kept,
                latent_source=latent_source,
                pooling=fs,
                model=model,
                cv_mode=args.cv_mode,
                k_folds=args.k_folds,
                n_shuffle_splits=args.n_shuffle_splits,
                test_size=args.test_size,
                c=c,
                random_state=args.random_state,
            )
            all_repeat_rows.extend(rep_rows)
            roc_s = _summary_stats(rocs)
            pr_s = _summary_stats(prs)
            summary_rows.append(
                {
                    "latent_source": latent_source,
                    "pooling": fs,
                    "model": model,
                    **{f"roc_{k}": v for k, v in roc_s.items()},
                    **{f"pr_{k}": v for k, v in pr_s.items()},
                }
            )
            cv_rows.append(
                {
                    "cv_mode": args.cv_mode,
                    "latent_source": latent_source,
                    "pooling": fs,
                    "model": model,
                    "blocks": "+".join(keys),
                    "n_patients_used": n_pat,
                    "feature_dim": dim_f,
                    "C": float(c),
                    "k_folds": int(args.k_folds) if args.cv_mode == "kfold" else np.nan,
                    "n_shuffle_splits": int(args.n_shuffle_splits) if args.cv_mode == "shuffle" else np.nan,
                    "test_size": float(args.test_size) if args.cv_mode == "shuffle" else np.nan,
                    "roc_auc_mean": roc_s["mean"],
                    "roc_auc_std": roc_s["std"],
                    "roc_auc_p2_5": roc_s["p2p5"],
                    "pr_auc_mean": pr_s["mean"],
                    "pr_auc_std": pr_s["std"],
                    "pr_auc_p2_5": pr_s["p2p5"],
                    "n_valid_splits": int(rocs.size),
                }
            )

    df_rep = pd.DataFrame(all_repeat_rows)
    df_rep.to_csv(os.path.join(out_dir, "repeats.csv"), index=False)

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum = _order_summary_columns(df_sum)
    sum_path = os.path.join(out_dir, "summary.csv")
    df_sum.to_csv(sum_path, index=False)
    df_sum.to_csv(os.path.join(out_dir, "repeated_summary_all.csv"), index=False)
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    print(f"Wrote {out_dir}/repeats.csv", file=sys.stderr)
    print(f"Wrote {sum_path}", file=sys.stderr)
    print(f"Wrote {out_dir}/repeated_summary_all.csv", file=sys.stderr)
    print(f"Wrote {out_dir}/cv_summary.csv", file=sys.stderr)
    print(f"Wrote {skip_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
