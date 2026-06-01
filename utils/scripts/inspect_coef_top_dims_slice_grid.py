#!/usr/bin/env python3
"""
Inspect slices that extremize top logistic std-block dimensions.

1) Read logistic_coef_importance_mean_vs_std_blocks.csv (or compatible): pick top-K latent_dim
   by abs_coef_std_block.
2) For each dim j, score each slice. Default score = (z[s,j] - mean_s z[:,j])**2 (contribution to
   patient-level std over slices, ddof=0 consistent with aggregate_std_only).
   Alternative: --rank_by raw_value uses z[s,j] globally.
3) Save CSV of top/bottom slice list (patient_id, slice_idx, score, y_true).
4) Correlate per-patient std on dim j with y_true (Pearson / Spearman).
5) Optional: load NIfTI per patient and draw 2×N grids (high vs low groups).
6) Optional: --overlay_segmentation — semi-transparent mask on top of the grayscale slice
   (auto-find under patient_dir/segmentation or patient_seg_csv / seg_path column).

Example
-------
  python inspect_coef_top_dims_slice_grid.py \\
    --coef_csv .../diagnostics/logistic_coef_importance_mean_vs_std_blocks.csv \\
    --npz_dir .../diff3dformer_stageB_plain_ae/patients \\
    --labels_csv /.../DatasetRaw/Combined_Labels.csv \\
    --dataset_raw_root /.../DatasetRaw \\
    --out_dir ./coef_slice_inspect

Combined_Labels.csv: columns ID (or patient_id), case (0=normal, 1=TTS), roi_file.
  Volumes resolved as: {raw_root}/TTS_RM_V2/{ID}/…  if case==1 (else normal_RM_V2).
  Folder may be {ID} or {ID_ageSex} from roi_file stem; then tries roi_file, LV ROI, directory scan.

paths.csv (optional): columns patient_id, volume_path — overrides auto layout.

Slice index i must match the order used when building embeddings (same as training dataloader).
If your NIfTI slice axis differs from encoding, set --slice_axis.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr, pointbiserialr
except ImportError:
    pearsonr = spearmanr = pointbiserialr = None  # type: ignore

try:
    from scipy.ndimage import zoom as ndi_zoom
except ImportError:
    ndi_zoom = None  # type: ignore

try:
    import nibabel as nib
except ImportError:
    nib = None


def _merge_clinical(paths: list[str]) -> pd.DataFrame:
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
        use_cols = [id_col, lab_col]
        if "roi_file" in df.columns:
            use_cols.append("roi_file")
        sub = df[use_cols].copy()
        sub.columns = ["patient_id", "raw_label"] + (["roi_file"] if "roi_file" in df.columns else [])
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
    if "roi_file" not in out.columns:
        out["roi_file"] = np.nan
    else:
        out["roi_file"] = out["roi_file"].replace("", np.nan)
    return out


def _patient_folder_keys(pid: str, roi: Optional[str]) -> list[str]:
    """
    Dataset folders may be {patient_id} (many TTS) or {patient_id}_{age}{sex} (many normals).
    The latter matches roi_file stem without _roi (e.g. AAP_50415783_61F_roi.nii.gz → AAP_50415783_61F).
    """
    keys: list[str] = [pid]
    if roi and str(roi).strip():
        stem = str(roi).strip().replace(".nii.gz", "").replace(".nii", "")
        if stem.endswith("_roi"):
            stem = stem[: -len("_roi")]
        if stem and stem not in keys:
            keys.append(stem)
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _find_roi_volume_in_folder(subdir: str, pid: str, roi: Optional[str]) -> Optional[str]:
    """Pick one ROI NIfTI inside subdir (patient folder)."""
    candidates: list[str] = []
    if roi and str(roi).strip():
        candidates.append(os.path.join(subdir, str(roi).strip()))
    candidates.append(os.path.join(subdir, f"{pid}_roi.nii.gz"))
    candidates.append(os.path.join(subdir, "heart_ventricle_left_roi.nii.gz"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    if not os.path.isdir(subdir):
        return None
    nii = sorted(
        f
        for f in os.listdir(subdir)
        if f.endswith(".nii.gz") or (f.endswith(".nii") and not f.endswith(".nii.gz"))
    )
    if len(nii) == 1:
        return os.path.join(subdir, nii[0])
    if len(nii) > 1:
        roi_cands = [f for f in nii if "roi" in f.lower()]
        lv = [
            f
            for f in roi_cands
            if "heart" in f.lower() and "ventricle" in f.lower() and "left" in f.lower()
        ]
        if len(lv) == 1:
            return os.path.join(subdir, lv[0])
        if len(lv) > 1:
            return os.path.join(subdir, sorted(lv)[0])
        if len(roi_cands) == 1:
            return os.path.join(subdir, roi_cands[0])
        if len(roi_cands) > 1:
            return os.path.join(subdir, sorted(roi_cands)[0])
    return None


def _build_path_map_dataset_raw(
    clinical: pd.DataFrame,
    raw_root: str,
    tts_subdir: str,
    normal_subdir: str,
) -> tuple[dict[str, str], list[str]]:
    """
    {raw_root}/{TTS|normal}_RM_V2/{folder_key}/…
    Tries folder_key = patient_id, then folder_key derived from roi_file (normal-style).
    If still missing, glob {branch}/{patient_id}_* for one matching directory.
    """
    raw_root = os.path.abspath(raw_root)
    path_map: dict[str, str] = {}
    missing: list[str] = []
    for _, row in clinical.iterrows():
        pid = str(row["patient_id"])
        y = int(row["y_true"])
        branch = tts_subdir if y == 1 else normal_subdir
        branch_root = os.path.join(raw_root, branch)
        roi = row["roi_file"] if pd.notna(row["roi_file"]) else None
        chosen: str | None = None
        for folder_key in _patient_folder_keys(pid, roi):
            subdir = os.path.join(branch_root, folder_key)
            chosen = _find_roi_volume_in_folder(subdir, pid, roi)
            if chosen is not None:
                break
        if chosen is None:
            pattern = os.path.join(branch_root, pid + "_*")
            for d in sorted(glob.glob(pattern)):
                if os.path.isdir(d):
                    chosen = _find_roi_volume_in_folder(d, pid, roi)
                    if chosen is not None:
                        break
        if chosen is not None:
            path_map[pid] = chosen
        else:
            missing.append(pid)
    return path_map, missing


def _load_top_dims(coef_csv: str, k: int) -> list[int]:
    df = pd.read_csv(coef_csv)
    need = {"latent_dim", "abs_coef_std_block"}
    if not need.issubset(df.columns):
        raise ValueError(f"{coef_csv} must have columns {need}, got {list(df.columns)}")
    df = df.sort_values("abs_coef_std_block", ascending=False).head(k)
    dims = [int(x) for x in df["latent_dim"].tolist()]
    return dims


def _slice_mask(z: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "mask" not in z.files:
        return None
    m = np.asarray(z["mask"]).reshape(-1)
    return m


def _score_slices_for_dim(
    emb: np.ndarray,
    j: int,
    rank_by: str,
) -> np.ndarray:
    """Per-slice scores, shape (N_slices,)."""
    col = emb[:, j].astype(np.float64)
    if rank_by == "raw_value":
        return col
    if rank_by == "dev_sq":
        mu = col.mean()
        return (col - mu) ** 2
    raise ValueError(f"rank_by must be raw_value or dev_sq, got {rank_by}")


def _collect_extremes_for_dim(
    npz_dir: str,
    clinical: pd.DataFrame,
    j: int,
    n_each: int,
    rank_by: str,
    respect_mask: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool all valid slices across patients, rank by score, take global top n_each and bottom n_each.

    Returns (extremes_table, patient_ids, sigma_j_per_patient, y_per_patient) for correlations.
    """
    paths = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    slice_rows: list[dict] = []
    pids_sigma: list[str] = []
    sigma_vals: list[float] = []
    y_vals: list[int] = []

    clin = clinical.set_index("patient_id")

    for fn in paths:
        pid = fn.replace(".npz", "")
        if pid not in clin.index:
            continue
        path = os.path.join(npz_dir, fn)
        z = np.load(path, allow_pickle=True)
        if "embeddings" not in z.files:
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)
        if emb.ndim != 2 or j >= emb.shape[1]:
            continue
        mask = _slice_mask(z)
        scores = _score_slices_for_dim(emb, j, rank_by=rank_by)
        n = emb.shape[0]
        valid = np.ones(n, dtype=bool)
        if respect_mask and mask is not None and mask.shape[0] == n:
            valid = mask.astype(bool)

        y = int(clin.loc[pid, "y_true"])
        sig = float(emb[:, j].std(ddof=0))
        pids_sigma.append(pid)
        sigma_vals.append(sig)
        y_vals.append(y)

        for sidx in range(n):
            if not valid[sidx]:
                continue
            slice_rows.append(
                {
                    "latent_dim": j,
                    "patient_id": pid,
                    "slice_idx": int(sidx),
                    "score": float(scores[sidx]),
                    "z_sj": float(emb[sidx, j]),
                    "y_true": y,
                }
            )

    pool = pd.DataFrame(slice_rows)
    if pool.empty:
        return pool, np.array([]), np.array([]), np.array([])

    pool = pool.sort_values("score", ascending=True, kind="mergesort")
    n_take = min(n_each, len(pool))
    low = pool.iloc[:n_take].copy()
    high = pool.iloc[-n_take:].iloc[::-1].copy()
    low["group"] = "low_score"
    high["group"] = "high_score"
    tab = pd.concat([high, low], ignore_index=True)
    return tab, np.array(pids_sigma), np.array(sigma_vals), np.array(y_vals)


def _correlation_table(
    sigma_per_patient: np.ndarray,
    y: np.ndarray,
    dim: int,
) -> dict:
    yv = np.asarray(y, dtype=int)
    sig = np.asarray(sigma_per_patient, dtype=np.float64)
    n0, n1 = int(np.sum(yv == 0)), int(np.sum(yv == 1))
    out: dict = {
        "latent_dim": dim,
        "n_patients": int(len(yv)),
        "y_n_class0": n0,
        "y_n_class1": n1,
        "note": "",
    }
    if sigma_per_patient.size < 3:
        out["note"] = "too_few_patients"
    elif n0 == 0 or n1 == 0:
        out["note"] = "single_class_in_aligned_npz_cohort"
    elif np.nanstd(sig) < 1e-14:
        out["note"] = "sigma_constant_across_patients"
    if out["note"]:
        out["pearson_r"] = out["pearson_p"] = np.nan
        out["spearman_r"] = out["spearman_p"] = np.nan
        out["pointbiserial_r"] = out["pointbiserial_p"] = np.nan
        return out
    if pearsonr is None:
        raise RuntimeError("scipy required for correlation; pip install scipy")
    pr = pearsonr(sig, yv.astype(np.float64))
    sr = spearmanr(sig, yv)
    pbr = pointbiserialr(yv.astype(np.float64), sig)
    out["pearson_r"] = float(pr.statistic) if np.isfinite(pr.statistic) else np.nan
    out["pearson_p"] = float(pr.pvalue) if np.isfinite(pr.pvalue) else np.nan
    out["spearman_r"] = float(sr.statistic) if np.isfinite(sr.statistic) else np.nan
    out["spearman_p"] = float(sr.pvalue) if np.isfinite(sr.pvalue) else np.nan
    out["pointbiserial_r"] = float(pbr.correlation) if np.isfinite(pbr.correlation) else np.nan
    out["pointbiserial_p"] = float(pbr.pvalue) if np.isfinite(pbr.pvalue) else np.nan
    if not np.isfinite(out["pearson_r"]) and not out["note"]:
        out["note"] = "correlation_undefined"
    return out


def _rank_seg_paths(
    paths: list[str],
    reference_volume_path: Optional[str],
) -> Optional[str]:
    """
    Prefer a label/mask that is NOT the same file as the displayed ROI crop, then LV-related names;
    avoid alphabetically-first unrelated organs (e.g. aorta).
    """
    if reference_volume_path:
        ref_abs = os.path.abspath(reference_volume_path)
        paths = [p for p in paths if os.path.abspath(p) != ref_abs]
    if not paths:
        return None

    ref_bn = os.path.basename(reference_volume_path).lower() if reference_volume_path else ""
    lower = [os.path.basename(p).lower() for p in paths]

    def pick(predicate: Callable[[str], bool]) -> Optional[str]:
        hit = [paths[i] for i in range(len(paths)) if predicate(lower[i])]
        return sorted(hit)[0] if hit else None

    if "heart_ventricle_left" in ref_bn:
        p = pick(lambda s: "heart_ventricle_left" in s)
        if p:
            return p
    p = pick(lambda s: "heart_ventricle_left" in s)
    if p:
        return p
    p = pick(lambda s: "left_ventricle" in s or "lv_endo" in s)
    if p:
        return p
    p = pick(lambda s: "myocardium" in s)
    if p:
        return p
    non_ao = [paths[i] for i in range(len(paths)) if "aorta" not in lower[i]]
    if non_ao:
        return sorted(non_ao)[0]
    return sorted(paths)[0]


def _find_seg_nifti(
    patient_dir: str,
    seg_subdir: str,
    reference_volume_path: Optional[str] = None,
) -> Optional[str]:
    """Pick one segmentation / label NIfTI; uses displayed ROI path as a hint."""
    candidates: list[str] = []
    sub = os.path.join(patient_dir, seg_subdir) if seg_subdir else ""
    if sub and os.path.isdir(sub):
        candidates.extend(glob.glob(os.path.join(sub, "*.nii.gz")))
        candidates.extend(glob.glob(os.path.join(sub, "*.nii")))
    for pat in (
        "*heart*ventricle*left*.nii.gz",
        "*seg*.nii.gz",
        "*mask*.nii.gz",
        "*label*.nii.gz",
    ):
        candidates.extend(glob.glob(os.path.join(patient_dir, pat)))
    candidates = sorted(set(candidates))
    if not candidates:
        return None
    return _rank_seg_paths(candidates, reference_volume_path)


def _build_seg_path_map(
    path_map: dict[str, str],
    *,
    patient_seg_csv: Optional[str],
    volume_csv_has_seg_col: Optional[pd.DataFrame],
    seg_subdir: str,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if patient_seg_csv:
        sdf = pd.read_csv(patient_seg_csv)
        if "patient_id" not in sdf.columns or "seg_path" not in sdf.columns:
            raise ValueError("patient_seg_csv needs columns: patient_id, seg_path")
        for _, row in sdf.iterrows():
            p = str(row["patient_id"])
            sp = str(row["seg_path"])
            if p in path_map and os.path.isfile(sp):
                out[p] = sp
        return out
    if volume_csv_has_seg_col is not None and "seg_path" in volume_csv_has_seg_col.columns:
        for _, row in volume_csv_has_seg_col.iterrows():
            p = str(row["patient_id"])
            sp = row.get("seg_path")
            if pd.isna(sp) or not str(sp).strip():
                continue
            sp = str(sp).strip()
            if p in path_map and os.path.isfile(sp):
                out[p] = sp
        return out
    for pid, vpath in path_map.items():
        pd_dir = os.path.dirname(os.path.abspath(vpath))
        hit = _find_seg_nifti(pd_dir, seg_subdir, reference_volume_path=vpath)
        if hit:
            out[pid] = hit
    return out


def _load_slice_2d(vol_path: str, slice_idx: int, axis: int) -> np.ndarray:
    if nib is None:
        raise RuntimeError("nibabel required for NIfTI; pip install nibabel")
    img = nib.load(vol_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if axis == 0:
        sl = data[slice_idx, :, :]
    elif axis == 1:
        sl = data[:, slice_idx, :]
    elif axis == 2:
        sl = data[:, :, slice_idx]
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")
    return sl


def _plot_grid_for_dim(
    dim: int,
    high_rows: pd.DataFrame,
    low_rows: pd.DataFrame,
    path_map: dict[str, str],
    slice_axis: int,
    out_png: str,
    rank_by: str,
    n_per_row: int,
    seg_path_map: Optional[dict[str, str]] = None,
    seg_overlay_alpha: float = 0.32,
    seg_axis: Optional[int] = None,
    seg_rgb: tuple[float, float, float] = (1.0, 0.35, 0.35),
) -> None:
    n_h = min(n_per_row, len(high_rows))
    n_l = min(n_per_row, len(low_rows))
    fig, axes = plt.subplots(2, max(n_h, n_l, 1), figsize=(2.8 * max(n_h, n_l, 1), 5.6))
    if max(n_h, n_l) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    title = (
        f"dim {dim} — top row: highest {rank_by} (n={n_h}); "
        f"bottom row: lowest {rank_by} (n={n_l})"
    )
    fig.suptitle(title, fontsize=11)

    for row_idx, (rows, n) in enumerate([(high_rows, n_h), (low_rows, n_l)]):
        for c in range(max(n_h, n_l)):
            ax = axes[row_idx, c]
            if c >= n or rows is None or c >= len(rows):
                ax.axis("off")
                continue
            r = rows.iloc[c]
            pid = str(r["patient_id"])
            sidx = int(r["slice_idx"])
            vol_path = path_map.get(pid)
            if not vol_path or not os.path.isfile(vol_path):
                ax.text(0.5, 0.5, f"missing volume\n{pid}", ha="center", va="center")
                ax.axis("off")
                continue
            try:
                sl = _load_slice_2d(vol_path, sidx, slice_axis)
            except Exception as exc:
                ax.text(0.5, 0.5, f"load error\n{exc}", ha="center", va="center", fontsize=7)
                ax.axis("off")
                continue
            ax.imshow(sl.T, cmap="gray", origin="lower", vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99))
            s_axis = slice_axis if seg_axis is None else int(seg_axis)
            if seg_path_map:
                spath = seg_path_map.get(pid)
                if (
                    spath
                    and os.path.isfile(spath)
                    and os.path.abspath(spath) != os.path.abspath(vol_path)
                ):
                    try:
                        seg_sl = _load_slice_2d(spath, sidx, s_axis)
                        disp = sl.T
                        seg_disp = np.asarray(seg_sl.T, dtype=np.float64)
                        if seg_disp.shape != disp.shape and ndi_zoom is not None:
                            zf = (
                                disp.shape[0] / max(seg_disp.shape[0], 1),
                                disp.shape[1] / max(seg_disp.shape[1], 1),
                            )
                            seg_disp = ndi_zoom(seg_disp, zf, order=0)
                        if seg_disp.shape != disp.shape:
                            pass
                        else:
                            m = seg_disp > 0.5
                            if np.any(seg_disp != 0) and not np.any(m):
                                m = seg_disp != 0
                            rgba = np.zeros((*disp.shape, 4), dtype=np.float64)
                            rgba[m, 0] = seg_rgb[0]
                            rgba[m, 1] = seg_rgb[1]
                            rgba[m, 2] = seg_rgb[2]
                            rgba[m, 3] = float(seg_overlay_alpha)
                            ax.imshow(rgba, origin="lower", interpolation="nearest")
                    except Exception:
                        pass
            ax.set_title(
                f"{pid}\nslice={sidx} y={int(r['y_true'])}\nscore={float(r['score']):.4g}",
                fontsize=7,
            )
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Top std-block coef dims: slice extremes + grids + correlations")
    ap.add_argument("--coef_csv", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_csv", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k_dims", type=int, default=3)
    ap.add_argument("--n_extremes", type=int, default=5, help="Per dim, per patient: take this many high/low (then pooled across patients in table)")
    ap.add_argument(
        "--rank_by",
        choices=("dev_sq", "raw_value"),
        default="dev_sq",
        help="dev_sq=(z-mean)^2 within patient (drives std); raw_value=z[s,j]",
    )
    ap.add_argument("--respect_mask", action="store_true", help="If NPZ has mask, ignore mask==0 slices")
    ap.add_argument(
        "--patient_volume_csv",
        default=None,
        help="CSV with patient_id, volume_path for NIfTI visualization",
    )
    ap.add_argument(
        "--dataset_raw_root",
        default=None,
        help="e.g. .../DatasetRaw — builds TTS_RM_V2/{ID}/{roi_file} vs normal_RM_V2/... from labels",
    )
    ap.add_argument(
        "--tts_subdir",
        default="TTS_RM_V2",
        help="Subfolder under dataset_raw_root for case==1",
    )
    ap.add_argument(
        "--normal_subdir",
        default="normal_RM_V2",
        help="Subfolder under dataset_raw_root for case==0",
    )
    ap.add_argument("--slice_axis", type=int, default=2, help="Axis along which slice_idx indexes (0,1,2)")
    ap.add_argument(
        "--overlay_segmentation",
        action="store_true",
        help="Draw semi-transparent seg on top of each slice (needs seg path or auto-find)",
    )
    ap.add_argument(
        "--seg_overlay_alpha",
        type=float,
        default=0.32,
        help="Seg overlay opacity (0=hidden, 1=solid)",
    )
    ap.add_argument(
        "--seg_subdir",
        default="segmentation",
        help="Under patient folder (dirname of ROI), look here for *.nii / *.nii.gz; then fallback globs",
    )
    ap.add_argument(
        "--seg_axis",
        type=int,
        default=-1,
        help="Slice axis for seg volume (-1 = same as --slice_axis)",
    )
    ap.add_argument(
        "--patient_seg_csv",
        default=None,
        help="CSV with patient_id, seg_path (optional; overrides auto-find)",
    )
    args = ap.parse_args()

    if args.patient_volume_csv and args.dataset_raw_root:
        print("Use only one of --patient_volume_csv or --dataset_raw_root", file=sys.stderr)
        sys.exit(1)

    clinical = _merge_clinical([os.path.abspath(p) for p in args.labels_csv])
    dims = _load_top_dims(os.path.abspath(args.coef_csv), args.top_k_dims)
    os.makedirs(args.out_dir, exist_ok=True)

    path_map: dict[str, str] = {}
    vdf_for_seg: Optional[pd.DataFrame] = None
    miss_ids: list[str] = []
    if args.patient_volume_csv:
        vdf = pd.read_csv(args.patient_volume_csv)
        vdf_for_seg = vdf
        if "patient_id" not in vdf.columns or "volume_path" not in vdf.columns:
            print("patient_volume_csv needs columns: patient_id, volume_path", file=sys.stderr)
            sys.exit(1)
        for _, row in vdf.iterrows():
            path_map[str(row["patient_id"])] = str(row["volume_path"])
    elif args.dataset_raw_root:
        path_map, miss_ids = _build_path_map_dataset_raw(
            clinical,
            os.path.abspath(args.dataset_raw_root),
            args.tts_subdir,
            args.normal_subdir,
        )

    seg_path_map: Optional[dict[str, str]] = None
    if args.overlay_segmentation:
        seg_path_map = _build_seg_path_map(
            path_map,
            patient_seg_csv=os.path.abspath(args.patient_seg_csv)
            if args.patient_seg_csv
            else None,
            volume_csv_has_seg_col=vdf_for_seg,
            seg_subdir=args.seg_subdir,
        )
        n_hit = len(seg_path_map)
        print(
            f"Segmentation overlay: {n_hit}/{len(path_map)} patients with a seg volume "
            f"(alpha={args.seg_overlay_alpha})"
        )

    if args.dataset_raw_root and path_map:
        resolved_csv = os.path.join(args.out_dir, "resolved_volume_paths.csv")
        res_rows = [{"patient_id": k, "volume_path": v} for k, v in sorted(path_map.items())]
        rdf = pd.DataFrame(res_rows)
        if seg_path_map:
            rdf["seg_path"] = rdf["patient_id"].map(lambda p: seg_path_map.get(p, ""))
        rdf.to_csv(resolved_csv, index=False)
        print(
            f"Auto volume paths: {len(path_map)} resolved, {len(miss_ids)} missing → {resolved_csv}"
        )
        if miss_ids and len(miss_ids) <= 20:
            print(f"  missing IDs: {miss_ids}")
        elif miss_ids:
            print(f"  missing IDs (first 20): {miss_ids[:20]} …")

    seg_axis_plot = None if args.seg_axis < 0 else args.seg_axis

    corr_rows: list[dict] = []
    for j in dims:
        tab, pids, sigma, y = _collect_extremes_for_dim(
            os.path.abspath(args.npz_dir),
            clinical,
            j,
            n_each=args.n_extremes,
            rank_by=args.rank_by,
            respect_mask=args.respect_mask,
        )
        tpath = os.path.join(args.out_dir, f"slice_extremes_dim_{j}.csv")
        tab.to_csv(tpath, index=False)
        print(f"[dim {j}] wrote {tpath} ({len(tab)} rows)")

        corr_rows.append(_correlation_table(sigma, y, j))

        if path_map:
            if tab.empty:
                continue
            hi = tab[tab["group"] == "high_score"]
            lo = tab[tab["group"] == "low_score"]
            png = os.path.join(args.out_dir, f"grid_dim_{j}_high_vs_low.png")
            _plot_grid_for_dim(
                j,
                hi,
                lo,
                path_map,
                args.slice_axis,
                png,
                args.rank_by,
                args.n_extremes,
                seg_path_map=seg_path_map if args.overlay_segmentation else None,
                seg_overlay_alpha=args.seg_overlay_alpha,
                seg_axis=seg_axis_plot,
            )
            print(f"[dim {j}] figure {png}")

    cdf = pd.DataFrame(corr_rows)
    cpath = os.path.join(args.out_dir, "correlation_sigma_vs_y.csv")
    cdf.to_csv(cpath, index=False)
    print(f"Wrote {cpath}")
    print(cdf.to_string(index=False))


if __name__ == "__main__":
    main()
