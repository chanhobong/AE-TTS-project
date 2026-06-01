#!/usr/bin/env python3
"""
Single GIF: fixed 3×4 grid (12 orientation hypotheses × one slice each frame).

Every frame advances ``slice_index`` together: each cell repeats the ROI background for
that slice and draws the hypothesis overlay for **that same** slice_index. Panels move
“in sync” from first to last axial index.

Companion to:
  plot_spatial_orientation_sweep_panel.py   — hypotheses on ONE fixed slice
  make_spatial_coeff_roi_gif.py             — ONE hypothesis, axial sweep
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image


def _load_sweep_module():
    """plot_spatial_orientation_sweep_panel: heat decode + scipy zoom patterns."""
    here = Path(__file__).resolve().parent
    path = here / "plot_spatial_orientation_sweep_panel.py"
    spec = importlib.util.spec_from_file_location("_sweep_ori", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_overlay_helpers():
    here = Path(__file__).resolve().parent
    path = here / "plot_spatial_coeff_grid_overlay.py"
    spec = importlib.util.spec_from_file_location("_spatial_ovl", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slice_indices_in_npz(npz_path: str, *, respect_mask: bool) -> list[int]:
    z = np.load(npz_path, allow_pickle=True)
    sid = np.asarray(z["slice_indices"], dtype=np.int64).ravel()
    if respect_mask and "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1).astype(np.int64)
        if m.shape[0] == sid.shape[0]:
            sid = sid[m.astype(bool)]
    return sorted(int(x) for x in np.unique(sid))


def _window_slices(
    vol: np.ndarray,
    slice_idxs: list[int],
    *,
    axis_z: int,
    p_lo: float,
    p_hi: float,
) -> tuple[float, float]:
    parts: list[np.ndarray] = []
    for z in slice_idxs:
        if axis_z == 2:
            sl = vol[:, :, z]
        elif axis_z == 1:
            sl = vol[:, z, :]
        elif axis_z == 0:
            sl = vol[z, :, :]
        else:
            raise ValueError("axis_z must be 0..2")
        parts.append(sl.ravel())
    flat = np.concatenate(parts, axis=0)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(flat, [p_lo, p_hi])
    lo, hi = float(lo), float(hi)
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    return lo, hi


def _extract_vol_slice(vol: np.ndarray, axis_z: int, z: int) -> np.ndarray:
    if axis_z == 2:
        return vol[:, :, z]
    if axis_z == 1:
        return vol[:, z, :]
    return vol[z, :, :]


def _fig_to_rgb(fig: plt.Figure, *, dpi: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    dup = im.copy()
    im.close()
    buf.close()
    return dup


def main() -> None:
    ap = argparse.ArgumentParser(
        description="3×4 grid of orientation hypotheses × synchronized axial GIF (single file)."
    )
    ap.add_argument("--npz", required=True)
    ap.add_argument("--nii_roi", required=True)

    coef = ap.add_mutually_exclusive_group(required=True)
    coef.add_argument("--coef_npz", default="")
    coef.add_argument("--coef_csv", default="")
    coef.add_argument("--highlight_clusters", default="")

    ap.add_argument("--out_gif", required=True)

    ap.add_argument("--nii_axial_axis", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--slice_start", type=int, default=None)
    ap.add_argument("--slice_end", type=int, default=None)
    ap.add_argument("--respect_mask", action="store_true")

    ap.add_argument("--window_pct_lo", type=float, default=1.0)
    ap.add_argument("--window_pct_hi", type=float, default=99.0)

    ap.add_argument("--base_flip", choices=["none", "lr", "ud", "lr_ud"], default="none")
    ap.add_argument("--relu_weights", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.46)
    ap.add_argument("--cmap", default="hot")
    ap.add_argument("--vmin_pct", type=float, default=5.0)
    ap.add_argument("--vmin_floor", type=float, default=0.0)
    ap.add_argument("--upsample_order", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--frame_duration_ms", type=int, default=110)

    args = ap.parse_args()

    try:
        from scipy.ndimage import zoom
    except ImportError as e:
        raise SystemExit("pip install scipy") from e

    sw = _load_sweep_module()
    hlp = _load_overlay_helpers()

    combos: list[tuple[str, str, str]] = []
    for d in ["row_major", "col_major", "row_major_transpose_heat"]:
        for f in ["none", "lr", "ud", "lr_ud"]:
            combos.append((f"{d}|heat_{f}", d, f))

    npz_path = os.path.abspath(args.npz)
    znp = np.load(npz_path, allow_pickle=True)
    for k in ("cluster_ids", "slice_indices", "spatial_flat_idx"):
        if k not in znp.files:
            raise SystemExit(f"NPZ missing {k}")

    cid = np.asarray(znp["cluster_ids"], dtype=np.int64).ravel()
    sidx = np.asarray(znp["slice_indices"], dtype=np.int64).ravel()
    fid = np.asarray(znp["spatial_flat_idx"], dtype=np.int64).ravel()
    mask = np.ones(cid.shape[0], dtype=bool)
    if args.respect_mask and "mask" in znp.files:
        m = np.asarray(znp["mask"]).reshape(-1).astype(np.int64)
        if m.shape[0] == cid.shape[0]:
            mask = m.astype(bool)

    Hg, Wg = hlp._bottleneck_hw(znp)

    coef = hlp._load_coef(args.coef_npz or args.coef_csv or None)
    highlight = None
    hl = str(args.highlight_clusters or "").strip()
    if hl:
        highlight = [int(x) for x in hl.replace(",", " ").split() if x.strip()]

    cid_m = cid[mask]
    sidx_m = sidx[mask]
    fid_m = fid[mask]
    w_tok = hlp._token_weights(cid_m, coef, highlight)
    if args.relu_weights:
        w_tok = np.maximum(w_tok, 0.0)

    avail = _slice_indices_in_npz(npz_path, respect_mask=args.respect_mask)
    lo_i = args.slice_start if args.slice_start is not None else avail[0]
    hi_i = args.slice_end if args.slice_end is not None else avail[-1]
    slices = [s for s in avail if lo_i <= s <= hi_i]
    if not slices:
        raise SystemExit("No slices in requested range.")

    roi = nib.load(os.path.abspath(args.nii_roi)).get_fdata()
    n_ax = roi.shape[args.nii_axial_axis]
    for s in slices:
        if s < 0 or s >= n_ax:
            raise SystemExit(f"slice {s} out of NIfTI range (axis {args.nii_axial_axis}).")

    w_lo, w_hi = _window_slices(
        roi, slices, axis_z=args.nii_axial_axis, p_lo=args.window_pct_lo, p_hi=args.window_pct_hi
    )

    order = {"nearest": 0, "bilinear": 1}[args.upsample_order]

    # Pass 1: cache float32 overlays + backgrounds; compute shared color limits.
    all_bases: list[np.ndarray] = []
    all_his: list[list[np.ndarray]] = []
    pos_chunks: list[np.ndarray] = []
    max_hi = 0.0

    decode_key = {
        "row_major": "row_major",
        "col_major": "col_major",
        "row_major_transpose_heat": "row_major_transpose_heat",
    }
    flip_key = {"none": "none", "lr": "lr", "ud": "ud", "lr_ud": "lr_ud"}

    for z in slices:
        sl = _extract_vol_slice(roi, args.nii_axial_axis, z)
        base_raw = np.clip((sl.astype(np.float64) - w_lo) / (w_hi - w_lo + 1e-9), 0.0, 1.0)
        base_bw = np.asarray(sw._apply_flip2d(base_raw.astype(np.float64), args.base_flip), dtype=np.float64)
        Himg, Wimg = int(base_bw.shape[0]), int(base_bw.shape[1])
        zh, zw = float(Himg) / float(Hg), float(Wimg) / float(Wg)

        sel = sidx_m == int(z)
        fid_sel = fid_m[sel]
        w_sel = w_tok[sel]

        cell_his: list[np.ndarray] = []
        for _t, d_str, hf_str in combos:
            dec = decode_key[d_str]
            hf = flip_key[hf_str]
            hr = sw._build_heat_raw(fid_sel, w_sel, Hg, Wg, dec)
            hr = sw._apply_flip2d(hr, hf)
            hi = zoom(hr.astype(np.float64), (zh, zw), order=order).astype(np.float32)
            cell_his.append(hi)
            max_hi = max(max_hi, float(np.nanmax(hi)))
            nz = hi[hi > float(args.vmin_floor) + 1e-18]
            if nz.size > 0:
                flatnz = nz.ravel()
                step = max(1, flatnz.size // 5000)
                pos_chunks.append(flatnz[::step])

        all_bases.append(base_bw)
        all_his.append(cell_his)

    vmin = float(args.vmin_floor)
    if max_hi <= vmin + 1e-18:
        vmax = vmin + 1e-9
    else:
        vmax = float(max_hi)
        if pos_chunks:
            pooled = np.concatenate(pos_chunks)
            vmin = float(
                np.clip(np.percentile(pooled, float(args.vmin_pct)), float(args.vmin_floor), vmax - 1e-9)
            )
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    cmap_in = plt.get_cmap(args.cmap)

    nrows, ncols = 3, 4
    frames: list[Image.Image] = []
    stem = Path(npz_path).stem

    for zi, (z, base_bw, hil) in enumerate(zip(slices, all_bases, all_his), start=1):
        fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 10.8))
        fig.set_dpi(int(args.dpi))
        axi = axes.ravel()
        for ax, hi, (title, _d, _f) in zip(axi, hil, combos):
            ax.imshow(base_bw, cmap="gray", interpolation="nearest", aspect="equal")
            rgba = cmap_in(norm(np.asarray(hi, dtype=np.float64)))
            ax.imshow(rgba[..., :3], alpha=float(args.overlay_alpha))
            ax.set_title(title, fontsize=5.9, pad=2)
            ax.axis("off")

        fig.suptitle(
            f"{stem}\nSynced axial frame {zi}/{len(slices)}  slice_index={z}  bottleneck {Hg}x{Wg}",
            fontsize=11,
            y=1.005,
        )
        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.10, hspace=0.32)
        frames.append(_fig_to_rgb(fig, dpi=int(args.dpi)))

    out = os.path.abspath(args.out_gif)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.frame_duration_ms),
        loop=0,
        optimize=False,
    )
    for f in frames:
        f.close()
    print(f"Wrote {out} | {len(frames)} frames × {len(combos)} panels", file=sys.stderr)


if __name__ == "__main__":
    main()
