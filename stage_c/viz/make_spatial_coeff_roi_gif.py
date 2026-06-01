#!/usr/bin/env python3
"""
Axial **cine**: each frame uses one ``slice_index`` — the ROI background and the
coef/cluster heatmap both come from that slice (anatomy and overlay move together).

This is different from ``plot_spatial_orientation_sweep_panel.py``, which compares
decode/flip hypotheses on a **fixed** slice.

Delegates each frame to plot_spatial_coeff_grid_overlay.run() after saving a temporary 2D underlying .npy per slice.

Paths: ``utils/scripts/spatial_viz_paths.example.sh`` + ``SPATIAL_VIZ_COMMANDS.md``

Example::

  cd /Users/ch.b/Desktop/24-25/TSS/Code/TTS_Project/LDAE_TTS/AE_TTS
  source utils/scripts/spatial_viz_paths.example.sh
  export MPLCONFIGDIR="${MPLCONFIGDIR}"
  python3 utils/scripts/make_spatial_coeff_roi_gif.py \\
    --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \\
    --nii_roi "${ROI_TTS}/${PATIENT_ID}/${PATIENT_ID}_roi.nii.gz" \\
    --coef_npz "${COEF_HIST_TRAINVAL}" \\
    --out_gif "${FIG_OUT_DIR}/${PATIENT_ID}_axial.gif"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def _load_overlay_module():
    here = Path(__file__).resolve().parent
    path = here / "plot_spatial_coeff_grid_overlay.py"
    spec = importlib.util.spec_from_file_location("_spatial_coeff_overlay", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load overlay module from {path}")
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
    """Return global (low, high) percentiles across listed slices for stable brightness."""
    parts: list[np.ndarray] = []
    for z in slice_idxs:
        if axis_z == 2:
            sl = vol[:, :, z]
        elif axis_z == 1:
            sl = vol[:, z, :]
        elif axis_z == 0:
            sl = vol[z, :, :]
        else:
            raise ValueError("axis_z must be 0, 1, or 2")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="ROI + spatial coef overlay as animated GIF.")
    ap.add_argument("--npz", required=True, help="Per-patient Stage B spatial NPZ.")
    ap.add_argument("--nii_roi", required=True, help="3D ROI NIfTI (.nii or .nii.gz).")
    ap.add_argument("--out_gif", required=True)

    coef = ap.add_mutually_exclusive_group(required=True)
    coef.add_argument("--coef_npz", default="", help="1D logistic coef .npy, length K.")
    coef.add_argument("--coef_csv", default="", help="CSV first column: K coefs.")
    coef.add_argument(
        "--highlight_clusters",
        default="",
        help="Like plot_spatial_coeff_grid_overlay: comma/space-separated cluster ids.",
    )

    ap.add_argument(
        "--nii_axial_axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Which volume axis advances slices (default 2 → vol[:,:,z]).",
    )
    ap.add_argument(
        "--slice_start",
        type=int,
        default=None,
        help="Inclusive; default = min(slice_indices in NPZ).",
    )
    ap.add_argument(
        "--slice_end",
        type=int,
        default=None,
        help="Inclusive; default = max(slice_indices in NPZ).",
    )
    ap.add_argument(
        "--respect_mask",
        action="store_true",
        help="Infer slice indices from masked tokens only (match overlay semantics).",
    )
    ap.add_argument(
        "--window_pct_lo",
        type=float,
        default=1.0,
        help="Global HU window lower percentile across all GIF slices.",
    )
    ap.add_argument(
        "--window_pct_hi",
        type=float,
        default=99.0,
        help="Global HU window upper percentile across all GIF slices.",
    )

    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--cmap", type=str, default="hot")
    ap.add_argument("--vmin_pct", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--upsample_order", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--relu_weights", action="store_true")
    ap.add_argument("--frame_duration_ms", type=int, default=120, help="Delay per GIF frame in ms.")
    ap.add_argument(
        "--spatial_decode",
        choices=["row_major", "col_major", "row_major_transpose_heat"],
        default="row_major",
        help="Forwarded to plot_spatial_coeff_grid_overlay (grid index interpretation).",
    )
    ap.add_argument(
        "--heatmap_flip",
        choices=["none", "lr", "ud", "lr_ud"],
        default="none",
        help="Forwarded: flip low-res heat before upsampling.",
    )
    ap.add_argument("--underlying_gamma", type=float, default=1.0)
    ap.add_argument("--underlying_brightness_mult", type=float, default=1.0)
    ap.add_argument(
        "--slice_selector",
        choices=["slice_index", "position_norm"],
        default="slice_index",
        help="For GIF axial stack use slice_index. position_norm overlays same pooled heat on every frame (experimental).",
    )
    ap.add_argument("--slice_position_center", type=float, default=0.8)
    ap.add_argument("--slice_position_halfwidth", type=float, default=0.12)
    ap.add_argument("--contrib_top_positive_k", type=int, default=0)
    ap.add_argument(
        "--contrib_emb_alignment",
        choices=["none", "cosine_relu"],
        default="none",
    )
    ap.add_argument(
        "--emb_raw_norm_dampen",
        action="store_true",
        help="Forwarded to overlay run.",
    )

    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Need Pillow (`pip install pillow`) for GIF export.") from e

    avail = _slice_indices_in_npz(args.npz, respect_mask=args.respect_mask)
    if not avail:
        raise SystemExit("No slice indices found in NPZ.")

    lo_i = args.slice_start if args.slice_start is not None else avail[0]
    hi_i = args.slice_end if args.slice_end is not None else avail[-1]

    slices = [s for s in avail if lo_i <= s <= hi_i]
    if not slices:
        raise SystemExit(f"No slices in [{lo_i}, {hi_i}] intersect NPZ slices {avail[:5]}…")

    roi = nib.load(os.path.abspath(args.nii_roi)).get_fdata()
    n = roi.shape[args.nii_axial_axis]
    for s in slices:
        if s < 0 or s >= n:
            raise SystemExit(
                f"slice_index {s} out of range for NIfTI shape {roi.shape} along axis {args.nii_axial_axis}."
            )

    w_lo, w_hi = _window_slices(roi, slices, axis_z=args.nii_axial_axis, p_lo=args.window_pct_lo, p_hi=args.window_pct_hi)

    ovl = _load_overlay_module()
    tmpdir = tempfile.mkdtemp(prefix="spatial_gif_frames_")

    png_paths: list[str] = []
    try:
        for s in slices:
            if args.nii_axial_axis == 2:
                sl = roi[:, :, s]
            elif args.nii_axial_axis == 1:
                sl = roi[:, s, :]
            else:
                sl = roi[s, :, :]
            u = np.clip((sl - w_lo) / (w_hi - w_lo), 0.0, 1.0).astype(np.float32)
            u_path = os.path.join(tmpdir, f"under_{s:04d}.npy")
            np.save(u_path, u)

            out_png = os.path.join(tmpdir, f"frame_{s:04d}.png")
            ns = argparse.Namespace(
                npz=os.path.abspath(args.npz),
                slice_mode=str(args.slice_selector),
                slice_index=int(s),
                out_png=os.path.abspath(out_png),
                coef_npz=os.path.abspath(args.coef_npz) if args.coef_npz else "",
                coef_csv=os.path.abspath(args.coef_csv) if args.coef_csv else "",
                highlight_clusters=str(args.highlight_clusters or "").strip(),
                relu_weights=bool(args.relu_weights),
                underlying_npy=u_path,
                underlying_png="",
                fallback_canvas=256,
                respect_mask=bool(args.respect_mask),
                overlay_alpha=float(args.overlay_alpha),
                cmap=str(args.cmap),
                vmin_pct=float(args.vmin_pct),
                dpi=int(args.dpi),
                upsample_order=str(args.upsample_order),
                show_colorbar_gray=False,
                spatial_decode=str(args.spatial_decode),
                heatmap_flip=str(args.heatmap_flip),
                underlying_gamma=float(args.underlying_gamma),
                underlying_brightness_mult=float(args.underlying_brightness_mult),
                slice_position_center=float(args.slice_position_center),
                slice_position_halfwidth=float(args.slice_position_halfwidth),
                contrib_top_positive_k=int(args.contrib_top_positive_k),
                contrib_emb_alignment=str(args.contrib_emb_alignment),
                emb_raw_norm_dampen=bool(args.emb_raw_norm_dampen),
            )
            if not ns.coef_npz and not ns.coef_csv and not ns.highlight_clusters:
                raise SystemExit("Internal error: need coef or highlight.")
            ovl.run(ns)
            png_paths.append(os.path.abspath(out_png))

        imgs = [Image.open(p).convert("RGB") for p in png_paths]
        outp = os.path.abspath(args.out_gif)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        imgs[0].save(
            outp,
            save_all=True,
            append_images=imgs[1:],
            duration=int(args.frame_duration_ms),
            loop=0,
            optimize=False,
        )
        for im in imgs:
            im.close()
        print(f"Wrote {outp} ({len(png_paths)} frames)", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
