#!/usr/bin/env python3
"""
Montage spatial bottleneck overlay variants for manual alignment checks.

Covers typical ambiguity sources:
  - flat_idx interpreted as row-major (C-order) vs column-major vs row-major then heat.T
  - fliplr / flipud / both on low-res heat before zoom
  - optional flips on underlying 2D (NIfTI vs viewer convention)

Uses the same token weighting as plot_spatial_coeff_grid_overlay.py.

This is NOT an **axial cine**: it does not sweep ``slice_index`` through the volume.
For ROI + coef overlay that advances with each axial slice:
  • one hypothesis → ``make_spatial_coeff_roi_gif.py``
  • all 12 hypotheses in a 3×4 grid advancing together →
    ``make_spatial_12panel_orientation_axial_cine_gif.py``

Optional animated GIF (--out_gif) cycles the same overlays frame-by-frame; add
``--gif_cycle_base_flip`` to append all four underlying-flip hypotheses (48 frames).

Example::

  python3 utils/scripts/plot_spatial_orientation_sweep_panel.py \\
    --npz patient.npz --slice_index 0 --coef_npz coef.npy --underlying_npy sl.npy \\
    --out_gif figures/sweep.gif --gif_frame_ms 400
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Iterable, Literal

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import importlib.util


def _load_overlay_helpers():
    """Import _bottleneck_hw, _load_coef, _load_underlying, _token_weights from sibling script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "plot_spatial_coeff_grid_overlay.py")
    spec = importlib.util.spec_from_file_location("_spatial_overlay_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DecodeMode = Literal["row_major", "col_major", "row_major_transpose_heat"]
FlipHeat = Literal["none", "lr", "ud", "lr_ud"]
FlipBase = Literal["none", "lr", "ud", "lr_ud"]


def _flat_to_rc(fl: int, Hg: int, Wg: int, decode: DecodeMode) -> tuple[int, int] | None:
    ngrid = int(Hg * Wg)
    if fl < 0 or fl >= ngrid:
        return None
    if decode == "row_major":
        return int(fl // Wg), int(fl % Wg)
    if decode == "col_major":
        return int(fl % Hg), int(fl // Hg)
    if decode == "row_major_transpose_heat":
        return int(fl // Wg), int(fl % Wg)
    raise ValueError(decode)


def _apply_flip2d(arr: np.ndarray, which: FlipHeat | FlipBase) -> np.ndarray:
    out = arr
    if which in ("lr", "lr_ud"):
        out = np.fliplr(out)
    if which in ("ud", "lr_ud"):
        out = np.flipud(out)
    return out


def _build_heat_raw(
    fid_sel: np.ndarray,
    w_sel: np.ndarray,
    Hg: int,
    Wg: int,
    decode: DecodeMode,
) -> np.ndarray:
    heat = np.zeros((Hg, Wg), dtype=np.float64)
    ngrid = Hg * Wg
    for fl, ww in zip(fid_sel, w_sel):
        fl = int(fl)
        if fl < 0 or fl >= ngrid:
            continue
        rc = _flat_to_rc(fl, Hg, Wg, decode)
        if rc is None:
            continue
        r, c = rc
        heat[r, c] += float(ww)
    if decode == "row_major_transpose_heat":
        heat = heat.T
    return heat


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(fig.dpi), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    dup = im.copy()
    im.close()
    buf.close()
    return dup


def run(args: argparse.Namespace) -> None:
    hlp = _load_overlay_helpers()
    npz_path = os.path.abspath(args.npz)
    z = np.load(npz_path, allow_pickle=True)
    for k in ("cluster_ids", "slice_indices", "spatial_flat_idx"):
        if k not in z.files:
            raise SystemExit(f"NPZ missing {k}: {npz_path}")

    cid = np.asarray(z["cluster_ids"], dtype=np.int64).ravel()
    sidx = np.asarray(z["slice_indices"], dtype=np.int64).ravel()
    fid = np.asarray(z["spatial_flat_idx"], dtype=np.int64).ravel()
    if cid.shape[0] != sidx.shape[0] or cid.shape[0] != fid.shape[0]:
        raise SystemExit("NPZ arrays length mismatch.")

    mask = np.ones(cid.shape[0], dtype=bool)
    if args.respect_mask and "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1).astype(np.int64)
        if m.shape[0] == cid.shape[0]:
            mask = m.astype(bool)

    Hg, Wg = hlp._bottleneck_hw(z)
    ngrid = int(Hg * Wg)

    coef = hlp._load_coef(args.coef_npz or args.coef_csv)
    highlight = None
    if args.highlight_clusters.strip():
        highlight = [int(x) for x in args.highlight_clusters.replace(",", " ").split() if x.strip()]

    cid_m = cid[mask]
    sidx_m = sidx[mask]
    fid_m = fid[mask]
    w_tok = hlp._token_weights(cid_m, coef, highlight)

    sel = sidx_m == int(args.slice_index)
    if not np.any(sel):
        raise SystemExit(f"No tokens for slice_index={args.slice_index}")

    fid_sel = fid_m[sel]
    w_sel = w_tok[sel]

    if args.relu_weights:
        w_sel = np.maximum(w_sel, 0.0)

    # Underlying base
    if args.underlying_png or args.underlying_npy:
        base0 = hlp._load_underlying(str(args.underlying_png or args.underlying_npy))
    else:
        base0 = np.full((256, 256), 0.35, dtype=np.float64)

    try:
        from scipy.ndimage import zoom
    except ImportError as e:
        raise SystemExit("pip install scipy (ndimage.zoom)") from e

    order = {"nearest": 0, "bilinear": 1}[args.upsample_order]

    decodes: list[DecodeMode] = ["row_major", "col_major", "row_major_transpose_heat"]
    flips: list[FlipHeat] = ["none", "lr", "ud", "lr_ud"]

    combos: list[tuple[str, DecodeMode, FlipHeat]] = []
    for d in decodes:
        for f in flips:
            combos.append((f"{d}|heat_{f}", d, f))

    def render_one(
        dec: DecodeMode,
        hf: FlipHeat,
        base_hw: tuple[int, int],
        base_flip_setting: FlipBase,
    ) -> tuple[np.ndarray, np.ndarray]:
        heat_raw = _build_heat_raw(fid_sel, w_sel, Hg, Wg, dec)
        heat_raw = _apply_flip2d(heat_raw, hf)
        Hi, Wi = heat_raw.shape
        Himg, Wimg = base_hw
        zh, zw = float(Himg) / float(Hi), float(Wimg) / float(Wi)
        hi = zoom(heat_raw, (zh, zw), order=order)
        bf = np.asarray(base0, dtype=np.float64).copy()
        bf = _apply_flip2d(bf, base_flip_setting)
        return bf, hi

    stem = npz_path.split(os.sep)[-1].replace(".npz", "")
    want_png = bool(str(getattr(args, "out_png", "")).strip())
    want_gif = bool(str(getattr(args, "out_gif", "")).strip())
    gif_cycle_bf = bool(getattr(args, "gif_cycle_base_flip", False))

    # Montage uses current --base_flip only; GIF may sweep base flips.
    bases_his_mosaic = [
        render_one(d, f, base0.shape[:2], args.base_flip) for _, d, f in combos
    ]

    if want_gif:
        gif_rows: list[tuple[str, FlipBase, tuple[str, DecodeMode, FlipHeat], tuple[np.ndarray, np.ndarray]]] = []
        if gif_cycle_bf:
            for bbf in ("none", "lr", "ud", "lr_ud"):
                for comb in combos:
                    t, d, hf = comb
                    bf_hi = render_one(d, hf, base0.shape[:2], bbf)
                    gif_rows.append((f"baseflip={bbf}\n{t}", bbf, comb, bf_hi))
        else:
            for comb in combos:
                t, d, hf = comb
                bf_hi = render_one(d, hf, base0.shape[:2], args.base_flip)
                gif_rows.append((t, args.base_flip, comb, bf_hi))
        gif_his_only = [row[3][1] for row in gif_rows]
    else:
        gif_rows = []
        gif_his_only = []

    # Shared color scale: union of mosaic (if saved) + GIF frames (if saved)
    his_for_scale: list[np.ndarray] = []
    if want_png:
        his_for_scale.extend(hi for _, hi in bases_his_mosaic)
    if want_gif:
        his_for_scale.extend(gif_his_only)

    vmax = max(float(np.max(hi)) for hi in his_for_scale if np.any(np.isfinite(hi)))
    vmin = float(args.vmin_floor)
    if vmax > vmin + 1e-18:
        pos_all = np.concatenate([hi[np.isfinite(hi) & (hi > 1e-18)].ravel() for hi in his_for_scale])
        if pos_all.size > 0:
            vmin = float(np.clip(np.percentile(pos_all, float(args.vmin_pct)), float(args.vmin_floor), vmax - 1e-9))
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    cmap = plt.get_cmap(args.cmap)

    n = len(combos)
    title0 = (
        f"{stem} slice={args.slice_index} bottleneck={Hg}x{Wg}\n"
        f"montage base_flip={args.base_flip}"
    )

    if want_png:
        ncols = int(args.cols)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(float(args.fig_width), float(args.fig_height)))
        axes_flat: Iterable = np.atleast_1d(axes).ravel()

        for ax, (title, _dec, _hf), (bf, hi) in zip(axes_flat, combos, bases_his_mosaic):
            ax.imshow(bf, cmap="gray", interpolation="nearest", aspect="equal")
            rgba = cmap(norm(hi.astype(np.float64)))
            ax.imshow(rgba[..., :3], alpha=float(args.overlay_alpha))
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        for ax in axes_flat[len(combos) :]:
            ax.axis("off")

        fig.suptitle(title0 + f"\nvmin={vmin:.4g} vmax={max(vmax, vmin):.4g}", fontsize=9, y=1.02)
        fig.tight_layout()
        outp = os.path.abspath(args.out_png)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        fig.savefig(outp, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outp} ({n} variants)", file=sys.stderr)

    if want_gif:
        gif_fig_edge = float(getattr(args, "gif_fig_edge", 5.5))
        gif_dpi = int(getattr(args, "gif_dpi", 120))
        pil_frames: list[Image.Image] = []
        for panel_title, _bbf, _comb, (bf, hi) in gif_rows:
            fig, ax = plt.subplots(figsize=(gif_fig_edge, gif_fig_edge))
            fig.set_dpi(gif_dpi)
            ax.imshow(bf, cmap="gray", interpolation="nearest", aspect="equal")
            rgba = cmap(norm(hi.astype(np.float64)))
            ax.imshow(rgba[..., :3], alpha=float(args.overlay_alpha))
            ax.set_title(panel_title, fontsize=8)
            ax.axis("off")
            top = (
                f"{stem}  z={args.slice_index}  {Hg}x{Wg} bottleneck\n"
                f"GIF vmin={vmin:.4g} vmax={max(vmax, vmin):.4g}"
            )
            fig.suptitle(top, fontsize=9, y=0.98)
            pil_frames.append(_fig_to_image(fig))

        gif_path = os.path.abspath(args.out_gif)
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        dur = int(getattr(args, "gif_frame_ms", 400))
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=dur,
            loop=0,
            optimize=False,
        )
        for im in pil_frames:
            im.close()
        print(f"Wrote {gif_path} ({len(pil_frames)} frames)", file=sys.stderr)


def _run_bundle_base_flip(args: argparse.Namespace) -> None:
    """Write the same orientation grid once per underlying-flip hypothesis."""
    out_dir = os.path.abspath(os.path.expanduser(args.bundle_base_flip_dir))
    os.makedirs(out_dir, exist_ok=True)
    stem = (
        os.path.splitext(os.path.basename(args.npz))[0]
        + f"_slice{int(args.slice_index)}_orient_sweep_bundle"
    )
    original_out = args.out_png
    for bf in ("none", "lr", "ud", "lr_ud"):
        args.base_flip = bf  # noqa: PLW2901 intentional sweep
        args.out_png = os.path.join(out_dir, f"{stem}_baseflip_{bf}.png")
        run(args)
    args.base_flip = "none"
    args.out_png = original_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial grid overlay orientation sweep (multi-panel PNG).")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--slice_index", type=int, required=True)
    ap.add_argument(
        "--out_png",
        default="",
        help="Output montage PNG. Optional if --bundle_base_flip_dir is set (placeholder allowed).",
    )

    cg = ap.add_mutually_exclusive_group()
    cg.add_argument("--coef_npz", default="", help="K-vector .npy")
    cg.add_argument("--coef_csv", default="", help="First column coefs.")

    ap.add_argument("--highlight_clusters", default="")
    ap.add_argument("--relu_weights", action="store_true")

    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--underlying_npy", default="")
    bg.add_argument("--underlying_png", default="")

    ap.add_argument(
        "--base_flip",
        choices=["none", "lr", "ud", "lr_ud"],
        default="none",
        help="Apply to grayscale underlying only (simulate radiological LR).",
    )
    ap.add_argument("--respect_mask", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.48)
    ap.add_argument("--cmap", type=str, default="hot")
    ap.add_argument("--vmin_pct", type=float, default=5.0)
    ap.add_argument("--vmin_floor", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--upsample_order", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--cols", type=int, default=4, help="Panels per row (default 4 → 12 panels in 3 rows).")
    ap.add_argument("--fig_width", type=float, default=18.0)
    ap.add_argument("--fig_height", type=float, default=12.5)
    ap.add_argument("--out_gif", default="", help="Optional animated GIF of each orientation variant (single frame each).")
    ap.add_argument("--gif_frame_ms", type=int, default=400, help="Delay between GIF frames in ms.")
    ap.add_argument("--gif_cycle_base_flip", action="store_true", help="GIF: cycle base_flip none,lr,ud,lr_ud (48 frames).")
    ap.add_argument("--gif_fig_edge", type=float, default=5.6, help="Figure size inches (square) per GIF frame.")
    ap.add_argument("--gif_dpi", type=int, default=115, help="DPI when rasterizing GIF frames.")
    ap.add_argument(
        "--bundle_base_flip_dir",
        default="",
        help="If set (directory path): write four 12-panel PNGs (underlying none/lr/ud/lr_ud) "
        "into this folder; ignores --out_png name for those files.",
    )

    args = ap.parse_args()
    if not args.coef_npz and not args.coef_csv and not args.highlight_clusters.strip():
        ap.error("Need --coef_npz/--coef_csv or --highlight_clusters.")

    if args.bundle_base_flip_dir.strip():
        _run_bundle_base_flip(args)
    elif args.out_png.strip() or args.out_gif.strip():
        run(args)
    else:
        ap.error("Provide at least one of: --out_png, --out_gif, --bundle_base_flip_dir.")


if __name__ == "__main__":
    main()
