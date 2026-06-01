#!/usr/bin/env python3
"""
Spatial grid heatmap overlay from Stage B NPZ (spatial_v2-style schema).

This is NOT Grad-CAM. It maps **histogram-classifier coefficients** (or a single-cluster
highlight) onto encoder bottleneck grid cells using `spatial_flat_idx` + `slice_indices`.

Typical NPZ keys (adapt if your exporter differs):

  embeddings (N_tokens, D)
  cluster_ids (N_tokens,)
  slice_indices (N_tokens,) — axial slice index (or ordinal along Z)
  spatial_flat_idx (N_tokens,) — flat bottleneck cell id; default decode is row_major (see ``--spatial_decode``)
  bottleneck_hw — (Hg, Wg) or length-2 vector
  mask (optional) — same length as tokens

Underlying slice image:

  Provide --underlying_npy (H,W) float/uint16 **or** --underlying_png grayscale/RGB path.
  If omitted, draws **heatmap only** (grid upsampled) on gray canvas.

Weighted heat::

  Prefer ``spatial_y`` / ``spatial_x`` from NPZ (schema v2) when present instead of unpacking
  ``spatial_flat_idx`` with arithmetic.

Optional ``slice_mode=position_norm`` pools tokens whose ``slice_position_norm`` lies in a band
around ``slice_position_center`` with a triangular kernel, yielding one aggregate heatmap
  (underlying slice should match the token closest to the center; see stderr hint).

Contribution options: ``contrib_top_positive_k``, ``contrib_emb_alignment``, ``--emb_raw_norm_dampen``.
See ``utils/scripts/SPATIAL_VIZ_COMMANDS.md`` + ``spatial_viz_paths.example.sh`` for Chanho disk paths.

Example::

  cd /Users/ch.b/Desktop/24-25/TSS/Code/TTS_Project/LDAE_TTS/AE_TTS
  source utils/scripts/spatial_viz_paths.example.sh

  python3 utils/scripts/plot_spatial_coeff_grid_overlay.py \\
    --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \\
    --coef_npz "${COEF_HIST_TRAINVAL}" \\
    --slice_selector slice_index --slice_index 30 \\
    --underlying_npy figures/window_slice30.npy \\
    --out_png figures/patient_slice30_overlay.png

  If NPZ lacks ``spatial_y``/``spatial_x``, add e.g. ``--spatial_decode col_major --heatmap_flip ud``.

  Highlight-only (no coef)::

    python3 utils/scripts/plot_spatial_coeff_grid_overlay.py \\
      --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \\
      --highlight_clusters 43,40,48 \\
      --slice_selector slice_index --slice_index 0 \\
      --out_png /tmp/h.png --fallback_canvas 256
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _accumulate_heat_grid(
    fid_sel: np.ndarray,
    w_sel: np.ndarray,
    *,
    Hg: int,
    Wg: int,
    spatial_decode: str,
) -> np.ndarray:
    """
    Scatter token weights into bottleneck grid Hg×Wg.

    spatial_decode:
      row_major — flat = r*Wg + c (numpy C-order convention in plot_spatial_orientation_sweep_panel)
      col_major — r = flat % Hg, c = flat // Hg
      row_major_transpose_heat — accumulate row_major then transpose heat before upsampling
    """
    ngrid = int(Hg * Wg)
    heat = np.zeros((Hg, Wg), dtype=np.float64)

    fid_sel = fid_sel.astype(np.int64).ravel()
    w_sel = np.asarray(w_sel, dtype=np.float64).ravel()

    decode = spatial_decode.strip().lower()
    for fl, ww in zip(fid_sel, w_sel):
        fl = int(fl)
        if fl < 0 or fl >= ngrid:
            continue
        if decode == "row_major" or decode == "row_major_transpose_heat":
            r, c = int(fl // Wg), int(fl % Wg)
        elif decode == "col_major":
            r, c = int(fl % Hg), int(fl // Hg)
        else:
            raise ValueError(f"Unknown spatial_decode: {spatial_decode!r}")

        heat[r, c] += float(ww)

    if decode == "row_major_transpose_heat":
        heat = heat.T.copy()
    return heat


def _accumulate_heat_grid_yx(
    sy: np.ndarray,
    sx: np.ndarray,
    w_sel: np.ndarray,
    *,
    Hg: int,
    Wg: int,
) -> np.ndarray:
    """Scatter-token weights using exporter ``spatial_y`` / ``spatial_x`` (grid row/col)."""
    heat = np.zeros((Hg, Wg), dtype=np.float64)
    sy = sy.astype(np.int64).ravel()
    sx = sx.astype(np.int64).ravel()
    w_sel = np.asarray(w_sel, dtype=np.float64).ravel()
    for r, c, ww in zip(sy, sx, w_sel):
        if abs(ww) < 1e-18:
            continue
        ri, ci = int(r), int(c)
        if ri < 0 or ri >= Hg or ci < 0 or ci >= Wg:
            continue
        heat[ri, ci] += float(ww)
    return heat


def _position_norm_kernel(spn: np.ndarray, *, center: float, halfwidth: float) -> np.ndarray:
    """Triangular kernel in [0,1]; zero outside ``center ± halfwidth``."""
    dw = float(max(halfwidth, 1e-6))
    d = np.abs(np.asarray(spn, dtype=np.float64).reshape(-1) - float(center))
    return np.clip(1.0 - d / dw, 0.0, 1.0)


def _contrib_coef_top_positive(
    cluster_ids: np.ndarray,
    coef: np.ndarray,
    *,
    top_positive_k: int,
) -> np.ndarray:
    """
    coef[cluster] gated by membership in clusters with largest **positive** linear weights.
    If top_positive_k <= 0, no gating (all clusters weighted by coef as before).
    """
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    c = cluster_ids.astype(np.int64).ravel()
    n = int(c.shape[0])
    base = np.zeros(n, dtype=np.float64)
    valid = (c >= 0) & (c < coef.shape[0])
    bv = np.where(valid)[0]
    base[bv] = coef[c[bv]]

    if top_positive_k <= 0:
        return base

    pos_ix = np.where(coef > 0)[0]
    if pos_ix.size == 0:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(-coef[pos_ix])
    take = min(int(top_positive_k), order.size)
    top_clusters = set(pos_ix[order[:take]].astype(int).tolist())
    gated = np.array([int(x) in top_clusters for x in c], dtype=bool)
    return np.where(gated & valid, base, 0.0)


def _emb_raw_norm_dampen(embeddings: np.ndarray, row_sel: np.ndarray) -> np.ndarray:
    """
    Rows not in ``row_sel`` → 1.0. For selected rows, factor sqrt(median(||emb||₂) / ||emb||₂).

    Intended to tame tokens with far-larger-than-typical raw activation norm when juxtaposed with
    L2-normalised ``embeddings_l2``.

    embeddings: (N_tokens, D) raw latent (not requiring row L2-normalisation).
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    n_tokens = emb.shape[0]
    out = np.ones((n_tokens,), dtype=np.float64)
    sel = np.asarray(row_sel, dtype=bool).ravel()
    if emb.ndim != 2 or not np.any(sel):
        return out
    norms = np.linalg.norm(emb[sel], axis=1)
    med = float(np.median(norms))
    if med < 1e-12:
        return out
    inv_sub = np.sqrt(med / np.maximum(norms, 1e-12)).astype(np.float64)
    out[np.flatnonzero(sel)] = inv_sub
    return out


def _flip_heatmap_2d(heat: np.ndarray, heatmap_flip: str) -> np.ndarray:
    hm = heatmap_flip.strip().lower()
    out = heat
    if hm in ("lr", "lr_ud"):
        out = np.fliplr(out)
    if hm in ("ud", "lr_ud"):
        out = np.flipud(out)
    return out


def _infer_hw_from_flat(fid: np.ndarray) -> tuple[int, int]:
    m = int(np.max(fid[np.isfinite(fid)]))
    s = int(np.ceil(np.sqrt(m + 1)))
    return s, s


def _bottleneck_hw(z: np.lib.npyio.NpzFile) -> tuple[int, int]:
    if "bottleneck_hw" in z.files:
        bw = np.asarray(z["bottleneck_hw"]).ravel().astype(np.int64)
        if bw.size >= 2:
            return int(bw[0]), int(bw[1])
    fid = np.asarray(z["spatial_flat_idx"], dtype=np.int64).ravel()
    return _infer_hw_from_flat(fid)


def _load_coef(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or str(path).strip() == "":
        return None
    p = os.path.abspath(path)
    if p.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(p, header=None)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0).to_numpy(np.float64)
    return np.load(p).astype(np.float64).reshape(-1)


def _token_weights(
    cluster_ids: np.ndarray,
    coef: Optional[np.ndarray],
    highlight: Optional[list[int]],
) -> np.ndarray:
    c = cluster_ids.astype(np.int64).ravel()
    n = int(c.shape[0])
    if highlight is not None and len(highlight) > 0:
        hv = np.zeros(n, dtype=np.float64)
        for k in highlight:
            hv[c == int(k)] = 1.0
        return hv
    if coef is None:
        raise SystemExit("Need --coef_npz/_csv or --highlight_clusters.")
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    w = np.zeros(n, dtype=np.float64)
    valid = (c >= 0) & (c < coef.shape[0])
    vv = np.where(valid)[0]
    w[vv] = coef[c[vv]]
    # Optionally ReLU-positive contribution for "risk" direction
    return w


def _tone_underlying_display(
    base: np.ndarray,
    *,
    gamma: float,
    brightness_mult: float,
) -> np.ndarray:
    """
    Adjust grayscale base for display. Intended for intensities roughly in ``[0, 1]``
    after HU windowing; ``gamma < 1`` darkens midtones, ``brightness_mult < 1``
    dims the whole ramp.
    """
    b = np.nan_to_num(np.asarray(base, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    g = float(gamma)
    m = float(brightness_mult)
    mx = float(np.nanmax(b)) if np.size(b) else 0.0

    unitish = mx <= 1.05 + 1e-6
    if unitish:
        b = np.clip(b, 0.0, 1.0)
        if g > 0.0 and abs(g - 1.0) > 1e-9:
            b = np.power(np.maximum(b, 0.0), g)
        if abs(m - 1.0) > 1e-9:
            b = np.clip(b * m, 0.0, 1.0)
        return b

    if abs(m - 1.0) > 1e-9:
        b = b * m
    return b


def _load_underlying(path: str) -> np.ndarray:
    p = os.path.abspath(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in IMAGE_EXTS:
        import matplotlib.image as mpimg

        im = mpimg.imread(p)
        if im.ndim == 3:
            im = (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).astype(np.float64)
        else:
            im = np.asarray(im, dtype=np.float64)
        im = np.nan_to_num(im, nan=0.0)
        return im

    arr = np.load(p)
    if arr.ndim != 2:
        raise SystemExit(f"underlying_npy must be 2D (H,W), got {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    npz_path = os.path.abspath(args.npz)
    z = np.load(npz_path, allow_pickle=True)
    mandatory = {"cluster_ids", "slice_indices"}
    miss_m = mandatory - set(z.files)
    if miss_m:
        raise SystemExit(f"NPZ missing keys {sorted(miss_m)}: {npz_path}")

    use_xy = ("spatial_y" in z.files) and ("spatial_x" in z.files)
    if not use_xy and "spatial_flat_idx" not in z.files:
        raise SystemExit("NPZ missing spatial_y/spatial_x and spatial_flat_idx (need one spatial layout source).")

    cid = np.asarray(z["cluster_ids"], dtype=np.int64).ravel()
    sidx = np.asarray(z["slice_indices"], dtype=np.int64).ravel()
    n_tokens = cid.shape[0]
    fid = np.asarray(z["spatial_flat_idx"], dtype=np.int64).ravel() if "spatial_flat_idx" in z.files else None
    if fid is not None and fid.shape[0] != n_tokens:
        raise SystemExit("spatial_flat_idx length mismatch.")
    if sidx.shape[0] != n_tokens:
        raise SystemExit("slice_indices length mismatch.")

    sy_all = sx_all = None
    if use_xy:
        sy_all = np.asarray(z["spatial_y"], dtype=np.int64).ravel()
        sx_all = np.asarray(z["spatial_x"], dtype=np.int64).ravel()
        if sy_all.shape[0] != n_tokens or sx_all.shape[0] != n_tokens:
            raise SystemExit("spatial_y / spatial_x length mismatch.")

    spn_all = None
    if "slice_position_norm" in z.files:
        spn_all = np.asarray(z["slice_position_norm"], dtype=np.float64).ravel()
        if spn_all.shape[0] != n_tokens:
            raise SystemExit("slice_position_norm length mismatch.")

    emb_raw = np.asarray(z["embeddings"]) if "embeddings" in z.files else None
    if emb_raw is not None and emb_raw.shape[0] != n_tokens:
        raise SystemExit("embeddings length mismatch.")

    emb_l2 = np.asarray(z["embeddings_l2"]) if "embeddings_l2" in z.files else None
    if emb_l2 is not None and emb_l2.shape[0] != n_tokens:
        raise SystemExit("embeddings_l2 length mismatch.")

    mask = np.ones(n_tokens, dtype=bool)
    if args.respect_mask and "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1).astype(np.int64)
        if m.shape[0] == n_tokens:
            mask = m.astype(bool)

    Hg, Wg = _bottleneck_hw(z)

    coef = _load_coef(args.coef_npz or args.coef_csv)
    highlight_list = (
        [int(x) for x in args.highlight_clusters.replace(",", " ").split() if x.strip()]
        if args.highlight_clusters.strip()
        else None
    )

    cid_m = cid[mask]
    sidx_m = sidx[mask]
    if use_xy:
        sy_m = sy_all[mask]
        sx_m = sx_all[mask]
        fid_m = None
    else:
        fid_m = fid[mask]  # type: ignore[index]
        sy_m = sx_m = None

    spn_m = spn_all[mask] if spn_all is not None else None
    emb_rm = emb_raw[mask] if emb_raw is not None else None
    emb_l2m = emb_l2[mask] if emb_l2 is not None else None

    n_m = cid_m.shape[0]
    if n_m == 0:
        raise SystemExit("No tokens after masking.")

    # --- token weights ---
    slice_mode = str(getattr(args, "slice_mode", "slice_index") or "slice_index").strip()
    top_k_pos = int(getattr(args, "contrib_top_positive_k", 0) or 0)
    emb_alignment = str(getattr(args, "contrib_emb_alignment", "none") or "none").strip().lower()

    highlight = highlight_list if highlight_list else None

    if highlight is not None:
        base_w = _token_weights(cid_m, coef, highlight)
        align_fac = np.ones((n_m,), dtype=np.float64)
    elif coef is not None:
        base_w = _contrib_coef_top_positive(cid_m, coef, top_positive_k=top_k_pos)
        if top_k_pos > 0 and not np.any(np.abs(base_w) > 1e-18):
            print(
                "[warn] contrib_top_positive_k emptied all weights (coef may have no positive entries).",
                file=sys.stderr,
            )
        align_fac = np.ones((n_m,), dtype=np.float64)
        if emb_alignment == "cosine_relu" and emb_l2m is not None:
            pool = base_w > 1e-9
            if np.any(pool):
                ev = np.mean(emb_l2m.astype(np.float64)[pool], axis=0)
                evn = float(np.linalg.norm(ev))
                if evn > 1e-12:
                    ev /= evn
                    proj = emb_l2m.astype(np.float64) @ ev
                    align_fac = np.maximum(proj, 0.0).astype(np.float64)
        elif emb_alignment != "none" and emb_alignment != "cosine_relu":
            raise SystemExit(f"Unknown --contrib_emb_alignment: {emb_alignment}")
    else:
        raise SystemExit("Need coef or highlight_clusters.")

    raw_dampen = bool(getattr(args, "emb_raw_norm_dampen", False))
    if raw_dampen and emb_rm is not None:
        damp = _emb_raw_norm_dampen(emb_rm, np.ones((n_m,), dtype=bool))
    else:
        damp = np.ones((n_m,), dtype=np.float64)

    combo = base_w * align_fac * damp

    spatial_decode = str(getattr(args, "spatial_decode", "row_major") or "row_major").strip()
    heatmap_flip = str(getattr(args, "heatmap_flip", "none") or "none").strip()

    display_slice_txt = ""
    if slice_mode == "slice_index":
        sl_ix = int(args.slice_index)
        if sl_ix < 0:
            raise SystemExit("slice_mode=slice_index requires --slice_index >= 0.")
        sel = sidx_m == sl_ix
        if not np.any(sel):
            available = ",".join(map(str, sorted(set(sidx_m.tolist()))[:40]))
            raise SystemExit(f"No tokens slice_index={sl_ix}. Examples: {available} …")

        pos_w = np.ones((n_m,), dtype=np.float64)
        w_eff = combo * pos_w * np.asarray(sel, dtype=np.float64)
        display_slice_txt = str(sl_ix)
        print("[info] slice_mode=slice_index accumulating single slice.", file=sys.stderr)

    elif slice_mode == "position_norm":
        if spn_m is None:
            raise SystemExit("slice_mode=position_norm requires slice_position_norm in NPZ.")

        cen = float(getattr(args, "slice_position_center", 0.8))
        hw = float(getattr(args, "slice_position_halfwidth", 0.12))
        pos_w = _position_norm_kernel(spn_m, center=cen, halfwidth=max(hw, 1e-6))

        w_eff = combo * pos_w

        j_star = int(np.argmax(w_eff))
        display_slice_ix = int(sidx_m[j_star])
        display_slice_txt = f"display≈slice_index {display_slice_ix} | norm_peak {float(spn_m[j_star]):.3f}"

        band = np.abs(np.asarray(spn_m, dtype=np.float64) - cen) <= max(hw * 1.0001, 1e-9)
        n_band = int(np.sum(band & (combo > 0)))
        sum_w_eff = float(np.sum(w_eff))
        print(
            f"[info] slice_mode=position_norm center={cen} ±{hw} tokens_in_band∩contrib={n_band} "
            f"sum(weights)={sum_w_eff:.4g}",
            file=sys.stderr,
        )
        print(f"[hint] Underlying axial slice aligned to {display_slice_txt}", file=sys.stderr)

    else:
        raise SystemExit(f"Unknown --slice_mode: {slice_mode!r}")

    if use_xy:
        heat = _accumulate_heat_grid_yx(
            sy_m,
            sx_m,
            w_eff,
            Hg=int(Hg),
            Wg=int(Wg),
        )
        geo_note = "grid=spatial_yx"
    elif fid_m is not None:
        heat = _accumulate_heat_grid(
            fid_m,
            w_eff,
            Hg=int(Hg),
            Wg=int(Wg),
            spatial_decode=spatial_decode,
        )
        geo_note = f"grid=flat({spatial_decode})"
    else:
        raise SystemExit("Internal spatial layout error.")

    if np.max(np.abs(heat)) < 1e-18:
        print("[warn] heatmap is numerically flat (check coef/kernels/mask overlap).", file=sys.stderr)

    if args.relu_weights:
        heat = np.maximum(heat, 0.0)

    heat = _flip_heatmap_2d(heat, heatmap_flip)

    # Upsample heat to overlay size
    if args.underlying_png or args.underlying_npy:
        base = _load_underlying(str(args.underlying_png or args.underlying_npy))
        Himg, Wimg = base.shape[:2]
    else:
        Himg = Wimg = int(args.fallback_canvas)
        base = np.full((Himg, Wimg), 0.3, dtype=np.float64)

    ug = float(getattr(args, "underlying_gamma", 1.0) or 1.0)
    ub = float(getattr(args, "underlying_brightness_mult", 1.0) or 1.0)
    base = _tone_underlying_display(base, gamma=ug, brightness_mult=ub)

    Hi_eff, Wi_eff = int(heat.shape[0]), int(heat.shape[1])
    try:
        from scipy.ndimage import zoom
    except ImportError as e:
        raise SystemExit(
            "This script needs scipy.ndimage.zoom for resizing. pip install scipy"
        ) from e

    zh = float(Himg) / float(Hi_eff)
    zw = float(Wimg) / float(Wi_eff)
    method = getattr(args, "upsample_order", "nearest")
    order = {"nearest": 0, "bilinear": 1}.get(method, 0)
    hi = zoom(heat, (zh, zw), order=order)

    vmax = float(np.max(hi)) if np.any(np.isfinite(hi)) else 1.0
    if vmax < 1e-18:
        vmin, vmax = 0.0, 1.0
    else:
        nz = hi[hi > 0]
        vmin = float(np.percentile(nz, float(args.vmin_pct))) if nz.size > 0 else 0.0
        vmin = float(np.clip(vmin, 0.0, vmax - 1e-9))
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    plt_cm_hot = plt.get_cmap(args.cmap)

    fig, ax = plt.subplots(figsize=(6, 6))
    imx = ax.imshow(base, cmap="gray", interpolation="nearest", aspect="equal")
    if args.show_colorbar_gray:
        fig.colorbar(imx, ax=ax, fraction=0.046, pad=0.04)
    him = plt_cm_hot(norm(hi.astype(np.float64)))
    ax.imshow(him[..., :3], alpha=float(args.overlay_alpha))
    meta = []
    tid = npz_path.split(os.sep)[-1].replace(".npz", "")
    meta.append(f"patient.npz stem: {tid}")
    meta.append(f"bottleneck {Hg}x{Wg}")
    meta.append(f"slice_mode={slice_mode} ({display_slice_txt})")
    meta.append(geo_note)
    if slice_mode == "position_norm":
        meta.append(
            f"norm center={float(getattr(args, 'slice_position_center', 0.8)):.2f}"
            f"±{float(getattr(args, 'slice_position_halfwidth', 0.12)):.2f}"
        )
    if highlight_list:
        meta.append(f"clusters={highlight_list}")
    if coef is not None:
        meta.append(f"coef K={coef.shape[0]} top+={top_k_pos or 'ALL'} emb={emb_alignment}")
    meta.append(f"heatmap_flip={heatmap_flip}")
    if not use_xy:
        meta.append(f"spatial_decode={spatial_decode}")
    if abs(ug - 1.0) > 1e-6 or abs(ub - 1.0) > 1e-6:
        meta.append(f"underlying γ={ug} ×{ub}")
    ax.set_title("\n".join(meta), fontsize=7)
    ax.axis("off")
    outp = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    plt.savefig(outp, dpi=int(args.dpi), bbox_inches="tight")
    plt.close()
    print(f"Wrote {outp}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid heatmap overlay from spatial Stage B NPZ + linear coef.")
    ap.add_argument("--npz", required=True, help="Per-patient .npz (spatial_v2 style).")
    ap.add_argument(
        "--slice_selector",
        choices=["slice_index", "position_norm"],
        default="slice_index",
        help="slice_index = one axial index; position_norm = band around slice_position_norm.",
    )

    ap.add_argument(
        "--slice_index",
        type=int,
        default=-1,
        help="Used when slice_selector=slice_index (required for that mode). Ignored otherwise.",
    )
    ap.add_argument(
        "--slice_position_center",
        type=float,
        default=0.8,
        help="APEX-heavy norm ≈ 0.85–0.95 depending on preprocessing; apex band tuning.",
    )
    ap.add_argument(
        "--slice_position_halfwidth",
        type=float,
        default=0.12,
        help="Half-width for triangular kernel in slice_position_norm space.",
    )
    ap.add_argument("--out_png", required=True)

    cg = ap.add_mutually_exclusive_group()
    cg.add_argument("--coef_npz", default="", help="1D coef array (.npy), length K=num clusters.")
    cg.add_argument("--coef_csv", default="", help="CSV first column only: K coefficients.")

    ap.add_argument(
        "--highlight_clusters",
        default="",
        help="Space/comma-separated cluster ids → binary highlight (instead of coef).",
    )

    ap.add_argument(
        "--contrib_top_positive_k",
        type=int,
        default=0,
        help="Keep only clusters with largest positive logistic coef (>0 ⇒ top-k gated). 0 disables.",
    )
    ap.add_argument(
        "--contrib_emb_alignment",
        choices=["none", "cosine_relu"],
        default="none",
        help="With cosine_relu: coef weight × max(0, cos(l2_emb, pooled positive direction)). Needs embeddings_l2.",
    )
    ap.add_argument(
        "--emb_raw_norm_dampen",
        action="store_true",
        help="Scale each token sqrt(median(||raw_emb||₂)/||raw_emb||₂) to tame outlier norms vs L2 embeddings.",
    )

    ap.add_argument("--relu_weights", action="store_true")

    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--underlying_npy", default="", help="2D slice intensities.")
    bg.add_argument("--underlying_png", default="", help="Grayscale/RGB PNG slice.")

    ap.add_argument("--fallback_canvas", type=int, default=256)
    ap.add_argument("--respect_mask", action="store_true")
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--cmap", type=str, default="hot")
    ap.add_argument("--vmin_pct", type=float, default=5.0, help=" vmin = percentile(nonzero heat, this). ")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--upsample_order", choices=["nearest", "bilinear"], default="nearest")
    ap.add_argument("--show_colorbar_gray", action="store_true")

    ap.add_argument(
        "--spatial_decode",
        choices=["row_major", "col_major", "row_major_transpose_heat"],
        default="row_major",
        help="Only when NPZ lacks spatial_y/x; unpacking spatial_flat_idx.",
    )

    ap.add_argument(
        "--heatmap_flip",
        choices=["none", "lr", "ud", "lr_ud"],
        default="none",
        help="Flip low-res heatmap (Hg×Wg) before zoom.",
    )
    ap.add_argument("--underlying_gamma", type=float, default=1.0)
    ap.add_argument("--underlying_brightness_mult", type=float, default=1.0)

    args = ap.parse_args()

    if not args.coef_npz and not args.coef_csv and not str(args.highlight_clusters).strip():
        ap.error("Provide --coef_npz/--coef_csv or --highlight_clusters.")

    setattr(args, "slice_mode", str(args.slice_selector))

    if getattr(args, "slice_mode") == "slice_index" and int(args.slice_index) < 0:
        ap.error("slice_selector=slice_index requires --slice_index >= 0.")

    try:
        run(args)
    except ImportError as e:
        if "scipy" in str(e).lower():
            raise SystemExit("Need scipy (`pip install scipy`) for zoom upsampling.") from e
        raise


if __name__ == "__main__":
    main()

