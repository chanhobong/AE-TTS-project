#!/usr/bin/env python3
"""
Latent profile along slices: z[s, j] vs stack position for latent dim j.

Primary use — compare **one normal + one TTS** (NPZ stem = patient_id):

  python plot_latent_slice_profile.py \\
    --npz_dir .../patients --latent_dim 63 \\
    --patient_normal AAP_50415783 --patient_tts DCA_09939935 \\
    --out_png ./latent_profile_dim63.png

X-axis: consecutive index 0 .. K-1 over **valid** NPZ rows (same order as embeddings).

`--firstdiff`: plot Δz = z[t] − z[t−1] along valid rows (removes slow trend; fluctuation near 0).

Optional: `--cohort` + `--labels_csv` for group mean ± band.

All files in folder — color by NPZ stem (normal = `_12F` / `_71M` style suffix):

  python plot_latent_slice_profile.py --folder_all_colored \\
    --npz_dir .../patients --latent_dim 63 --out_png ./all_lines.png

  # normal만 / TTS만
  ... --folder_subset normal --out_png ./normal_only.png
  ... --folder_subset tts --out_png ./tts_only.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _load_profile_path(
    npz_path: str,
    j: int,
    respect_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (original row indices for valid slices, z[:, j])."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in z.files:
        raise ValueError(f"No embeddings in {npz_path}")
    emb = np.asarray(z["embeddings"], dtype=np.float64)
    if emb.ndim != 2 or j >= emb.shape[1]:
        raise ValueError(f"bad shape {emb.shape} or dim {j} out of range")
    n = emb.shape[0]
    valid = np.ones(n, dtype=bool)
    if respect_mask and "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1)
        if m.shape[0] == n:
            valid = m.astype(bool)
    idx = np.where(valid)[0]
    y = emb[idx, j]
    return idx.astype(np.float64), y


def _load_profile(
    npz_dir: str,
    patient_id: str,
    j: int,
    respect_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(npz_dir, f"{patient_id}.npz")
    return _load_profile_path(path, j, respect_mask)


def _apply_firstdiff_valid(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Along valid-slice sequence: Δ[t] = y[t] - y[t-1].
    x = 1, 2, .. len(y)-1 (index t of the upper slice in the pair).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    dy = np.diff(y)
    x = np.arange(1, y.size, dtype=np.float64)
    return x, dy


def _profile_resampled(
    npz_dir: str,
    patient_id: str,
    j: int,
    respect_mask: bool,
    n_grid: int,
    firstdiff: bool = False,
) -> Optional[np.ndarray]:
    """z[:,j] on valid slices, linearly interpolated to n_grid points in [0,1]; optional np.diff along grid."""
    path = os.path.join(npz_dir, f"{patient_id}.npz")
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    if "embeddings" not in z.files:
        return None
    emb = np.asarray(z["embeddings"], dtype=np.float64)
    if emb.ndim != 2 or j >= emb.shape[1]:
        return None
    n = emb.shape[0]
    valid = np.ones(n, dtype=bool)
    if respect_mask and "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1)
        if m.shape[0] == n:
            valid = m.astype(bool)
    y = emb[valid, j].astype(np.float64)
    k = int(y.size)
    if k == 0:
        return None
    grid_n = max(2, int(n_grid))
    grid = np.linspace(0.0, 1.0, grid_n)
    if k == 1:
        out = np.full(grid_n, float(y[0]))
    else:
        xi = np.linspace(0.0, 1.0, k)
        out = np.interp(grid, xi, y)
    if firstdiff:
        if out.size < 2:
            return None
        return np.diff(out)
    return out


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


def _plot_cohort_bands(
    npz_dir: str,
    labels: dict[str, int],
    j: int,
    respect_mask: bool,
    n_grid: int,
    n_std: float,
    use_sem: bool,
    title: Optional[str],
    out_png: str,
    firstdiff: bool = False,
) -> None:
    grid = np.linspace(0.0, 1.0, max(2, int(n_grid)))
    if firstdiff:
        grid_plot = 0.5 * (grid[:-1] + grid[1:])
    else:
        grid_plot = grid
    by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz"):
            continue
        pid = fn.replace(".npz", "")
        if pid not in labels:
            continue
        row = _profile_resampled(npz_dir, pid, j, respect_mask, n_grid, firstdiff=firstdiff)
        if row is None:
            continue
        by_class[int(labels[pid])].append(row)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    # Normal = dark blue line + light blue band; TTS = dark gold/yellow line + light yellow band
    styles = [
        (0, "Normal (case=0)", "#0d47a1", "#64b5f6", 0.38),
        (1, "TTS (case=1)", "#f57f17", "#ffee58", 0.42),
    ]
    for cls, name, line_c, fill_c, fill_a in styles:
        rows = by_class.get(cls, [])
        if not rows:
            continue
        M = np.stack(rows, axis=0)
        mean = np.nanmean(M, axis=0)
        if use_sem:
            n_ok = np.sum(np.isfinite(M), axis=0).clip(min=1)
            spread = np.nanstd(M, axis=0, ddof=0) / np.sqrt(n_ok)
        else:
            spread = np.nanstd(M, axis=0, ddof=0) * float(n_std)
        lo, hi = mean - spread, mean + spread
        band_txt = "SEM" if use_sem else f"{n_std:g}·SD"
        ax.fill_between(grid_plot, lo, hi, color=fill_c, alpha=fill_a, linewidth=0, zorder=1)
        ax.plot(
            grid_plot,
            mean,
            color=line_c,
            linestyle="-",
            linewidth=2.6,
            zorder=3,
            label=f"{name}: mean (solid), fill=±{band_txt} (n={len(rows)})",
        )

    ax.set_xlabel(
        "Relative position along stack (midpoint of Δ step)"
        if firstdiff
        else "Relative position along slice stack (0→1, NPZ order resampled)"
    )
    y_label = (
        f"Delta z on grid: z[s,{j}] - z[s-1,{j}] (after interp)"
        if firstdiff
        else f"Latent z[s, {j}] (interpolated)"
    )
    ax.set_ylabel(y_label)
    ax.set_title(
        title
        or (
            f"Cohort latent profile — dim {j} (first differences)\n(solid = mean; tint = ± spread)"
            if firstdiff
            else f"Cohort latent profile — dim {j}\n(solid = group mean; tint = ± spread)"
        )
    )
    if firstdiff:
        ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle=":", zorder=0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_folder_all_colored(
    npz_dir: str,
    j: int,
    respect_mask: bool,
    normal_stem_regex: str,
    normal_color: str,
    tts_color: str,
    title: Optional[str],
    out_png: str,
    subset: str = "all",
    firstdiff: bool = False,
) -> None:
    """
    One line per .npz; normal if stem matches regex (default: *_<age><F|M>).
    subset: 'all' | 'normal' | 'tts' — only plot that group.
    """
    if subset not in ("all", "normal", "tts"):
        raise ValueError("subset must be all, normal, or tts")
    pat = re.compile(normal_stem_regex, re.IGNORECASE)
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    fig, ax = plt.subplots(figsize=(10, 5))
    n_n = n_t = 0
    skipped = 0
    plotted = 0
    for fn in files:
        stem = fn[: -len(".npz")]
        path = os.path.join(npz_dir, fn)
        try:
            _, y = _load_profile_path(path, j, respect_mask)
        except Exception:
            skipped += 1
            continue
        if firstdiff:
            x, y = _apply_firstdiff_valid(y)
            if y.size == 0:
                skipped += 1
                continue
        else:
            x = np.arange(len(y), dtype=np.float64)
        is_normal = bool(pat.match(stem))
        if subset == "normal" and not is_normal:
            continue
        if subset == "tts" and is_normal:
            continue
        if is_normal:
            c = normal_color
            n_n += 1
        else:
            c = tts_color
            n_t += 1
        plotted += 1
        ax.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.05,
            color=c,
            alpha=0.88,
            label="_nolegend_",
        )

    if subset == "all":
        leg = [
            Line2D(
                [0], [0], color=normal_color, lw=2.2, marker="o", ms=4, label=f"Normal (n={n_n})"
            ),
            Line2D(
                [0], [0], color=tts_color, lw=2.2, marker="o", ms=4, label=f"TTS / other (n={n_t})"
            ),
        ]
    elif subset == "normal":
        leg = [
            Line2D(
                [0], [0], color=normal_color, lw=2.2, marker="o", ms=4, label=f"Normal only (n={plotted})"
            ),
        ]
    else:
        leg = [
            Line2D(
                [0], [0], color=tts_color, lw=2.2, marker="o", ms=4, label=f"TTS / other only (n={plotted})"
            ),
        ]
    ax.legend(handles=leg, loc="best", fontsize=9)
    ax.set_xlabel(
        "Slice index t (dz = z[t] - z[t-1] along valid rows)"
        if firstdiff
        else "Valid slice order index (0 .. K-1, NPZ row order)"
    )
    ax.set_ylabel(
        f"dz = z[s,{j}] - z[s-1,{j}]" if firstdiff else f"Latent z[s, {j}]"
    )
    sub_note = {"all": "all stems", "normal": "normal stems only", "tts": "TTS / other stems only"}[
        subset
    ]
    ax.set_title(
        title
        or f"Folder profiles — dim {j} ({sub_note}; regex {normal_stem_regex!r} → normal)"
        + ("; first differences" if firstdiff else "")
    )
    if firstdiff:
        ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle=":", zorder=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(
        f"Wrote {out_png}  (subset={subset}, lines={plotted}, normal_in_run={n_n}, tts_in_run={n_t}, skipped_load={skipped})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Line plot: latent dim vs slice index")
    ap.add_argument("--npz_dir", required=True, help="Directory of per-patient .npz")
    ap.add_argument("--latent_dim", type=int, required=True, help="Axis index j (0 … D-1)")
    ap.add_argument(
        "--folder_all_colored",
        action="store_true",
        help="Plot every .npz in folder: blue if stem matches --normal_stem_regex, else gold (TTS)",
    )
    ap.add_argument(
        "--normal_stem_regex",
        default=r".*_\d+[FM]$",
        help=r"Python regex on stem w/o .npz (default: *_<digits>F or M e.g. LBH_22392393_71F)",
    )
    ap.add_argument(
        "--color_normal",
        default="#1565c0",
        help="Matplotlib color for normal lines (--folder_all_colored)",
    )
    ap.add_argument(
        "--color_tts",
        default="#f9a825",
        help="Matplotlib color for TTS/other lines (--folder_all_colored)",
    )
    ap.add_argument(
        "--folder_subset",
        choices=("all", "normal", "tts"),
        default="all",
        help="With --folder_all_colored: plot only normal-like stems, only TTS-like, or both",
    )
    ap.add_argument(
        "--cohort",
        action="store_true",
        help="All NPZ patients: normal vs TTS cohort mean (blue/gold solid) ± light same-hue band",
    )
    ap.add_argument(
        "--labels_csv",
        action="append",
        default=None,
        help="With --cohort: e.g. Combined_Labels.csv (repeat to merge)",
    )
    ap.add_argument(
        "--n_grid",
        type=int,
        default=65,
        help="Resampling resolution along [0,1] for cohort mode",
    )
    ap.add_argument(
        "--band_std",
        type=float,
        default=1.0,
        help="Half-width multiplier for SD band (ignored if --band_sem)",
    )
    ap.add_argument(
        "--band_sem",
        action="store_true",
        help="Shaded band = ± SEM per grid point instead of ± band_std·SD",
    )
    ap.add_argument(
        "--patient_normal",
        default=None,
        help="patient_id stem (one normal); optional if only --patient",
    )
    ap.add_argument(
        "--patient_tts",
        default=None,
        help="patient_id stem (one TTS); optional if only --patient",
    )
    ap.add_argument(
        "--patient",
        action="append",
        default=None,
        metavar="ID",
        help="Repeat for arbitrary patients (label shown as ID). Overrides paired normal/tts if set.",
    )
    ap.add_argument("--out_png", required=True)
    ap.add_argument(
        "--respect_mask",
        action="store_true",
        help="Use NPZ mask to drop invalid slices if present",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Figure title (default: auto from dim)",
    )
    ap.add_argument(
        "--firstdiff",
        action="store_true",
        help="Plot z[t]−z[t−1] along valid slices (or np.diff on cohort grid)",
    )
    args = ap.parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    j = int(args.latent_dim)

    if args.cohort and args.folder_all_colored:
        print("Use only one of --cohort or --folder_all_colored", file=sys.stderr)
        sys.exit(1)

    if args.cohort:
        if not args.labels_csv:
            print("--cohort requires at least one --labels_csv", file=sys.stderr)
            sys.exit(1)
        labels = _merge_labels([os.path.abspath(p) for p in args.labels_csv])
        _plot_cohort_bands(
            npz_dir,
            labels,
            j,
            args.respect_mask,
            args.n_grid,
            args.band_std,
            args.band_sem,
            args.title,
            os.path.abspath(args.out_png),
            firstdiff=args.firstdiff,
        )
        return

    if args.folder_all_colored:
        if args.patient or args.patient_normal or args.patient_tts:
            print(
                "With --folder_all_colored, omit --patient / --patient_normal / --patient_tts",
                file=sys.stderr,
            )
            sys.exit(1)
        _plot_folder_all_colored(
            npz_dir,
            j,
            args.respect_mask,
            args.normal_stem_regex,
            args.color_normal,
            args.color_tts,
            args.title,
            os.path.abspath(args.out_png),
            subset=args.folder_subset,
            firstdiff=args.firstdiff,
        )
        return

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.patient:
        for pid in args.patient:
            _x, y = _load_profile(npz_dir, pid, j, args.respect_mask)
            if args.firstdiff:
                x, y = _apply_firstdiff_valid(y)
                if y.size == 0:
                    print(f"skip {pid}: need >=2 valid slices for --firstdiff", file=sys.stderr)
                    continue
            else:
                x = np.arange(len(y), dtype=np.float64)
            series.append((str(pid), x, y))
    else:
        if not args.patient_normal or not args.patient_tts:
            print(
                "Provide both --patient_normal and --patient_tts, or use --patient ID (repeat)",
                file=sys.stderr,
            )
            sys.exit(1)
        for label, pid in (
            ("normal", args.patient_normal),
            ("TTS", args.patient_tts),
        ):
            _x, y = _load_profile(npz_dir, pid, j, args.respect_mask)
            if args.firstdiff:
                x, y = _apply_firstdiff_valid(y)
                if y.size == 0:
                    print(f"skip {pid}: need >=2 valid slices for --firstdiff", file=sys.stderr)
                    continue
            else:
                x = np.arange(len(y), dtype=np.float64)
            series.append((f"{label} ({pid})", x, y))

    if not series:
        print("No series to plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, x, y in series:
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2, label=label, alpha=0.9)
    ax.set_xlabel(
        "Slice index t (dz = z[t] - z[t-1])"
        if args.firstdiff
        else "Valid slice order index (0 .. K-1, NPZ row order)"
    )
    ax.set_ylabel(
        f"dz = z[s,{j}] - z[s-1,{j}]" if args.firstdiff else f"Latent z[s, {j}]"
    )
    ax.set_title(
        args.title
        or (
            f"Latent slice profile — dim {j} (first differences)"
            if args.firstdiff
            else f"Latent slice profile — dim {j}\n(smooth vs fluctuating along stack)"
        )
    )
    if args.firstdiff:
        ax.axhline(0.0, color="0.45", linewidth=0.9, linestyle=":", zorder=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
