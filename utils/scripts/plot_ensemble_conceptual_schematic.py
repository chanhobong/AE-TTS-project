#!/usr/bin/env python3
"""
Conceptual (schematic) figure for the paper: two complementary geometries → fused score.

**Not data-driven** — annotate as ``Conceptual schematic`` in the caption.

Example
-------
  python3 utils/scripts/plot_ensemble_conceptual_schematic.py \\
    --out_png Report/Draft/figures/fig_ensemble_concept_schematic.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    ap = argparse.ArgumentParser(description="Conceptual ensemble geometry schematic (publication-style).")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(xy, w, h, text, sub, face, edge="#2f2f2f"):
        x, y = xy
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=11, fontweight="600")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9.2, color="#333")

    # Plain branch (left)
    box((0.35, 1.05), 2.65, 1.9, "Plain AE", "Variability geometry\n(slice trajectory / spread)", "#dbeafe")
    # MONAI branch (left lower visual — middle column)
    box((3.85, 1.05), 2.65, 1.9, "MONAI AE", "Occupancy geometry\n(cluster histogram / mixing)", "#fef3c7")

    # Fusion
    box((7.35, 1.15), 2.35, 1.7, "Fused score", r"$s = w\,s_{\mathrm{P}} + (1-w)\,s_{\mathrm{M}}$" + "\n" + "disease risk (OOS)", "#dcfce7")

    def arrow(p1, p2):
        a = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.35,
            color="#374151",
            shrinkA=4,
            shrinkB=4,
        )
        ax.add_patch(a)

    arrow((3.05, 2.0), (7.25, 2.05))
    arrow((6.55, 2.0), (7.25, 2.05))

    ax.text(5.0, 3.25, "Complementary patient-level representations", ha="center", fontsize=12, fontweight="600")
    ax.text(
        5.0,
        0.35,
        "Conceptual schematic (not to scale; not a joint embedding of features).",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#555",
    )

    out = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
