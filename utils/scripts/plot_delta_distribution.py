#!/usr/bin/env python3
"""
Plot delta distribution (Normal vs TTS) and compute Cliff's delta.
"""

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def cliffs_delta(x: List[float], y: List[float]) -> float:
    # Effect size in [-1, 1]
    greater = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1
            elif xi < yi:
                less += 1
    denom = len(x) * len(y)
    if denom == 0:
        return 0.0
    return (greater - less) / denom


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Delta distribution plot (Normal vs TTS)")
    p.add_argument("--scores_csv", required=True, help="Path to triplane_scores.csv")
    p.add_argument("--use_norm", action="store_true", help="Use S_delta_norm instead of S_delta")
    p.add_argument("--out_png", required=True, help="Output figure path")
    p.add_argument("--out_txt", default=None, help="Optional summary text output")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    scores_path = Path(args.scores_csv)

    normal_vals = []
    tts_vals = []
    key = "S_delta_norm" if args.use_norm else "S_delta"

    with scores_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            val = float(row[key])
            if label == 0:
                normal_vals.append(val)
            else:
                tts_vals.append(val)

    cd = cliffs_delta(normal_vals, tts_vals)
    normal_mean = sum(normal_vals) / max(len(normal_vals), 1)
    tts_mean = sum(tts_vals) / max(len(tts_vals), 1)

    plt.figure(figsize=(6, 4))
    plt.violinplot([normal_vals, tts_vals], showmeans=True, showmedians=True)
    plt.boxplot([normal_vals, tts_vals], widths=0.2)
    plt.xticks([1, 2], ["Normal", "TTS"])
    plt.ylabel(key)
    plt.title(f"Delta distribution ({key}) | Cliff's delta={cd:.3f}")
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)

    if args.out_txt:
        out_txt = Path(args.out_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w") as f:
            f.write(f"key: {key}\n")
            f.write(f"n_normal: {len(normal_vals)}\n")
            f.write(f"n_tts: {len(tts_vals)}\n")
            f.write(f"mean_normal: {normal_mean}\n")
            f.write(f"mean_tts: {tts_mean}\n")
            f.write(f"cliffs_delta: {cd}\n")


if __name__ == "__main__":
    main()
