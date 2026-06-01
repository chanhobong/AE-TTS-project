#!/usr/bin/env python3
"""
Run repeated StratifiedShuffleSplit evaluation for the 2 stable settings per latent_source.

Reads `test_metrics_stable_minstd.csv` (from latent_kfold_cv.py --eval_stable_min_std)
and runs `repeated_stratified_shuffle_eval.py` for EACH row:
  - one (latent_source, pooling, model) per row
  - by default 100 repeats, test_size=0.2, random_state=42

Outputs are timestamped via repeated_stratified_shuffle_eval.py's run_tag.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch repeated-split eval from stable settings CSV")
    ap.add_argument("--settings_csv", required=True, help="Path to test_metrics_stable_minstd.csv")
    ap.add_argument("--labels_csv", action="append", required=True, help="Repeat: train.csv/val.csv/test.csv with age/sex/label")
    ap.add_argument(
        "--latent_source",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ_DIR"),
        required=True,
        help="Repeat: latent_source name and NPZ_DIR (StageB patients dir).",
    )
    ap.add_argument("--out_dir", required=True, help="Base output dir (a timestamped run_* subdir will be created).")
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument(
        "--fixed_test_csv",
        default=None,
        help="Optional: fixed test metrics CSV for overlay (e.g., the same stable settings CSV).",
    )
    ap.add_argument(
        "--python",
        default="python",
        help="Python executable to run the repeated evaluator.",
    )
    args = ap.parse_args()

    df = pd.read_csv(os.path.abspath(args.settings_csv))
    need = {"latent_source", "pooling", "model"}
    if not need.issubset(df.columns):
        raise ValueError(f"settings_csv must contain columns {sorted(need)}; got {list(df.columns)}")

    src_map = {str(n): os.path.abspath(str(d)) for n, d in args.latent_source}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    script = os.path.join(os.path.dirname(__file__), "repeated_stratified_shuffle_eval.py")
    if not os.path.exists(script):
        raise FileNotFoundError(script)

    for i, row in df.iterrows():
        src = str(row["latent_source"])
        pooling = str(row["pooling"])
        model = str(row["model"])
        if src not in src_map:
            raise ValueError(f"latent_source '{src}' in settings_csv not provided via --latent_source")

        run_tag = f"{ts}_{src}_{pooling}_{model}"
        cmd = [
            args.python,
            script,
            "--pooling",
            pooling,
            "--classifier",
            model,
            "--n_splits",
            str(int(args.n_splits)),
            "--test_size",
            str(float(args.test_size)),
            "--random_state",
            str(int(args.random_state)),
            "--out_dir",
            os.path.abspath(args.out_dir),
            "--run_tag",
            run_tag,
            "--latent_source",
            src,
            src_map[src],
        ]
        for p in args.labels_csv:
            cmd.extend(["--labels_csv", os.path.abspath(p)])
        if args.fixed_test_csv:
            cmd.extend(["--fixed_test_csv", os.path.abspath(args.fixed_test_csv)])

        print(f"\n[{i+1}/{len(df)}] {src} / {pooling} / {model}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

