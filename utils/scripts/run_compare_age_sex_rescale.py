#!/usr/bin/env python3
"""
Compare two clinical-concat scaling strategies on repeated splits:

(A) default concat:
    - StandardScaler on latent only (train fit -> test transform)
    - age MinMaxScaler (train fit -> test transform)
    - sex: M=0, F=1
    - concat at the end

(B) final rescale after concat (recommended):
    - raw concat: [latent_raw, age_raw, sex_bin]
    - final StandardScaler on the concatenated matrix (train fit -> test transform)
    - (age MinMax skipped)

Runs `repeated_stratified_shuffle_eval.py` twice per latent source and merges summaries.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _read_summary(run_root: Path, latent_source: str, pooling: str, classifier: str) -> pd.DataFrame:
    p = run_root / latent_source / pooling / classifier / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _run_one(
    *,
    python: str,
    script: Path,
    out_dir: str,
    run_tag: str,
    labels_csv: list[str],
    latent_source: str,
    npz_dir: str,
    pooling: str,
    classifier: str,
    n_splits: int,
    test_size: float,
    random_state: int,
    final_rescale_after_concat: bool,
) -> Path:
    cmd = [
        python,
        str(script),
        "--labels_csv",
        os.path.abspath(labels_csv[0]),
    ]
    for p in labels_csv[1:]:
        cmd.extend(["--labels_csv", os.path.abspath(p)])

    cmd += [
        "--latent_source",
        latent_source,
        os.path.abspath(npz_dir),
        "--pooling",
        pooling,
        "--classifier",
        classifier,
        "--n_splits",
        str(int(n_splits)),
        "--test_size",
        str(float(test_size)),
        "--random_state",
        str(int(random_state)),
        "--out_dir",
        os.path.abspath(out_dir),
        "--run_tag",
        run_tag,
        "--concat_age_sex",
    ]
    if final_rescale_after_concat:
        cmd.append("--final_rescale_after_concat")

    subprocess.check_call(cmd)
    return Path(out_dir) / f"run_{run_tag}"


def main() -> None:
    here = Path(__file__).resolve().parent
    script = here / "repeated_stratified_shuffle_eval.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    ap = argparse.ArgumentParser(description="Compare concat scaling strategies (A vs B) for plainAE/MONAI.")
    ap.add_argument("--labels_csv", action="append", required=True, help="Repeat: train.csv/val.csv/test.csv")
    ap.add_argument("--out_dir", required=True, help="Base output directory")
    ap.add_argument("--pooling", default="std", choices=["mean", "std", "mean_std", "cluster_hist"])
    ap.add_argument("--classifier", default="logistic", choices=["logistic", "linear_svc", "rbf_svc"])
    ap.add_argument("--n_splits", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--latent_source",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ_DIR"),
        required=True,
        help="Repeat: latent_source name and NPZ_DIR",
    )
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for src_name, npz_dir in args.latent_source:
        src_name = str(src_name)
        npz_dir = os.path.abspath(str(npz_dir))

        tag_a = f"{ts}_{src_name}_{args.pooling}_{args.classifier}_A_concat_then_append"
        tag_b = f"{ts}_{src_name}_{args.pooling}_{args.classifier}_B_final_rescale"

        run_a = _run_one(
            python=args.python,
            script=script,
            out_dir=out_dir,
            run_tag=tag_a,
            labels_csv=list(args.labels_csv),
            latent_source=src_name,
            npz_dir=npz_dir,
            pooling=str(args.pooling),
            classifier=str(args.classifier),
            n_splits=int(args.n_splits),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            final_rescale_after_concat=False,
        )
        run_b = _run_one(
            python=args.python,
            script=script,
            out_dir=out_dir,
            run_tag=tag_b,
            labels_csv=list(args.labels_csv),
            latent_source=src_name,
            npz_dir=npz_dir,
            pooling=str(args.pooling),
            classifier=str(args.classifier),
            n_splits=int(args.n_splits),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            final_rescale_after_concat=True,
        )

        df_a = _read_summary(run_a, src_name, str(args.pooling), str(args.classifier))
        df_b = _read_summary(run_b, src_name, str(args.pooling), str(args.classifier))
        df_a.insert(0, "variant", "A_latent_scaled_then_age_minmax_append")
        df_b.insert(0, "variant", "B_raw_concat_then_final_standardscaler")
        rows.append(df_a)
        rows.append(df_b)

    out_csv = Path(out_dir) / f"compare_age_sex_concat_scaling_{ts}.csv"
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()

