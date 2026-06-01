#!/usr/bin/env python3
"""
Compare multiple triplane eval summary.json files.

Usage:
  python compare_triplane_summaries.py \
    --summaries a/summary.json b/summary.json c/summary.json \
    --out_csv /path/to/compare.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare triplane eval summary.json files")
    p.add_argument("--summaries", nargs="+", required=True, help="Paths to summary.json files")
    p.add_argument("--out_csv", type=str, default=None, help="Optional output CSV path")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rows = []

    for s in args.summaries:
        path = Path(s)
        with path.open("r") as f:
            data = json.load(f)

        row = {
            "summary_path": str(path),
            "roc_auc": data.get("roc_auc"),
            "auc_ci_med": _get(data, ["auc_ci", 1]),
            "youden_threshold": data.get("youden_threshold"),
            "n_normal": data.get("n_normal"),
            "n_tts": data.get("n_tts"),
            "norm_method": data.get("norm_method"),
            "score_mode": data.get("score_mode", "mse"),
            "n_noise": data.get("n_noise"),
            "timesteps": ",".join([str(x) for x in data.get("timesteps", [])]),
            "auc_xy_raw": _get(data, ["plane_auc", "raw", "auc_xy"]),
            "auc_xz_raw": _get(data, ["plane_auc", "raw", "auc_xz"]),
            "auc_yz_raw": _get(data, ["plane_auc", "raw", "auc_yz"]),
            "auc_delta_raw": _get(data, ["plane_auc", "raw", "auc_delta"]),
            "auc_xy_norm": _get(data, ["plane_auc", "norm", "auc_xy"]),
            "auc_xz_norm": _get(data, ["plane_auc", "norm", "auc_xz"]),
            "auc_yz_norm": _get(data, ["plane_auc", "norm", "auc_yz"]),
            "auc_delta_norm": _get(data, ["plane_auc", "norm", "auc_delta"]),
        }
        rows.append(row)

    # Print a quick summary to stdout
    for r in rows:
        print(
            f"{r['summary_path']} | AUC={r['roc_auc']} | "
            f"xy={r['auc_xy_raw']} xz={r['auc_xz_raw']} yz={r['auc_yz_raw']} "
            f"delta={r['auc_delta_raw']}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            f.write(
                "summary_path,roc_auc,auc_ci_med,youden_threshold,n_normal,n_tts,"
                "norm_method,score_mode,n_noise,timesteps,"
                "auc_xy_raw,auc_xz_raw,auc_yz_raw,auc_delta_raw,"
                "auc_xy_norm,auc_xz_norm,auc_yz_norm,auc_delta_norm\n"
            )
            for r in rows:
                f.write(
                    f"{r['summary_path']},{r['roc_auc']},{r['auc_ci_med']},"
                    f"{r['youden_threshold']},{r['n_normal']},{r['n_tts']},"
                    f"{r['norm_method']},{r['score_mode']},{r['n_noise']},"
                    f"\"{r['timesteps']}\","
                    f"{r['auc_xy_raw']},{r['auc_xz_raw']},{r['auc_yz_raw']},{r['auc_delta_raw']},"
                    f"{r['auc_xy_norm']},{r['auc_xz_norm']},{r['auc_yz_norm']},{r['auc_delta_norm']}\n"
                )


if __name__ == "__main__":
    main()
