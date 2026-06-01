#!/usr/bin/env python3
"""
Evaluate center-slice jitter impact on delta and plane AUCs.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError("scikit-learn is required for ROC/AUC. Install via pip.") from exc

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)

from data.dataset import CTROIVolumeDataset
from models import TriplaneFlowMatcher, compute_triplane_anomaly_scores


def parse_timesteps(ts: str) -> List[float]:
    return [float(x.strip()) for x in ts.split(",") if x.strip()]


def build_loader(root_dir: str, include_csv: str, expected_shape, batch_size: int, num_workers: int) -> DataLoader:
    ds = CTROIVolumeDataset(
        root_dir=root_dir,
        expected_shape_xyz=tuple(expected_shape),
        include_patient_csv=include_csv,
        return_patient_id=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jitter analysis for triplane FM eval")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_csv", type=str, required=True)
    p.add_argument("--tts_csv", type=str, required=True)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--timesteps", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--n_noise", type=int, default=1)
    p.add_argument("--score_mode", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--cos_eps", type=float, default=1e-8)
    p.add_argument("--offsets", type=str, default="-2,-1,0,1,2")
    p.add_argument("--out_dir", type=str, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = parse_timesteps(args.timesteps)
    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]

    ckpt = torch.load(args.ckpt, map_location=device)
    model_args = ckpt.get("args", {})
    model = TriplaneFlowMatcher(
        in_channels=1,
        feat_channels=int(model_args.get("feat_channels", 64)),
        base_channels=int(model_args.get("base_channels", 64)),
        time_dim=int(model_args.get("time_dim", 128)),
        shared_encoder=bool(model_args.get("shared_encoder", False)),
        shared_velocity=bool(model_args.get("shared_velocity", False)),
        plane_mode=str(model_args.get("plane_mode", "slab")),
        slab_k=int(model_args.get("slab_k", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader_normal = build_loader(
        args.normal_root, args.normal_csv, args.expected_shape, args.batch_size, args.num_workers
    )
    loader_tts = build_loader(
        args.tts_root, args.tts_csv, args.expected_shape, args.batch_size, args.num_workers
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "jitter_auc.csv")
    out_json = os.path.join(args.out_dir, "jitter_auc.json")

    rows = []
    for off in offsets:
        scores = []
        labels = []
        for label, loader in [(0, loader_normal), (1, loader_tts)]:
            for batch in loader:
                x = batch["x"].to(device)
                s = compute_triplane_anomaly_scores(
                    model,
                    x,
                    timesteps,
                    device=device,
                    n_noise=args.n_noise,
                    score_mode=args.score_mode,
                    cos_eps=args.cos_eps,
                    offsets=(off, off, off),
                )
                s_xy = s["xy"].detach().cpu().numpy()
                s_xz = s["xz"].detach().cpu().numpy()
                s_yz = s["yz"].detach().cpu().numpy()
                s_delta = 0.5 * (s_xz + s_yz) - s_xy

                for i in range(len(s_xy)):
                    scores.append((s_xy[i], s_xz[i], s_yz[i], s_delta[i]))
                    labels.append(label)

        y = np.array(labels, dtype=int)
        s_xy = np.array([s[0] for s in scores], dtype=float)
        s_xz = np.array([s[1] for s in scores], dtype=float)
        s_yz = np.array([s[2] for s in scores], dtype=float)
        s_delta = np.array([s[3] for s in scores], dtype=float)
        s_avg = 0.5 * (s_xz + s_yz)

        def _safe_auc(y_true: np.ndarray, s_val: np.ndarray) -> float:
            try:
                return float(roc_auc_score(y_true, s_val))
            except ValueError:
                return float("nan")

        row = {
            "offset": off,
            "auc_xy": _safe_auc(y, s_xy),
            "auc_xz": _safe_auc(y, s_xz),
            "auc_yz": _safe_auc(y, s_yz),
            "auc_delta": _safe_auc(y, s_delta),
            "auc_yz_only": _safe_auc(y, s_yz),
            "auc_xz_yz_avg": _safe_auc(y, s_avg),
        }
        rows.append(row)

    with open(out_csv, "w") as f:
        f.write("offset,auc_xy,auc_xz,auc_yz,auc_delta,auc_yz_only,auc_xz_yz_avg\n")
        for r in rows:
            f.write(
                f"{r['offset']},{r['auc_xy']},{r['auc_xz']},{r['auc_yz']},"
                f"{r['auc_delta']},{r['auc_yz_only']},{r['auc_xz_yz_avg']}\n"
            )

    with open(out_json, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"✅ Jitter analysis complete: {out_csv}")


if __name__ == "__main__":
    main()
