#!/usr/bin/env python3
"""
eval_triplane_flow_matching.py
- Evaluate triplane flow matching model (Normal vs TTS).
- Compute plane-wise scores and weighted patient score.
- Estimate threshold via Youden's J with stratified bootstrap.
"""

import os
import sys
import json
import argparse
from typing import Iterable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError as exc:
    raise ImportError("scikit-learn is required for ROC/AUC. Install via pip.") from exc

# Add AE_TTS root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)

from data.dataset import CTROIVolumeDataset
from models import (
    TriplaneFlowMatcher,
    PlaneScoreNormalizer,
    compute_triplane_anomaly_scores,
    weighted_patient_score,
)


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


@torch.no_grad()
def collect_plane_scores(
    model: TriplaneFlowMatcher,
    loader: DataLoader,
    timesteps: Iterable[float],
    device: torch.device,
    n_noise: int,
    label: int,
    score_mode: str,
    cos_eps: float,
    center_offset: int,
) -> List[dict]:
    model.eval()
    rows: List[dict] = []
    for batch in loader:
        x = batch["x"].to(device)
        pids = batch["patient_id"]
        scores = compute_triplane_anomaly_scores(
            model,
            x,
            timesteps,
            device=device,
            n_noise=n_noise,
            score_mode=score_mode,
            cos_eps=cos_eps,
            offsets=(center_offset, center_offset, center_offset),
        )

        for i, pid in enumerate(pids):
            rows.append({
                "patient_id": str(pid),
                "label": int(label),
                "S_xy": float(scores["xy"][i].item()),
                "S_xz": float(scores["xz"][i].item()),
                "S_yz": float(scores["yz"][i].item()),
            })
    return rows


def youden_j_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def stratified_resample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    res_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
    res_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
    return np.concatenate([res_pos, res_neg])


def bootstrap_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    thresholds = []
    sens = []
    spec = []
    aucs = []

    for _ in range(n_boot):
        idx = stratified_resample(y_true, rng)
        ys = y_true[idx]
        ss = y_score[idx]
        thr = youden_j_threshold(ys, ss)
        thresholds.append(thr)

        yhat = (ss >= thr).astype(int)
        tp = np.sum((yhat == 1) & (ys == 1))
        tn = np.sum((yhat == 0) & (ys == 0))
        fp = np.sum((yhat == 1) & (ys == 0))
        fn = np.sum((yhat == 0) & (ys == 1))

        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
        try:
            aucs.append(float(roc_auc_score(ys, ss)))
        except ValueError:
            aucs.append(float("nan"))

    return {
        "thresholds": thresholds,
        "sens": sens,
        "spec": spec,
        "auc": aucs,
    }


def ci(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    return [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 50.0)), float(np.nanpercentile(arr, 97.5))]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate triplane flow matching model")
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--normal_csv", type=str, required=True)
    p.add_argument("--tts_csv", type=str, required=True)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--timesteps", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--n_noise", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--norm_method", type=str, default="meanstd", choices=["meanstd", "medianmad"])
    p.add_argument("--score_mode", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--cos_eps", type=float, default=1e-8)
    p.add_argument("--center_offset", type=int, default=0, help="Center slice offset (applied to z/y/x)")

    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./outputs/triplane_flow_matching_eval")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = parse_timesteps(args.timesteps)

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

    loader_normal = build_loader(
        args.normal_root, args.normal_csv, args.expected_shape, args.batch_size, args.num_workers
    )
    loader_tts = build_loader(
        args.tts_root, args.tts_csv, args.expected_shape, args.batch_size, args.num_workers
    )

    rows = []
    rows_normal = collect_plane_scores(
        model,
        loader_normal,
        timesteps,
        device,
        args.n_noise,
        label=0,
        score_mode=args.score_mode,
        cos_eps=args.cos_eps,
        center_offset=args.center_offset,
    )
    rows_tts = collect_plane_scores(
        model,
        loader_tts,
        timesteps,
        device,
        args.n_noise,
        label=1,
        score_mode=args.score_mode,
        cos_eps=args.cos_eps,
        center_offset=args.center_offset,
    )

    # Fit plane-wise normalizer on NORMAL scores only
    normal_scores = {
        "xy": torch.tensor([r["S_xy"] for r in rows_normal]),
        "xz": torch.tensor([r["S_xz"] for r in rows_normal]),
        "yz": torch.tensor([r["S_yz"] for r in rows_normal]),
    }
    normalizer = PlaneScoreNormalizer(method=args.norm_method)
    normalizer.fit(normal_scores)

    def _apply_norm(rows_in: List[dict]) -> List[dict]:
        s = {
            "xy": torch.tensor([r["S_xy"] for r in rows_in]),
            "xz": torch.tensor([r["S_xz"] for r in rows_in]),
            "yz": torch.tensor([r["S_yz"] for r in rows_in]),
        }
        s_norm = normalizer.normalize(s)
        s_patient = weighted_patient_score(s_norm, alpha=args.alpha, beta=args.beta)
        rows_out = []
        for i, r in enumerate(rows_in):
            r_out = dict(r)
            r_out["S_xy_norm"] = float(s_norm["xy"][i].item())
            r_out["S_xz_norm"] = float(s_norm["xz"][i].item())
            r_out["S_yz_norm"] = float(s_norm["yz"][i].item())
            r_out["S"] = float(s_patient[i].item())
            r_out["S_delta"] = float(0.5 * (r_out["S_xz"] + r_out["S_yz"]) - r_out["S_xy"])
            r_out["S_delta_norm"] = float(
                0.5 * (r_out["S_xz_norm"] + r_out["S_yz_norm"]) - r_out["S_xy_norm"]
            )
            rows_out.append(r_out)
        return rows_out

    rows += _apply_norm(rows_normal)
    rows += _apply_norm(rows_tts)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "triplane_scores.csv")
    with open(csv_path, "w") as f:
        f.write("patient_id,label,S_xy,S_xz,S_yz,S_xy_norm,S_xz_norm,S_yz_norm,S,S_delta,S_delta_norm\n")
        for r in rows:
            f.write(
                f"{r['patient_id']},{r['label']},"
                f"{r['S_xy']:.6f},{r['S_xz']:.6f},{r['S_yz']:.6f},"
                f"{r['S_xy_norm']:.6f},{r['S_xz_norm']:.6f},{r['S_yz_norm']:.6f},"
                f"{r['S']:.6f},{r['S_delta']:.6f},{r['S_delta_norm']:.6f}\n"
            )

    y_true = np.array([r["label"] for r in rows], dtype=int)
    y_score = np.array([r["S"] for r in rows], dtype=float)
    roc_auc = float(roc_auc_score(y_true, y_score))
    s_xy = np.array([r["S_xy"] for r in rows], dtype=float)
    s_xz = np.array([r["S_xz"] for r in rows], dtype=float)
    s_yz = np.array([r["S_yz"] for r in rows], dtype=float)
    s_yz_only = s_yz
    s_xz_yz_avg = 0.5 * (s_xz + s_yz)
    s_xy_n = np.array([r["S_xy_norm"] for r in rows], dtype=float)
    s_xz_n = np.array([r["S_xz_norm"] for r in rows], dtype=float)
    s_yz_n = np.array([r["S_yz_norm"] for r in rows], dtype=float)
    s_yz_only_n = s_yz_n
    s_xz_yz_avg_n = 0.5 * (s_xz_n + s_yz_n)
    s_delta = np.array([r["S_delta"] for r in rows], dtype=float)
    s_delta_n = np.array([r["S_delta_norm"] for r in rows], dtype=float)

    def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y, s))
        except ValueError:
            return float("nan")

    plane_auc = {
        "raw": {
            "auc_xy": _safe_auc(y_true, s_xy),
            "auc_xz": _safe_auc(y_true, s_xz),
            "auc_yz": _safe_auc(y_true, s_yz),
            "auc_delta": _safe_auc(y_true, s_delta),
            "auc_yz_only": _safe_auc(y_true, s_yz_only),
            "auc_xz_yz_avg": _safe_auc(y_true, s_xz_yz_avg),
        },
        "norm": {
            "auc_xy": _safe_auc(y_true, s_xy_n),
            "auc_xz": _safe_auc(y_true, s_xz_n),
            "auc_yz": _safe_auc(y_true, s_yz_n),
            "auc_delta": _safe_auc(y_true, s_delta_n),
            "auc_yz_only": _safe_auc(y_true, s_yz_only_n),
            "auc_xz_yz_avg": _safe_auc(y_true, s_xz_yz_avg_n),
        },
    }
    thr = youden_j_threshold(y_true, y_score)

    boot = bootstrap_thresholds(y_true, y_score, n_boot=args.bootstrap, seed=args.seed)
    summary = {
        "n_normal": int(np.sum(y_true == 0)),
        "n_tts": int(np.sum(y_true == 1)),
        "roc_auc": roc_auc,
        "youden_threshold": thr,
        "threshold_ci": ci(boot["thresholds"]),
        "sensitivity_ci": ci(boot["sens"]),
        "specificity_ci": ci(boot["spec"]),
        "auc_ci": ci(boot["auc"]),
        "alpha": args.alpha,
        "beta": args.beta,
        "norm_method": args.norm_method,
        "score_mode": args.score_mode,
        "center_offset": args.center_offset,
        "plane_auc": plane_auc,
        "norm_stats": {
            "mu_xy": float(normalizer.mu_xy.item()),
            "mu_xz": float(normalizer.mu_xz.item()),
            "mu_yz": float(normalizer.mu_yz.item()),
            "sigma_xy": float(normalizer.sigma_xy.item()),
            "sigma_xz": float(normalizer.sigma_xz.item()),
            "sigma_yz": float(normalizer.sigma_yz.item()),
        },
        "timesteps": timesteps,
        "n_noise": args.n_noise,
    }

    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Evaluation complete")
    print(f"Scores: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
