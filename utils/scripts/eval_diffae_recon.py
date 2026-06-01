#!/usr/bin/env python3
"""
Evaluate DiffAE reconstruction quality on CT volumes.

This script reconstructs 3D volumes slice-wise using a trained DiffAE model,
then computes:
  1) slice-level metrics (MAE, MSE, SSIM, edge diff),
  2) intra-volume continuity (adjacent-slice deltas),
  3) 3D structure proxies (histogram similarity, COM stability),
  4) failure cases (worst slices).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


def _add_paths() -> Tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ae_root = os.path.dirname(script_dir)
    diffae_root = os.path.join(ae_root, "diffae-master")
    if ae_root not in sys.path:
        sys.path.insert(0, ae_root)
    if diffae_root not in sys.path:
        sys.path.insert(0, diffae_root)
    return ae_root, diffae_root


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DiffAE reconstruction evaluation (CT)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to DiffAE Lightning .ckpt")
    p.add_argument("--tts_root", type=str, default=None, help="Root dir for TTS dataset")
    p.add_argument("--normal_root", type=str, default=None, help="Root dir for Normal dataset")
    p.add_argument("--metadata_csv", type=str, default=None, help="Combined metadata CSV")
    p.add_argument("--split_csv", type=str, default=None, help="Split CSV with patient IDs")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=1)
    p.add_argument("--style_ch", type=int, default=512)
    p.add_argument("--apply_lv_mask", action="store_true")
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz")
    p.add_argument("--batch_size", type=int, default=32, help="Slices per batch")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--T", type=int, default=100, help="Diffusion steps for reconstruction")
    p.add_argument("--use_ema", action="store_true", help="Use EMA weights (recommended)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--slice_stride", type=int, default=1, help="Use every n-th slice")
    p.add_argument("--max_volumes", type=int, default=0, help="Limit number of volumes (0=all)")
    p.add_argument("--max_failures", type=int, default=25, help="Number of worst slices to save")
    p.add_argument("--hist_bins", type=int, default=64)
    p.add_argument("--lv_thresh", type=float, default=0.0, help="Threshold for LV area in [-1,1]")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="val")
    return p


def sobel_kernels(device: torch.device, dtype: torch.dtype):
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    return kx, ky


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W) in [-1,1]
    kx, ky = sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def compute_ssim(img: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    # Expect img/recon in [-1,1], convert to [0,1]
    from ssim import ssim as ssim_fn
    img01 = (img + 1.0) / 2.0
    recon01 = (recon + 1.0) / 2.0
    return ssim_fn(img01, recon01, size_average=False)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def center_of_mass_2d(img: np.ndarray) -> Optional[Tuple[float, float]]:
    # img in [0,1], shape (H, W)
    mass = img.sum()
    if mass <= 1e-8:
        return None
    ys, xs = np.indices(img.shape)
    cy = float((ys * img).sum() / mass)
    cx = float((xs * img).sum() / mass)
    return cy, cx


def make_output_dirs(out_dir: str) -> Dict[str, str]:
    paths = {
        "root": out_dir,
        "plots": os.path.join(out_dir, "plots"),
        "failures": os.path.join(out_dir, "failures"),
        "curves": os.path.join(out_dir, "curves"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


@dataclass
class SliceMetric:
    patient_id: str
    label: int
    slice_idx: int
    mae: float
    mse: float
    ssim: float
    grad_mae: float


def main() -> None:
    _add_paths()
    args = build_argparser().parse_args()

    from data.dataset import CTROIVolumeDataset
    from experiment import LitModel
    from templates import ffhq128_autoenc_base

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_paths = make_output_dirs(args.out_dir)

    # Build dataset (TTS + Normal)
    expected_shape = tuple(args.expected_shape)
    ds_list = []
    if args.tts_root:
        ds_list.append(
            CTROIVolumeDataset(
                root_dir=args.tts_root,
                ct_suffix=args.ct_suffix,
                mask_filename=args.mask_filename,
                apply_lv_mask=args.apply_lv_mask,
                expected_shape_xyz=expected_shape,
                metadata_csv=args.metadata_csv,
                include_patient_csv=args.split_csv,
                return_patient_id=True,
            )
        )
    if args.normal_root:
        ds_list.append(
            CTROIVolumeDataset(
                root_dir=args.normal_root,
                ct_suffix=args.ct_suffix,
                mask_filename=args.mask_filename,
                apply_lv_mask=args.apply_lv_mask,
                expected_shape_xyz=expected_shape,
                metadata_csv=args.metadata_csv,
                include_patient_csv=args.split_csv,
                return_patient_id=True,
            )
        )
    if not ds_list:
        raise ValueError("Provide at least one of --tts_root or --normal_root")
    dataset = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]

    # Build model config and load checkpoint
    conf = ffhq128_autoenc_base()
    conf.data_name = "ct_slices"
    conf.img_size = args.img_size
    conf.in_channels = args.in_channels
    conf.out_channels = args.out_channels
    conf.style_ch = args.style_ch
    conf.net_beatgans_embed_channels = args.style_ch
    conf.make_model_conf()

    lit = LitModel(conf)
    state = torch.load(args.ckpt, map_location="cpu")
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.eval()
    lit.to(device)

    model = lit.ema_model if args.use_ema else lit.model
    model.eval()
    sampler = conf._make_diffusion_conf(T=args.T).make_sampler()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    slice_rows: List[SliceMetric] = []
    vol_rows: List[Dict[str, float]] = []
    failure_pool: List[Tuple[float, Dict]] = []

    max_vols = args.max_volumes if args.max_volumes > 0 else None

    for idx, batch in enumerate(loader):
        if max_vols is not None and idx >= max_vols:
            break

        x = batch["x"].squeeze(0)  # (1, Z, Y, X)
        label = int(batch.get("y", torch.tensor(-1)).item())
        pid = batch.get("patient_id", [f"vol_{idx}"])[0]

        # Convert to slices: (Z, 1, H, W)
        z_dim = x.shape[1]
        slice_indices = list(range(0, z_dim, args.slice_stride))
        slices = x[:, slice_indices, :, :].permute(1, 0, 2, 3).contiguous()
        slices = slices.to(device)

        recon_slices = []
        with torch.no_grad():
            for start in range(0, len(slices), args.batch_size):
                end = start + args.batch_size
                x_start = slices[start:end]
                noise = torch.randn_like(x_start)
                # Explicitly use the encoder condition tensor (avoid passing dict)
                cond = model.encode(x_start)["cond"]
                recon = sampler.sample(model=model, noise=noise, model_kwargs={"cond": cond})
                recon_slices.append(recon)
        recon_slices = torch.cat(recon_slices, dim=0)

        # Slice-level metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(recon_slices - slices), dim=[1, 2, 3])
            mse = torch.mean((recon_slices - slices) ** 2, dim=[1, 2, 3])
            ssim = compute_ssim(slices, recon_slices)
            grad = grad_mag(slices)
            grad_recon = grad_mag(recon_slices)
            grad_mae = torch.mean(torch.abs(grad_recon - grad), dim=[1, 2, 3])

        for i, z in enumerate(slice_indices):
            slice_rows.append(
                SliceMetric(
                    patient_id=pid,
                    label=label,
                    slice_idx=int(z),
                    mae=float(mae[i].item()),
                    mse=float(mse[i].item()),
                    ssim=float(ssim[i].item()),
                    grad_mae=float(grad_mae[i].item()),
                )
            )

        # Volume-level arrays for step 2/3
        orig_vol = slices.detach().cpu().numpy()  # (Z,1,H,W)
        recon_vol = recon_slices.detach().cpu().numpy()

        # Adjacent-slice deltas
        def slice_delta(vol: np.ndarray) -> np.ndarray:
            diffs = []
            for i in range(len(vol) - 1):
                d = vol[i + 1] - vol[i]
                diffs.append(np.mean(d * d))
            return np.array(diffs, dtype=np.float32)

        delta_orig = slice_delta(orig_vol)
        delta_recon = slice_delta(recon_vol)
        if len(delta_orig) > 1:
            corr = float(np.corrcoef(delta_orig, delta_recon)[0, 1])
        else:
            corr = float("nan")
        delta_mae = float(np.mean(np.abs(delta_orig - delta_recon)))

        # Histogram similarity
        hist_bins = args.hist_bins
        orig_flat = orig_vol.reshape(-1)
        recon_flat = recon_vol.reshape(-1)
        orig_hist, _ = np.histogram(orig_flat, bins=hist_bins, range=(-1.0, 1.0), density=True)
        recon_hist, _ = np.histogram(recon_flat, bins=hist_bins, range=(-1.0, 1.0), density=True)
        js = js_divergence(orig_hist, recon_hist)

        # Center of mass stability
        com_diffs = []
        for i in range(len(orig_vol)):
            o = (orig_vol[i, 0] + 1.0) / 2.0
            r = (recon_vol[i, 0] + 1.0) / 2.0
            com_o = center_of_mass_2d(o)
            com_r = center_of_mass_2d(r)
            if com_o is None or com_r is None:
                continue
            dy = com_o[0] - com_r[0]
            dx = com_o[1] - com_r[1]
            com_diffs.append(math.sqrt(dy * dy + dx * dx))
        com_rmse = float(np.mean(com_diffs)) if com_diffs else float("nan")

        # LV area proxy (optional mask threshold on intensity within mask)
        lv_area_corr = float("nan")
        lv_area_rmse = float("nan")
        if args.mask_filename and args.apply_lv_mask:
            try:
                import nibabel as nib
                ct_path = batch.get("ct_path", [None])[0]
                if ct_path is not None:
                    mask_path = os.path.join(os.path.dirname(ct_path), args.mask_filename)
                    if os.path.exists(mask_path):
                        mask_xyz = nib.load(mask_path).get_fdata().astype(np.float32)
                        # (X,Y,Z) -> (Z,Y,X)
                        mask = np.transpose(mask_xyz, (2, 1, 0))
                        mask = mask[slice_indices]
                        mask = (mask > 0).astype(np.float32)
                        orig_area = []
                        recon_area = []
                        for i in range(len(orig_vol)):
                            m = mask[i]
                            o = orig_vol[i, 0]
                            r = recon_vol[i, 0]
                            orig_area.append(float(((o > args.lv_thresh) * m).sum()))
                            recon_area.append(float(((r > args.lv_thresh) * m).sum()))
                        orig_area = np.array(orig_area)
                        recon_area = np.array(recon_area)
                        if len(orig_area) > 1:
                            lv_area_corr = float(np.corrcoef(orig_area, recon_area)[0, 1])
                        lv_area_rmse = float(np.sqrt(np.mean((orig_area - recon_area) ** 2)))
            except Exception:
                pass

        # Aggregate volume metrics
        vol_rows.append(
            {
                "patient_id": pid,
                "label": label,
                "n_slices": len(slices),
                "mae_mean": float(mae.mean().item()),
                "mae_std": float(mae.std().item()),
                "mse_mean": float(mse.mean().item()),
                "mse_std": float(mse.std().item()),
                "ssim_mean": float(ssim.mean().item()),
                "ssim_std": float(ssim.std().item()),
                "grad_mae_mean": float(grad_mae.mean().item()),
                "delta_corr": corr,
                "delta_mae": delta_mae,
                "hist_js": js,
                "com_rmse": com_rmse,
                "lv_area_corr": lv_area_corr,
                "lv_area_rmse": lv_area_rmse,
            }
        )

        # Save delta curves
        np.savez(
            os.path.join(out_paths["curves"], f"{pid}_delta_curves.npz"),
            delta_orig=delta_orig,
            delta_recon=delta_recon,
        )

        # Failure pool for top errors
        for i, z in enumerate(slice_indices):
            failure_pool.append(
                (float(mse[i].item()), {"pid": pid, "z": int(z), "orig": orig_vol[i, 0], "recon": recon_vol[i, 0]})
            )

    # Save CSVs
    import csv

    slice_csv = os.path.join(out_paths["root"], f"{args.split_name}_slice_metrics.csv")
    with open(slice_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "slice_idx", "mae", "mse", "ssim", "grad_mae"])
        for r in slice_rows:
            writer.writerow([r.patient_id, r.label, r.slice_idx, r.mae, r.mse, r.ssim, r.grad_mae])

    vol_csv = os.path.join(out_paths["root"], f"{args.split_name}_volume_metrics.csv")
    with open(vol_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(vol_rows[0].keys()) if vol_rows else [])
        for r in vol_rows:
            writer.writerow([r[k] for k in r.keys()])

    # Save top failure slices
    if plt is not None and failure_pool:
        failure_pool.sort(key=lambda x: x[0], reverse=True)
        worst = failure_pool[: args.max_failures]
        for rank, (err, info) in enumerate(worst):
            orig = info["orig"]
            recon = info["recon"]
            diff = np.abs(recon - orig)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(orig, cmap="gray")
            axes[0].set_title("orig")
            axes[1].imshow(recon, cmap="gray")
            axes[1].set_title("recon")
            axes[2].imshow(diff, cmap="hot")
            axes[2].set_title("abs diff")
            fig.suptitle(f"{info['pid']} z={info['z']} mse={err:.6f}")
            out_path = os.path.join(out_paths["failures"], f"{rank:03d}_{info['pid']}_z{info['z']}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Simple distribution plots (Normal vs TTS)
    if plt is not None and slice_rows:
        labels = np.array([r.label for r in slice_rows])
        for metric in ["mae", "mse", "ssim", "grad_mae"]:
            vals = np.array([getattr(r, metric) for r in slice_rows])
            fig, ax = plt.subplots(figsize=(6, 4))
            if sns is not None:
                sns.violinplot(x=labels, y=vals, ax=ax)
                ax.set_xticklabels(["Normal", "TTS"])
            else:
                ax.boxplot([vals[labels == 0], vals[labels == 1]], labels=["Normal", "TTS"])
            ax.set_title(f"{metric} distribution")
            ax.set_xlabel("label")
            ax.set_ylabel(metric)
            out_path = os.path.join(out_paths["plots"], f"{args.split_name}_{metric}_dist.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Save summary
    summary = {
        "split": args.split_name,
        "n_volumes": len(vol_rows),
        "n_slices": len(slice_rows),
        "ckpt": args.ckpt,
        "T": args.T,
        "use_ema": args.use_ema,
        "slice_stride": args.slice_stride,
        "metrics": {
            "mse_mean": float(np.mean([r.mse_mean for r in vol_rows])) if vol_rows else None,
            "ssim_mean": float(np.mean([r.ssim_mean for r in vol_rows])) if vol_rows else None,
            "delta_corr_mean": float(np.nanmean([r["delta_corr"] for r in vol_rows])) if vol_rows else None,
            "hist_js_mean": float(np.mean([r["hist_js"] for r in vol_rows])) if vol_rows else None,
        },
    }
    with open(os.path.join(out_paths["root"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved slice metrics: {slice_csv}")
    print(f"✅ Saved volume metrics: {vol_csv}")
    print(f"✅ Outputs in: {out_paths['root']}")


if __name__ == "__main__":
    main()
