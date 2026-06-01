#!/usr/bin/env python3
"""
train_triplane_flow_matching.py
- One-class training on Normal CT volumes using triplane flow matching.
"""

import os
import sys
import argparse
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add AE_TTS root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)

from data.dataset import CTROIVolumeDataset
from models import TriplaneFlowMatcher, compute_triplane_flow_matching_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train triplane flow matching (one-class)")
    p.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    p.add_argument("--train_csv", type=str, required=True, help="Train split CSV (IDs)")
    p.add_argument("--val_csv", type=str, default=None, help="Val split CSV (IDs)")
    p.add_argument("--metadata_csv", type=str, default=None, help="Optional metadata CSV (for ID filtering)")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected shape (X, Y, Z)")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stop", type=int, default=0,
                   help="Stop if val loss doesn't improve for N epochs (0 to disable)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable AMP")

    # Model config
    p.add_argument("--feat_channels", type=int, default=64)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--shared_encoder", action="store_true", help="Share encoder across planes (ablation)")
    p.add_argument("--shared_velocity", action="store_true", help="Share velocity network across planes (ablation)")
    p.add_argument("--plane_mode", type=str, default="slab", choices=["slab", "center"])
    p.add_argument("--slab_k", type=int, default=2, help="Center±k slab for plane extraction")
    p.add_argument("--t_eps", type=float, default=1e-3, help="Sample t in (eps, 1-eps)")

    # Loss weights per plane (optional)
    p.add_argument("--w_xy", type=float, default=1.0)
    p.add_argument("--w_xz", type=float, default=1.0)
    p.add_argument("--w_yz", type=float, default=1.0)

    p.add_argument("--out_dir", type=str, default="./outputs/triplane_flow_matching")
    p.add_argument("--save_every", type=int, default=0)
    return p


def build_dataset(
    root_dir: str,
    expected_xyz: Tuple[int, int, int],
    include_csv: str,
    metadata_csv: Optional[str] = None,
) -> CTROIVolumeDataset:
    return CTROIVolumeDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        expected_shape_xyz=tuple(expected_xyz),
        include_patient_csv=include_csv,
        return_patient_id=True,
    )


def run_epoch(
    model: TriplaneFlowMatcher,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    weights: Optional[dict] = None,
    t_eps: float = 1e-3,
) -> Tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = 0.0
    total_xy = 0.0
    total_xz = 0.0
    total_yz = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)  # (B, 1, D, H, W)
        b = x.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss, info = compute_triplane_flow_matching_loss(model, x, device, weights=weights, eps=t_eps)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total += float(loss.item()) * b
        total_xy += float(info["loss_xy"]) * b
        total_xz += float(info["loss_xz"]) * b
        total_yz += float(info["loss_yz"]) * b
        n += b

    avg = total / max(n, 1)
    avg_xy = total_xy / max(n, 1)
    avg_xz = total_xz / max(n, 1)
    avg_yz = total_yz / max(n, 1)
    return avg, {"xy": avg_xy, "xz": avg_xz, "yz": avg_yz}


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_shape_xyz = tuple(args.expected_shape)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train_csv not found: {args.train_csv}")

    ds_train = build_dataset(args.normal_root, expected_shape_xyz, args.train_csv, args.metadata_csv)
    ds_val = None
    if args.val_csv:
        if not os.path.exists(args.val_csv):
            raise FileNotFoundError(f"val_csv not found: {args.val_csv}")
        ds_val = build_dataset(args.normal_root, expected_shape_xyz, args.val_csv, args.metadata_csv)

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_val = None
    if ds_val is not None:
        loader_val = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = TriplaneFlowMatcher(
        in_channels=1,
        feat_channels=args.feat_channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        shared_encoder=args.shared_encoder,
        shared_velocity=args.shared_velocity,
        plane_mode=args.plane_mode,
        slab_k=args.slab_k,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    os.makedirs(args.out_dir, exist_ok=True)
    best_ckpt = os.path.join(args.out_dir, "triplane_fm_best.pth")
    last_ckpt = os.path.join(args.out_dir, "triplane_fm_last.pth")
    loss_csv = os.path.join(args.out_dir, "loss_history.csv")

    weights = {"xy": args.w_xy, "xz": args.w_xz, "yz": args.w_yz}

    best_val = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_plane = run_epoch(
            model, loader_train, device, optimizer=optimizer, scaler=scaler, weights=weights, t_eps=args.t_eps
        )
        val_loss = None
        val_plane = None
        if loader_val is not None:
            val_loss, val_plane = run_epoch(
                model, loader_val, device, optimizer=None, scaler=None, weights=weights, t_eps=args.t_eps
            )

        train_str = (f"[epoch {epoch:03d}/{args.epochs}] "
                     f"train={train_loss:.6f} "
                     f"(xy={train_plane['xy']:.6f}, xz={train_plane['xz']:.6f}, yz={train_plane['yz']:.6f})")
        if val_loss is not None and val_plane is not None:
            val_str = (f" val={val_loss:.6f} "
                       f"(xy={val_plane['xy']:.6f}, xz={val_plane['xz']:.6f}, yz={val_plane['yz']:.6f})")
        else:
            val_str = ""
        print(train_str + val_str)

        history.append((epoch, train_loss, val_loss, train_plane, val_plane))

        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": best_val,
            }, best_ckpt)
            print(f"💾 saved best checkpoint: {best_ckpt}")
        elif val_loss is not None and args.early_stop > 0:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"⏹️ Early stopping: no improvement for {args.early_stop} epochs.")
                break

        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            }, ckpt_path)

        with open(loss_csv, "w") as f:
            f.write(
                "epoch,train_loss,train_loss_xy,train_loss_xz,train_loss_yz,"
                "val_loss,val_loss_xy,val_loss_xz,val_loss_yz\n"
            )
            for e, tr, va, trp, vap in history:
                f.write(
                    f"{e},{tr},{trp['xy']},{trp['xz']},{trp['yz']},"
                    f"{'' if va is None else va},"
                    f"{'' if vap is None else vap['xy']},"
                    f"{'' if vap is None else vap['xz']},"
                    f"{'' if vap is None else vap['yz']}\n"
                )

    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
        "best_val_loss": best_val,
    }, last_ckpt)
    print(f"✅ Training complete! Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
