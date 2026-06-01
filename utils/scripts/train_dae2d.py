#!/usr/bin/env python3
"""
train_dae2d.py
- Stage A: Train DAE (encoder + diffusion UNet) for 2D axial slices.
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

# Add AE_TTS root and models to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)
models_dir = os.path.join(ae_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from data.dataset import CTROIVolumeDataset
from dae2d import DAE2D, DiffusionSchedule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DAE2D (slice-level diffusion autoencoder)")
    p.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    p.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    p.add_argument("--val_csv", type=str, default=None, help="Path to val.csv split file (optional)")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected shape (X, Y, Z)")

    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size (volumes)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--amp", action="store_true", help="Enable AMP")

    # Conditioning controls
    p.add_argument("--p_uncond", type=float, default=0.1,
                   help="Probability of dropping conditioning per slice (CFG-style)")
    p.add_argument("--p_uncond_warmup_steps", type=int, default=2000,
                   help="Linearly ramp p_uncond from 0 to --p_uncond over this many steps")
    p.add_argument("--lambda_x0", type=float, default=0.1,
                   help="Weight for auxiliary x0 reconstruction loss")
    p.add_argument("--lambda_x0_warmup_steps", type=int, default=2000,
                   help="Linearly ramp lambda_x0 from 0 to --lambda_x0 over this many steps")

    p.add_argument("--timesteps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--beta_start", type=float, default=1e-4, help="Beta schedule start")
    p.add_argument("--beta_end", type=float, default=2e-2, help="Beta schedule end")
    p.add_argument("--base_ch", type=int, default=64, help="Base channel width")
    p.add_argument("--y_sem_dim", type=int, default=512, help="Semantic vector dim")

    p.add_argument("--out_dir", type=str, default="./outputs/train_dae2d", help="Output directory")
    p.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0=off)")
    return p


def flatten_slices(x: torch.Tensor) -> torch.Tensor:
    """
    (B, 1, D, H, W) -> (B*D, 1, H, W)
    """
    b, c, d, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(b * d, c, h, w)


def run_epoch(
    model: DAE2D,
    loader: DataLoader,
    device: torch.device,
    loss_diff_fn: nn.Module,
    loss_x0_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    p_uncond: float = 0.1,
    lambda_x0: float = 0.1,
    p_uncond_warmup_steps: int = 2000,
    lambda_x0_warmup_steps: int = 2000,
    step_offset: int = 0,
    is_val: bool = False,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.0
    n = 0

    for step_idx, batch in enumerate(loader):
        # Batch is volumes; we flatten into per-slice batch for diffusion training
        x = batch["x"].to(device)  # (B, 1, D, H, W)
        x0 = flatten_slices(x)     # (B*D, 1, H, W)

        bsz = x0.shape[0]
        # Sample a random timestep for each slice
        t = torch.randint(0, model.schedule.timesteps, (bsz,), device=device)
        noise = torch.randn_like(x0)
        t_norm = t.float() / max(1, model.schedule.timesteps - 1)

        # Linear warm-up schedules (train only)
        global_step = step_offset + step_idx
        if not is_val:
            if p_uncond_warmup_steps > 0:
                p_uncond_now = min(p_uncond, p_uncond * (global_step / p_uncond_warmup_steps))
            else:
                p_uncond_now = p_uncond
            if lambda_x0_warmup_steps > 0:
                lambda_x0_now = min(lambda_x0, lambda_x0 * (global_step / lambda_x0_warmup_steps))
            else:
                lambda_x0_now = lambda_x0
        else:
            # Validation uses deterministic conditioning and full lambda_x0
            p_uncond_now = 0.0
            lambda_x0_now = lambda_x0

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Encode semantic vector
            y_sem = model.encoder(x0)

            # Classifier-free guidance style conditioning dropout (train only)
            # We hard-zero some embeddings to teach the denoiser to rely on y_sem
            if not is_val and p_uncond_now > 0:
                drop = (torch.rand(bsz, device=device) < p_uncond_now)
                y_sem_used = y_sem.clone()
                y_sem_used[drop] = 0.0
            else:
                # Validation uses deterministic full conditioning
                y_sem_used = y_sem

            # Forward diffusion then denoise
            x_t = model.q_sample(x0, t, noise)
            eps_pred = model.denoiser(x_t, t_norm, y_sem_used)

            # Diffusion loss (MSE on noise)
            loss_diff = loss_diff_fn(eps_pred, noise)

            # x0 reconstruction loss (L1)
            x0_hat = model.predict_x0(x_t, t, eps_pred)
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
            loss_x0 = loss_x0_fn(x0_hat, x0)

            loss = loss_diff + lambda_x0_now * loss_x0

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if step_idx == 0:
            # Diagnostics to check conditioning usage (same x_t, t_norm, noise)
            with torch.no_grad():
                y_var = y_sem.var(dim=0).mean().item()
                rand_idx = torch.randperm(bsz, device=device)
                eps_real = model.denoiser(x_t, t_norm, y_sem)
                eps_used = model.denoiser(x_t, t_norm, y_sem_used)
                eps_rand = model.denoiser(x_t, t_norm, y_sem[rand_idx])
                eps_zero = model.denoiser(x_t, t_norm, torch.zeros_like(y_sem))

                loss_real = loss_diff_fn(eps_real, noise).item()
                loss_used = loss_diff_fn(eps_used, noise).item()
                loss_rand = loss_diff_fn(eps_rand, noise).item()
                loss_zero = loss_diff_fn(eps_zero, noise).item()

                # AdaGN magnitude (mean across last block stats)
                adagn_scale, adagn_shift = model.denoiser.get_adagn_stats()

                print(
                    f"  [diag] y_sem_var={y_var:.6f} "
                    f"loss_real={loss_real:.6f} loss_used={loss_used:.6f} "
                    f"loss_rand={loss_rand:.6f} loss_zero={loss_zero:.6f} "
                    f"adagn_scale={adagn_scale if adagn_scale is not None else float('nan'):.6f} "
                    f"adagn_shift={adagn_shift if adagn_shift is not None else float('nan'):.6f} "
                    f"p_uncond={p_uncond_now:.4f} lambda_x0={lambda_x0_now:.4f}"
                )

        total += float(loss.item()) * bsz
        n += bsz

    return total / max(n, 1)


def build_dataset(root_dir: str, metadata_csv: str, expected_xyz: Tuple[int, int, int], include_csv: str) -> CTROIVolumeDataset:
    return CTROIVolumeDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        expected_shape_xyz=tuple(expected_xyz),
        include_patient_csv=include_csv,
        return_patient_id=True,
    )


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_shape_xyz = tuple(args.expected_shape)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train_csv not found: {args.train_csv}")

    ds_train = build_dataset(args.normal_root, args.metadata_csv, expected_shape_xyz, args.train_csv)
    ds_val = None
    if args.val_csv:
        if not os.path.exists(args.val_csv):
            raise FileNotFoundError(f"val_csv not found: {args.val_csv}")
        ds_val = build_dataset(args.normal_root, args.metadata_csv, expected_shape_xyz, args.val_csv)

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

    schedule = DiffusionSchedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    model = DAE2D(
        in_ch=1,
        y_sem_dim=args.y_sem_dim,
        base_ch=args.base_ch,
        schedule=schedule,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Fixed losses: diffusion=MSE, x0=L1
    loss_diff_fn = nn.MSELoss()
    loss_x0_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    os.makedirs(args.out_dir, exist_ok=True)
    loss_csv = os.path.join(args.out_dir, "loss_history.csv")
    best_ckpt = os.path.join(args.out_dir, "encoder_checkpoint.pth")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        step_offset = (epoch - 1) * len(loader_train)
        train_loss = run_epoch(
            model,
            loader_train,
            device,
            loss_diff_fn,
            loss_x0_fn,
            optimizer,
            scaler,
            p_uncond=args.p_uncond,
            lambda_x0=args.lambda_x0,
            p_uncond_warmup_steps=args.p_uncond_warmup_steps,
            lambda_x0_warmup_steps=args.lambda_x0_warmup_steps,
            step_offset=step_offset,
            is_val=False,
        )
        val_loss = None
        if loader_val is not None:
            # Validation is deterministic: p_uncond = 0
            val_loss = run_epoch(
                model,
                loader_val,
                device,
                loss_diff_fn,
                loss_x0_fn,
                optimizer=None,
                scaler=None,
                p_uncond=0.0,
                lambda_x0=args.lambda_x0,
                p_uncond_warmup_steps=0,
                lambda_x0_warmup_steps=0,
                step_offset=0,
                is_val=True,
            )
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_loss:.6f}" +
              (f" val={val_loss:.6f}" if val_loss is not None else ""))

        history.append((epoch, train_loss, val_loss))

        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss

        if is_best:
            torch.save({
                "encoder": model.encoder.state_dict(),
                "denoiser": model.denoiser.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": best_val,
            }, best_ckpt)
            print(f"💾 saved best checkpoint: {best_ckpt}")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                "encoder": model.encoder.state_dict(),
                "denoiser": model.denoiser.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            }, ckpt_path)

        with open(loss_csv, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for e, tr, va in history:
                f.write(f"{e},{tr},{'' if va is None else va}\n")

    print("✅ Training complete!")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
