
import copy
import os
import sys
import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add stageA shared deps + plain_ae to path
script_dir = os.path.dirname(os.path.abspath(__file__))
stage_a_root = os.path.dirname(script_dir)
ae_root = os.path.dirname(stage_a_root)
shared_root = os.path.join(stage_a_root, "shared")
diffae_root = os.path.join(shared_root, "diffae")
plain_ae_root = script_dir
for p in (shared_root, diffae_root, plain_ae_root, ae_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from plain_ae import PlainAE, plain_ae_ffhq128_ct


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _gaussian_1d(length: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(length, device=device, dtype=dtype) - (length - 1) * 0.5
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / (g.sum() + 1e-8)


def _ssim_window(channels: int, window_size: int, device, dtype) -> torch.Tensor:
    """(C,1,K,K) for depthwise conv."""
    g1d = _gaussian_1d(window_size, 1.5, device, dtype)
    w2d = g1d[:, None] * g1d[None, :]
    w2d = w2d / (w2d.sum() + 1e-8)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    c1: float,
    c2: float,
) -> torch.Tensor:
    """Per-spatial SSIM, then mean over channels -> (B, 1, H, W)."""
    c = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy
    t1 = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    t2 = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_ch = t1 / (t2 + 1e-12)
    return ssim_ch.mean(dim=1, keepdim=True)


def ssim_batch(x: torch.Tensor, y: torch.Tensor, data_range: float, window: torch.Tensor) -> torch.Tensor:
    """
    x, y: (B, C, H, W), same shape. Returns per-image SSIM in [0, 1] as shape (B,).
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = _ssim_map(x, y, window, c1, c2)
    return ssim_map.view(ssim_map.shape[0], -1).mean(dim=1)


class MseSsimLoss(nn.Module):
    """
    total = MSE(pred, target) + ssim_weight * (1 - mean_ssim)
    MSE/SSIM computed in full precision (stable under autocast).
    """

    def __init__(
        self,
        ssim_weight: float = 0.1,
        data_range: float = 2.0,
        in_channels: int = 1,
        window_size: int = 11,
    ):
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.data_range = float(data_range)
        self.in_channels = in_channels
        self.window_size = window_size
        self._ssim_win: torch.Tensor | None = None
        self._ssim_win_device = None

    def _get_window(self, device: torch.device) -> torch.Tensor:
        if self._ssim_win is None or self._ssim_win_device != device:
            self._ssim_win = _ssim_window(self.in_channels, self.window_size, device, torch.float32)
            self._ssim_win_device = device
        return self._ssim_win

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = pred.float(), target.float()
        mse = F.mse_loss(p, t)
        w = self._get_window(p.device)
        ssim_vals = ssim_batch(p, t, self.data_range, w)
        ssim_mean = ssim_vals.mean()
        return mse + self.ssim_weight * (1.0 - ssim_mean)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Plain AE (DiffAE arch, no diffusion)")
    p.add_argument("--ct_root", type=str, required=True, help="Root dir of CT ROI volumes")
    p.add_argument("--metadata_csv", type=str, default="", help="Metadata CSV (optional)")
    p.add_argument("--train_csv", type=str, default="", help="Train split CSV (optional)")
    p.add_argument("--val_csv", type=str, default=None, help="Val split CSV (optional)")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected CT shape (X, Y, Z)")
    p.add_argument("--img_size", type=int, default=128, help="Slice size (H=W)")

    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (slices)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--amp", action="store_true", help="Enable AMP")

    p.add_argument("--in_channels", type=int, default=1, help="Input channels (CT=1)")
    p.add_argument("--out_channels", type=int, default=1, help="Output channels")
    p.add_argument("--style_ch", type=int, default=512, help="Encoder output (latent dim)")
    p.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=["l1", "mse", "mse_ssim"],
        help="Reconstruction loss. mse_ssim = MSE + ssim_weight * (1-SSIM)",
    )
    p.add_argument(
        "--ssim_weight",
        type=float,
        default=0.1,
        help="Weight for (1-SSIM) when loss=mse_ssim (typical: 0.05–0.2)",
    )
    p.add_argument(
        "--ssim_data_range",
        type=float,
        default=2.0,
        help="SSIM data_range (2.0 for inputs in ~[-1,1], 1.0 for [0,1])",
    )
    p.add_argument(
        "--ssim_window",
        type=int,
        default=11,
        help="Gaussian window size for SSIM (odd, default 11)",
    )

    p.add_argument("--apply_lv_mask", action="store_true", help="Apply LV mask if available")
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz", help="CT file suffix")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                   help="Mask filename")

    p.add_argument("--out_dir", type=str, default="./outputs/train_plain_ae", help="Output directory")
    p.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0=off)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume (encoder, decoder, optimizer, epoch)")
    p.add_argument("--log_every", type=int, default=50, help="Log loss every N batches (0=epoch only)")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay (DiffAE default). Use 0 to disable EMA.")
    p.add_argument("--early_stop_patience", type=int, default=20, help="Early stop after N epochs without val improvement. 0=off.")
    return p


def ema_update(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    """Update target = decay * target + (1 - decay) * source (in-place)."""
    src_dict = source.state_dict()
    tgt_dict = target.state_dict()
    for key in src_dict.keys():
        if key in tgt_dict:
            tgt_dict[key].data.copy_(
                tgt_dict[key].data * decay + src_dict[key].data * (1.0 - decay)
            )


def run_epoch(
    model: PlainAE,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    is_val: bool = False,
    epoch: int = 0,
    log_every: int = 0,
    ema_encoder: torch.nn.Module | None = None,
    ema_decoder: torch.nn.Module | None = None,
    ema_decay: float = 0.0,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.0
    n = 0

    for batch_idx, batch in enumerate(loader):
        x0 = batch["img"].to(device)  # (B, 1, H, W) - slice-level from CTSliceDataset

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            x0_hat, cond = model(x0, return_cond=True)
            loss = loss_fn(x0_hat, x0)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # EMA update (after optimizer.step)
            if ema_decay > 0 and ema_encoder is not None and ema_decoder is not None:
                ema_update(model.encoder, ema_encoder, ema_decay)
                ema_update(model.decoder, ema_decoder, ema_decay)

        total += float(loss.item()) * x0.shape[0]
        n += x0.shape[0]

        if log_every > 0 and is_train and (batch_idx + 1) % log_every == 0:
            avg_so_far = total / n
            print(f"  epoch {epoch} batch {batch_idx + 1}/{len(loader)} loss={loss.item():.6f} avg={avg_so_far:.6f}", flush=True)

    return total / max(n, 1)


def main() -> None:
    args = build_argparser().parse_args()
    print("Plain AE Stage A training starting...", flush=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Build dataset (CTSliceDataset from diffae-master)
    from dataset import CTSliceDataset

    def make_ct_slice_dataset(include_csv):
        return CTSliceDataset(
            root_dir=args.ct_root,
            metadata_csv=args.metadata_csv or None,
            include_patient_csv=include_csv,
            expected_shape_xyz=tuple(args.expected_shape),
            apply_lv_mask=args.apply_lv_mask,
            ct_suffix=args.ct_suffix,
            mask_filename=args.mask_filename,
            num_channels=args.in_channels,
        )

    ds_train = make_ct_slice_dataset(args.train_csv or None)
    ds_val = None
    if args.val_csv and os.path.exists(args.val_csv):
        ds_val = make_ct_slice_dataset(args.val_csv)

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

    model = plain_ae_ffhq128_ct(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        style_ch=args.style_ch,
    ).to(device)

    ema_encoder = None
    ema_decoder = None
    if args.ema_decay > 0:
        ema_encoder = copy.deepcopy(model.encoder)
        ema_decoder = copy.deepcopy(model.decoder)
        ema_encoder.requires_grad_(False)
        ema_decoder.requires_grad_(False)
        ema_encoder.eval()
        ema_decoder.eval()
        print(f"EMA enabled (decay={args.ema_decay})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == "l1":
        loss_fn = nn.L1Loss()
    elif args.loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = MseSsimLoss(
            ssim_weight=args.ssim_weight,
            data_range=args.ssim_data_range,
            in_channels=args.in_channels,
            window_size=args.ssim_window,
        )
    print(
        f"Loss: {args.loss}"
        + (
            f" (ssim_weight={args.ssim_weight}, ssim_data_range={args.ssim_data_range})"
            if args.loss == "mse_ssim"
            else ""
        ),
        flush=True,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    os.makedirs(args.out_dir, exist_ok=True)
    loss_csv = os.path.join(args.out_dir, "loss_history.csv")
    best_ckpt = os.path.join(args.out_dir, "checkpoint_best.pth")

    start_epoch = 1
    best_val = float("inf")
    history = []
    epochs_no_improve = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.encoder.load_state_dict(ckpt["encoder"], strict=True)
        model.decoder.load_state_dict(ckpt["decoder"], strict=True)
        if ema_encoder is not None and "ema_encoder" in ckpt:
            ema_encoder.load_state_dict(ckpt["ema_encoder"], strict=True)
            ema_decoder.load_state_dict(ckpt["ema_decoder"], strict=True)
        elif ema_encoder is not None and "ema_encoder" not in ckpt:
            ema_encoder.load_state_dict(ckpt["encoder"], strict=True)
            ema_decoder.load_state_dict(ckpt["decoder"], strict=True)
            print("  (no ema in ckpt; initialized ema from encoder/decoder)", flush=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    n_train = len(ds_train)
    n_batches = len(loader_train)
    print(f"Training: {n_train} samples, {n_batches} batches/epoch, batch_size={args.batch_size}", flush=True)
    if args.log_every > 0:
        print(f"Logging every {args.log_every} batches", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model, loader_train, device, loss_fn, optimizer, scaler,
            is_val=False, epoch=epoch, log_every=args.log_every,
            ema_encoder=ema_encoder, ema_decoder=ema_decoder, ema_decay=args.ema_decay,
        )
        val_loss = None
        if loader_val is not None:
            val_loss = run_epoch(model, loader_val, device, loss_fn, None, None, is_val=True)

        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_loss:.6f}" +
              (f" val={val_loss:.6f}" if val_loss is not None else ""), flush=True)

        history.append((epoch, train_loss, val_loss))

        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt_dict = {
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": best_val,
                "history": history,
            }
            if ema_encoder is not None:
                ckpt_dict["ema_encoder"] = ema_encoder.state_dict()
                ckpt_dict["ema_decoder"] = ema_decoder.state_dict()
            torch.save(ckpt_dict, best_ckpt)
            print(f"  saved best: {best_ckpt}", flush=True)
        elif val_loss is not None and args.early_stop_patience > 0:
            epochs_no_improve += 1

        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch:03d}.pth")
            ckpt_dict = {
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": best_val,
                "history": history,
            }
            if ema_encoder is not None:
                ckpt_dict["ema_encoder"] = ema_encoder.state_dict()
                ckpt_dict["ema_decoder"] = ema_decoder.state_dict()
            torch.save(ckpt_dict, ckpt_path)
            print(f"  saved: {ckpt_path}", flush=True)

        with open(loss_csv, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for e, tr, va in history:
                f.write(f"{e},{tr},{'' if va is None else va}\n")

        # Early stopping
        if args.early_stop_patience > 0 and val_loss is not None and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping: no val improvement for {args.early_stop_patience} epochs", flush=True)
            break

    print("Training complete!", flush=True)
    print(f"Best checkpoint: {best_ckpt}", flush=True)


if __name__ == "__main__":
    main()
