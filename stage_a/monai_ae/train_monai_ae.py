#!/usr/bin/env python3
"""
Train MONAI 2D AutoEncoder on CT slices (same CTSliceDataset as Plain AE).

Prefer `monai.networks.nets.AutoEncoder` (installed MONAI). Local `AE.py` is a reference copy; use --use_local_module to import that instead.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# stageA layout: shared/diffae + plain_ae (MseSsimLoss)
_MONAI_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE_A = os.path.dirname(_MONAI_DIR)
_AE_TTS = os.path.dirname(_STAGE_A)
_SHARED = os.path.join(_STAGE_A, "shared")
_DIFFAE = os.path.join(_SHARED, "diffae")
_PLAIN_STAGEA = os.path.join(_STAGE_A, "plain_ae")
for p in (_SHARED, _DIFFAE, _PLAIN_STAGEA, _AE_TTS):
    if p not in sys.path:
        sys.path.insert(0, p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.use_local_module:
        from AE import AutoEncoder
    else:
        from monai.networks.nets import AutoEncoder

    chs = tuple(int(x) for x in args.channels.split(","))
    sts = tuple(int(x) for x in args.strides.split(","))
    if len(chs) != len(sts):
        raise ValueError("--channels and --strides must have the same length (see MONAI AutoEncoder).")

    net = AutoEncoder(
        spatial_dims=2,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=chs,
        strides=sts,
        num_res_units=args.num_res_units,
    )
    return net.to(device)


def build_loss_fn(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Match train_plain_ae Stage A: l1 | mse | mse_ssim."""
    if args.loss == "l1":
        return nn.L1Loss()
    if args.loss == "mse":
        return nn.MSELoss()
    from train_plain_ae import MseSsimLoss

    return MseSsimLoss(
        ssim_weight=args.ssim_weight,
        data_range=args.ssim_data_range,
        in_channels=args.in_channels,
        window_size=args.ssim_window,
    ).to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict):
            x = batch["img"].to(device)
        else:
            x = batch[0].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = loss_fn(out, x)
        if train:
            loss.backward()
            optimizer.step()  # type: ignore[union-attr]
        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train MONAI 2D AutoEncoder on CT slices")
    p.add_argument("--ct_root", type=str, default="", help="CT normal root")
    p.add_argument("--metadata_csv", type=str, default="")
    p.add_argument("--train_csv", type=str, default="")
    p.add_argument("--val_csv", type=str, default="")
    p.add_argument("--out_dir", type=str, default=os.path.join(_MONAI_DIR, "outputs", "monai_ae_run"))
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--apply_lv_mask", action="store_true")
    p.add_argument("--epochs", type=int, default=100, help="Same default as train_plain_ae.py")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--lr", type=float, default=1e-4, help="train_plain_ae default"
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--in_channels", type=int, default=1, help="CT slice channels (Plain AE default 1)"
    )
    p.add_argument(
        "--out_channels", type=int, default=1, help="Reconstruction channels (Plain AE default 1)"
    )
    p.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=["l1", "mse", "mse_ssim"],
        help="Reconstruction loss (same names as train_plain_ae)",
    )
    p.add_argument("--ssim_weight", type=float, default=0.1, help="For loss=mse_ssim")
    p.add_argument(
        "--ssim_data_range",
        type=float,
        default=2.0,
        help="~2.0 for [-1,1] CT (train_plain_ae default)",
    )
    p.add_argument("--ssim_window", type=int, default=11, help="SSIM window (odd)" )
    p.add_argument("--use_local_module", action="store_true", help="Import AutoEncoder from ./AE.py instead of monai.networks.nets")
    p.add_argument(
        "--channels",
        type=str,
        default="16,32,64,128",
        help="Comma-separated encoder channels (same length as --strides)",
    )
    p.add_argument(
        "--strides",
        type=str,
        default="2,2,2,2",
        help="Comma-separated strides (e.g. four 2s -> 128/16 spatial bottleneck for 2D)",
    )
    p.add_argument("--num_res_units", type=int, default=0, help="0 = plain conv blocks, >0 = residual")
    p.add_argument("--patience", type=int, default=0, help="0 = no early stop; else val-based (needs --val_csv)")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}  loss={args.loss}  in_ch={args.in_channels}  "
        f"opt=Adam(lr={args.lr})  (aligned with train_plain_ae Stage A)",
        flush=True,
    )

    model = build_model(args, device)
    loss_fn = build_loss_fn(args, device)
    # train_plain_ae: torch.optim.Adam(model.parameters(), lr=args.lr) — no weight_decay
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_path = os.path.join(args.out_dir, "monai_ae_best.pth")

    if not args.ct_root or not args.train_csv or not os.path.isfile(args.train_csv):
        raise SystemExit("Need --ct_root and --train_csv (existing file)")

    from dataset import CTSliceDataset  # type: ignore

    def make_ds(path: Optional[str]):
        if not path or not os.path.isfile(path):
            return None
        return CTSliceDataset(
            root_dir=args.ct_root,
            metadata_csv=args.metadata_csv or None,
            include_patient_csv=path,
            expected_shape_xyz=tuple(args.expected_shape),
            apply_lv_mask=args.apply_lv_mask,
        )

    ds_tr = make_ds(args.train_csv)
    ds_va = make_ds(args.val_csv) if args.val_csv else None
    if ds_tr is None or len(ds_tr) == 0:
        raise SystemExit("Train dataset is empty")
    loader_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_va = None
    if ds_va is not None and len(ds_va) > 0:
        loader_va = DataLoader(
            ds_va,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    os.makedirs(args.out_dir, exist_ok=True)
    best = float("inf")
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, loader_tr, device, loss_fn, opt)
        msg = f"epoch {ep}/{args.epochs}  train_{args.loss}={tr:.6f}"
        if loader_va is not None:
            va = run_epoch(model, loader_va, device, loss_fn, optimizer=None)
            msg += f"  val_{args.loss}={va:.6f}"
            if va < best:
                best = va
                no_improve = 0
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, best_path)
                msg += "  [best]"
            else:
                no_improve += 1
        else:
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": ep},
                os.path.join(args.out_dir, f"monai_ae_epoch{ep:03d}.pth"),
            )
        print(msg, flush=True)
        if args.patience > 0 and loader_va is not None and no_improve >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs", flush=True)
            break
    if loader_va is not None and os.path.isfile(best_path):
        print(f"best checkpoint: {best_path}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
