#!/usr/bin/env python3
"""
Quick diagnostics for DiffAE Stage A conditioning usage.
Computes loss_real/loss_zero/loss_rand and l1_real/l1_zero/l1_rand,
plus cond_var, from a few batches without retraining.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _add_diffae_to_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ae_root = os.path.dirname(script_dir)
    diffae_root = os.path.join(ae_root, "diffae-master")
    if diffae_root not in sys.path:
        sys.path.insert(0, diffae_root)
    return diffae_root


def _unique_out_dir(out_dir: Path) -> Path:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    # if exists and non-empty, create a versioned folder
    if any(out_dir.iterdir()):
        for i in range(1, 1000):
            cand = Path(f"{out_dir}_v{i}")
            if not cand.exists():
                cand.mkdir(parents=True, exist_ok=True)
                return cand
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose DiffAE conditioning usage")
    p.add_argument("--ckpt", type=str, required=True, help="Path to DiffAE Lightning checkpoint")
    p.add_argument("--ct_root", type=str, required=True, help="Root dir of CT ROI volumes")
    p.add_argument("--metadata_csv", type=str, default="", help="Metadata CSV (optional)")
    p.add_argument("--split_csv", type=str, default="", help="Split CSV (optional)")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected CT shape (X, Y, Z)")
    p.add_argument("--img_size", type=int, default=128, help="Slice size (H=W)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (slices)")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--max_batches", type=int, default=10, help="Max batches to evaluate")
    p.add_argument("--apply_lv_mask", action="store_true", help="Apply LV mask if available")
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz", help="CT file suffix")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                   help="Mask filename")
    p.add_argument("--in_channels", type=int, default=1, help="Input channels (CT=1)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory (no overwrite)")
    return p


def _mean_or_nan(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def main() -> None:
    _add_diffae_to_path()
    from templates import ffhq128_autoenc_base
    from experiment import LitModel
    from dataset import CTSliceDataset

    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = ffhq128_autoenc_base()
    conf.data_name = "ct_slices"
    conf.img_size = args.img_size
    conf.in_channels = args.in_channels
    conf.out_channels = args.in_channels
    conf.ct_root = args.ct_root
    metadata_csv = args.metadata_csv if args.metadata_csv and os.path.exists(args.metadata_csv) else ""
    split_csv = args.split_csv if args.split_csv and os.path.exists(args.split_csv) else ""
    conf.ct_metadata_csv = metadata_csv
    conf.ct_split_csv = split_csv
    conf.ct_expected_shape_xyz = tuple(args.expected_shape)
    conf.ct_apply_lv_mask = args.apply_lv_mask
    conf.ct_ct_suffix = args.ct_suffix
    conf.ct_mask_filename = args.mask_filename
    conf.ct_num_channels = args.in_channels
    conf.make_model_conf()

    model = LitModel(conf)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval().to(device)
    model.eval()

    dataset = CTSliceDataset(
        root_dir=args.ct_root,
        metadata_csv=metadata_csv or None,
        include_patient_csv=split_csv or None,
        expected_shape_xyz=tuple(args.expected_shape),
        apply_lv_mask=args.apply_lv_mask,
        ct_suffix=args.ct_suffix,
        mask_filename=args.mask_filename,
        num_channels=args.in_channels,
        do_augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    out_dir = _unique_out_dir(Path(args.out_dir))
    results = {
        "ckpt": args.ckpt,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": {},
    }

    loss_real_list = []
    loss_zero_list = []
    loss_rand_list = []
    l1_real_list = []
    l1_zero_list = []
    l1_rand_list = []
    cond_var_list = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= args.max_batches:
                break
            x = batch["img"].to(device)

            # sample t from the same scheduler
            t, _ = model.T_sampler.sample(len(x), x.device)
            noise = torch.randn_like(x)

            cond = model.encode(x)
            cond_var = cond.var(dim=0).mean().item()

            rand_idx = torch.randperm(len(cond), device=device)
            cond_rand = cond[rand_idx]
            cond_zero = torch.zeros_like(cond)

            def _loss_and_l1(cond_in):
                terms = model.sampler.training_losses(
                    model=model.ema_model,
                    x_start=x,
                    t=t,
                    model_kwargs={"cond": cond_in},
                    noise=noise,
                )
                loss = terms["loss"].mean().item()
                if "pred_xstart" in terms:
                    l1 = (terms["pred_xstart"] - x).abs().mean().item()
                else:
                    l1 = float("nan")
                return loss, l1

            loss_real, l1_real = _loss_and_l1(cond)
            loss_zero, l1_zero = _loss_and_l1(cond_zero)
            loss_rand, l1_rand = _loss_and_l1(cond_rand)

            loss_real_list.append(loss_real)
            loss_zero_list.append(loss_zero)
            loss_rand_list.append(loss_rand)
            l1_real_list.append(l1_real)
            l1_zero_list.append(l1_zero)
            l1_rand_list.append(l1_rand)
            cond_var_list.append(cond_var)

    results["metrics"] = {
        "loss_real": _mean_or_nan(loss_real_list),
        "loss_zero": _mean_or_nan(loss_zero_list),
        "loss_rand": _mean_or_nan(loss_rand_list),
        "l1_real": _mean_or_nan(l1_real_list),
        "l1_zero": _mean_or_nan(l1_zero_list),
        "l1_rand": _mean_or_nan(l1_rand_list),
        "cond_var": _mean_or_nan(cond_var_list),
    }

    out_json = out_dir / "cond_diagnosis.json"
    out_txt = out_dir / "cond_diagnosis.txt"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    with out_txt.open("w") as f:
        for k, v in results["metrics"].items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Saved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
