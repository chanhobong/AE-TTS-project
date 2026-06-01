#!/usr/bin/env python3
"""
Train DiffAE (Stage A only) on CT slices using the diffae-master codebase.

This uses the DiffAE autoencoder model (BeatGANsAutoenc) and a CT slice dataset
wrapper that returns {"img": tensor, "index": idx} compatible with DiffAE.
"""

import argparse
import os
import sys
from typing import List


def _add_diffae_to_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ae_root = os.path.dirname(script_dir)
    diffae_root = os.path.join(ae_root, "diffae-master")
    if diffae_root not in sys.path:
        sys.path.insert(0, diffae_root)
    return diffae_root


def _parse_gpus(gpu_str: str) -> List[int]:
    if gpu_str.strip().lower() in {"cpu", "none", ""}:
        return []
    return [int(x) for x in gpu_str.split(",") if x.strip() != ""]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DiffAE Stage-A on CT slices")
    p.add_argument("--ct_root", type=str, required=True, help="Root dir of CT ROI volumes")
    p.add_argument("--metadata_csv", type=str, default="", help="Metadata CSV (optional)")
    p.add_argument("--train_csv", type=str, default="", help="Train split CSV (optional)")
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                   metavar=("X", "Y", "Z"), help="Expected CT shape (X, Y, Z)")
    p.add_argument("--img_size", type=int, default=128, help="Slice size (H=W)")

    p.add_argument("--batch_size", type=int, default=4, help="Batch size (slices)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--total_samples", type=int, default=2_000_000, help="Total samples to train")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--fp16", action="store_true", help="Enable fp16 training")

    p.add_argument("--in_channels", type=int, default=1, help="Input channels (CT=1)")
    p.add_argument("--out_channels", type=int, default=1, help="Output channels")
    p.add_argument("--style_ch", type=int, default=512, help="Encoder output channels (latent dim)")

    p.add_argument("--apply_lv_mask", action="store_true", help="Apply LV mask if available")
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz", help="CT file suffix")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                   help="Mask filename")

    p.add_argument("--name", type=str, default="ct_slices_diffae", help="Run name")
    p.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. '0' or '0,1' or 'cpu'")
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    return p


def main() -> None:
    _add_diffae_to_path()
    from templates import ffhq128_autoenc_base
    from experiment import train

    args = build_argparser().parse_args()

    conf = ffhq128_autoenc_base()
    conf.name = args.name
    conf.data_name = "ct_slices"
    conf.img_size = args.img_size
    conf.batch_size = args.batch_size
    conf.lr = args.lr
    conf.total_samples = args.total_samples
    conf.num_workers = args.num_workers
    conf.fp16 = args.fp16
    conf.in_channels = args.in_channels
    conf.out_channels = args.out_channels
    conf.style_ch = args.style_ch
    conf.net_beatgans_embed_channels = args.style_ch

    # CT dataset fields
    conf.ct_root = args.ct_root
    conf.ct_metadata_csv = args.metadata_csv
    conf.ct_split_csv = args.train_csv
    conf.ct_expected_shape_xyz = tuple(args.expected_shape)
    conf.ct_apply_lv_mask = args.apply_lv_mask
    conf.ct_ct_suffix = args.ct_suffix
    conf.ct_mask_filename = args.mask_filename
    conf.ct_num_channels = args.in_channels

    # Rebuild model config after overriding channels and sizes
    conf.make_model_conf()

    gpus = _parse_gpus(args.gpus)
    train(conf, gpus=gpus, nodes=args.nodes)


if __name__ == "__main__":
    main()
