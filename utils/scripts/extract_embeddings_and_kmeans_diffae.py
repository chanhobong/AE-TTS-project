#!/usr/bin/env python3
"""
extract_embeddings_and_kmeans_diffae.py
- Stage B (DiffAE): Extract slice embeddings from a DiffAE encoder checkpoint
  and run spherical k-means (cosine) to build prototype IDs per slice.
"""

import argparse
import os
import sys
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset

# Add AE_TTS root and diffae-master to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
diffae_root = os.path.join(ae_root, "diffae-master")
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)
if diffae_root not in sys.path:
    sys.path.insert(0, diffae_root)

from data.dataset import CTROIVolumeDataset
from templates import ffhq128_autoenc_base
from experiment import LitModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Normalize each row vector to unit length (cosine space)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def spherical_kmeans(
    X: np.ndarray,
    k: int = 64,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical k-means using cosine similarity.
    Returns (centroids, assignments).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"Not enough samples ({n}) for k={k}")

    centroids = X[rng.choice(n, size=k, replace=False)]
    centroids = l2_normalize(centroids)

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Cosine similarity (dot product for normalized vectors)
        sims = X @ centroids.T  # (N, K)
        new_assignments = sims.argmax(axis=1)

        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments

        # Update centroids by averaging assigned points
        for j in range(k):
            idx = np.where(assignments == j)[0]
            if len(idx) == 0:
                # Re-init empty cluster
                centroids[j] = X[rng.integers(0, n)]
            else:
                centroids[j] = X[idx].mean(axis=0)
        centroids = l2_normalize(centroids)

    return centroids, assignments


def _load_hparams_yaml(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"hparams.yaml not found: {path}")
    with open(path, "r") as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid hparams.yaml format")
    # model_conf should be rebuilt from net_* fields
    data.pop("model_conf", None)
    return data


def _build_diffae_encoder(checkpoint: str, hparams_path: str, device: torch.device, use_ema: bool):
    conf = ffhq128_autoenc_base()
    hparams = _load_hparams_yaml(hparams_path)
    conf.from_dict(hparams, strict=False)

    # Ensure CT channel settings and rebuild model config
    conf.data_name = hparams.get("data_name", "ct_slices")
    conf.in_channels = int(hparams.get("in_channels", conf.in_channels))
    conf.out_channels = int(hparams.get("out_channels", conf.out_channels))
    conf.style_ch = int(hparams.get("style_ch", conf.style_ch))
    conf.net_beatgans_embed_channels = int(
        hparams.get("net_beatgans_embed_channels", conf.style_ch)
    )
    conf.img_size = int(hparams.get("img_size", conf.img_size))
    conf.make_model_conf()

    model = LitModel(conf)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device)
    model.eval()

    encoder = model.ema_model.encoder if use_ema else model.model.encoder
    encoder.eval()
    return encoder


def _build_dataset(
    normal_root: str,
    tts_root: str | None,
    metadata_csv: str,
    expected_xyz: Tuple[int, int, int],
    include_csv: str,
    apply_lv_mask: bool,
    ct_suffix: str,
    mask_filename: str,
) -> torch.utils.data.Dataset:
    datasets = [
        CTROIVolumeDataset(
            root_dir=normal_root,
            metadata_csv=metadata_csv,
            expected_shape_xyz=tuple(expected_xyz),
            include_patient_csv=include_csv,
            return_patient_id=True,
            apply_lv_mask=apply_lv_mask,
            ct_suffix=ct_suffix,
            mask_filename=mask_filename,
        )
    ]
    if tts_root:
        datasets.append(
            CTROIVolumeDataset(
                root_dir=tts_root,
                metadata_csv=metadata_csv,
                expected_shape_xyz=tuple(expected_xyz),
                include_patient_csv=include_csv,
                return_patient_id=True,
                apply_lv_mask=apply_lv_mask,
                ct_suffix=ct_suffix,
                mask_filename=mask_filename,
            )
        )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DiffAE slice embeddings and run spherical k-means"
    )
    parser.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    parser.add_argument("--tts_root", type=str, default="", help="Optional root directory for TTS dataset")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    parser.add_argument("--extra_csvs", type=str, nargs="*", default=None,
                        help="Optional extra split CSVs to include (e.g., val.csv)")
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                        metavar=("X", "Y", "Z"), help="Expected shape (X, Y, Z)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffAE checkpoint (Stage A)")
    parser.add_argument("--hparams", type=str, default="",
                        help="Path to hparams.yaml (default: <ckpt_dir>/hparams.yaml)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_slices", type=int, default=32, help="Slice batch size for encoder")
    parser.add_argument("--k_clusters", type=int, default=64, help="Number of spherical k-means clusters")
    parser.add_argument("--max_iter", type=int, default=50, help="Max k-means iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA encoder (default)")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="Use non-EMA encoder")
    parser.add_argument("--apply_lv_mask", action="store_true", help="Apply LV mask if available")
    parser.add_argument("--ct_suffix", type=str, default="_roi.nii.gz", help="CT file suffix")
    parser.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz",
                        help="Mask filename")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    expected_shape_xyz = tuple(args.expected_shape)

    hparams_path = args.hparams
    if not hparams_path:
        hparams_path = os.path.join(os.path.dirname(args.checkpoint), "hparams.yaml")

    include_csv = args.train_csv
    if args.extra_csvs:
        # Build a combined include CSV with union of IDs from train + extras
        ids: List[str] = []
        for csv_path in [args.train_csv] + list(args.extra_csvs):
            df = pd.read_csv(csv_path)
            if "patient_id" in df.columns:
                ids.extend(df["patient_id"].astype(str).tolist())
            elif "ID" in df.columns:
                ids.extend(df["ID"].astype(str).tolist())
            else:
                ids.extend(df[df.columns[0]].astype(str).tolist())
        ids = sorted(set([i.strip() for i in ids]))
        os.makedirs(args.out_dir, exist_ok=True)
        include_csv = os.path.join(args.out_dir, "combined_include_ids.csv")
        pd.DataFrame({"patient_id": ids}).to_csv(include_csv, index=False)
        print(f"✅ Combined include CSV saved: {include_csv} (n={len(ids)})")

    tts_root = args.tts_root.strip() or None
    dataset = _build_dataset(
        normal_root=args.normal_root,
        tts_root=tts_root,
        metadata_csv=args.metadata_csv,
        expected_xyz=expected_shape_xyz,
        include_csv=include_csv,
        apply_lv_mask=args.apply_lv_mask,
        ct_suffix=args.ct_suffix,
        mask_filename=args.mask_filename,
    )
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check train_csv and data paths.")

    encoder = _build_diffae_encoder(
        checkpoint=args.checkpoint,
        hparams_path=hparams_path,
        device=device,
        use_ema=args.use_ema,
    )

    all_embeddings = []
    meta = []  # (patient_id, slice_idx)
    per_patient: Dict[str, Dict[str, np.ndarray]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pid = sample["patient_id"]
        x = sample["x"].to(device)  # (1, D, H, W)

        # Build slice batch: (D, 1, H, W)
        x_slices = x.permute(1, 0, 2, 3).contiguous()
        d = x_slices.shape[0]

        emb_list = []
        for i in range(0, d, args.batch_slices):
            batch = x_slices[i:i + args.batch_slices]
            with torch.no_grad():
                emb = encoder(batch).float().cpu().numpy()
            emb_list.append(emb)
        emb_all = np.concatenate(emb_list, axis=0)  # (D, style_ch)
        emb_all = l2_normalize(emb_all)

        per_patient[pid] = {
            "embeddings": emb_all,
        }

        for s_idx in range(d):
            all_embeddings.append(emb_all[s_idx])
            meta.append((pid, s_idx))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} patients")

    all_embeddings = np.stack(all_embeddings, axis=0)  # (N, style_ch)
    prototypes, assignments = spherical_kmeans(
        all_embeddings,
        k=args.k_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    # Assign clusters back to each slice (patient-wise)
    for (pid, s_idx), c_id in zip(meta, assignments):
        if "cluster_ids" not in per_patient[pid]:
            per_patient[pid]["cluster_ids"] = np.zeros(
                per_patient[pid]["embeddings"].shape[0], dtype=np.int64
            )
        per_patient[pid]["cluster_ids"][s_idx] = c_id

    out_patients = os.path.join(args.out_dir, "patients")
    os.makedirs(out_patients, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    np.save(os.path.join(args.out_dir, "prototypes.npy"), prototypes)

    for pid, data in per_patient.items():
        emb = data["embeddings"]
        cids = data["cluster_ids"]
        # All slices are valid in this stage; mask is all ones
        mask = np.ones(emb.shape[0], dtype=np.int64)
        np.savez_compressed(
            os.path.join(out_patients, f"{pid}.npz"),
            embeddings=emb,
            cluster_ids=cids,
            mask=mask,
            patient_id=pid,
        )

    print(f"✅ Saved prototypes to: {os.path.join(args.out_dir, 'prototypes.npy')}")
    print(f"✅ Saved per-patient sequences to: {out_patients}")


if __name__ == "__main__":
    main()
