#!/usr/bin/env python3
"""
extract_embeddings_and_kmeans.py
- Stage B: Extract slice embeddings and run spherical k-means (cosine).
"""

import os
import sys
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

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


def build_dataset(root_dir: str, metadata_csv: str, expected_xyz: Tuple[int, int, int], include_csv: str) -> CTROIVolumeDataset:
    return CTROIVolumeDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        expected_shape_xyz=tuple(expected_xyz),
        include_patient_csv=include_csv,
        return_patient_id=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract slice embeddings and run spherical k-means")
    parser.add_argument("--normal_root", type=str, required=True, help="Root directory for Normal dataset")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv split file")
    parser.add_argument("--extra_csvs", type=str, nargs="*", default=None,
                        help="Optional extra split CSVs to include (e.g., val.csv)")
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64],
                        metavar=("X", "Y", "Z"), help="Expected shape (X, Y, Z)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to encoder checkpoint (Stage A)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_slices", type=int, default=32, help="Slice batch size for encoder")
    parser.add_argument("--k_clusters", type=int, default=64, help="Number of spherical k-means clusters")
    parser.add_argument("--max_iter", type=int, default=50, help="Max k-means iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    expected_shape_xyz = tuple(args.expected_shape)

    include_csv = args.train_csv
    if args.extra_csvs:
        # Build a combined include CSV with union of IDs from train + extras
        ids = []
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

    dataset = build_dataset(args.normal_root, args.metadata_csv, expected_shape_xyz, include_csv)
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check train_csv and data paths.")

    # Load model (encoder only)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    dae_args = checkpoint.get("args", {})
    schedule = DiffusionSchedule(
        timesteps=dae_args.get("timesteps", 1000),
        beta_start=dae_args.get("beta_start", 1e-4),
        beta_end=dae_args.get("beta_end", 2e-2),
    )
    model = DAE2D(
        in_ch=1,
        y_sem_dim=dae_args.get("y_sem_dim", 512),
        base_ch=dae_args.get("base_ch", 64),
        schedule=schedule,
    ).to(device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.encoder.eval()

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
                # Encode slice batch into 512-d semantic vectors
                emb = model.encoder(batch).cpu().numpy()
            emb_list.append(emb)
        emb_all = np.concatenate(emb_list, axis=0)  # (D, 512)
        emb_all = l2_normalize(emb_all)

        per_patient[pid] = {
            "embeddings": emb_all,
        }

        for s_idx in range(d):
            all_embeddings.append(emb_all[s_idx])
            meta.append((pid, s_idx))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} patients")

    all_embeddings = np.stack(all_embeddings, axis=0)  # (N, 512)
    prototypes, assignments = spherical_kmeans(
        all_embeddings,
        k=args.k_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    # Assign clusters back to each slice (patient-wise)
    for (pid, s_idx), c_id in zip(meta, assignments):
        if "cluster_ids" not in per_patient[pid]:
            per_patient[pid]["cluster_ids"] = np.zeros(per_patient[pid]["embeddings"].shape[0], dtype=np.int64)
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
