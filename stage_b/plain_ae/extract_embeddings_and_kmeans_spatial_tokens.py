#!/usr/bin/env python3
"""
Stage B (Plain AE) — spatial grid tokens without GAP.

Keeps Stage A Plain AE checkpoints fixed; only changes extraction.

Per-slice Plain BeatGANs head (adaptivenonzero pool):
  middle_block -> Norm -> SiLU -> GAP -> 1x1 conv -> flatten.
This script applies Norm -> SiLU -> 1x1 conv on the FULL spatial tensor
(skips GAP only), then flattens (H', W') -> one token row per spatial location.

Per-patient ``patients/*.npz`` keys (schema_version=2):
  - embeddings (float32): (n_slices * H'*W', d_raw) UNNORMALIZED conv outputs (k-means fits on L2 row-normalized copy).
  - embeddings_l2 (float32): same shape, row L2-normalized (for cosine / inspection).
  - cluster_ids (int64): (n_slices * H'*W',) spherical k-means assignment
  - slice_indices (int32): which volume slice index (0 .. n_slices-1) each token belongs to
  - spatial_flat_idx (int32): 0 .. H'*W'-1 repeating per slice (row-major flatten over H', W')
  - spatial_y, spatial_x (int32): grid coords from flat idx (y=flat//W', x=flat%W')
  - bottleneck_hw (int32): scalar array shape (2,) -> [H', W']
  - slice_z_per_slice (float32): (n_slices,) mm from NIfTI (slice axis)
  - slice_z_mm (float32): (n_slices * H'*W',) slice_z duplicated per spatial token (convenience)
  - mask (int64): (n_slices * H'*W',) typically all 1
  - slice_position_norm (float32): (n_slices * H'*W',) Apex–base norm repeated per-token
  - patient_id (str object array)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
plain_ae_root = os.path.join(ae_root, "plain_AE_stageA")
diffae_root = os.path.join(ae_root, "diffae-master")
diff3d_stageb_root = os.path.join(ae_root, "Diff_AE_STAGE_B_for_Diff3Dformer")
for p in (ae_root, diffae_root, plain_ae_root, diff3d_stageb_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import CTROIVolumeDataset
from nifti_slice_position import nifti_per_slice_positions
from plain_ae import plain_ae_ffhq128_ct

SCHEMA_VERSION = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def spherical_kmeans(
    X: np.ndarray,
    k: int = 64,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"Not enough samples ({n}) for k={k}")

    centroids = X[rng.choice(n, size=k, replace=False)]
    centroids = l2_normalize_rows(centroids)

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        sims = X @ centroids.T
        new_assignments = sims.argmax(axis=1)

        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments

        for j in range(k):
            idx = np.where(assignments == j)[0]
            if len(idx) == 0:
                centroids[j] = X[rng.integers(0, n)]
            else:
                centroids[j] = X[idx].mean(axis=0)
        centroids = l2_normalize_rows(centroids)

    return centroids, assignments


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


def _build_plain_ae_encoder(
    checkpoint: str,
    device: torch.device,
    style_ch: int = 512,
    use_ema: bool = True,
) -> nn.Module:
    model = plain_ae_ffhq128_ct(style_ch=style_ch)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    style_ch = saved_args.get("style_ch", style_ch) if saved_args else style_ch

    if use_ema and "ema_encoder" in ckpt:
        model.encoder.load_state_dict(ckpt["ema_encoder"], strict=True)
        print("  Using EMA encoder (use_ema=True)", flush=True)
    elif "encoder" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder"], strict=True)
        if use_ema and "ema_encoder" not in ckpt:
            print("  Note: checkpoint has no ema_encoder, using encoder", flush=True)
    else:
        model.encoder.load_state_dict(ckpt, strict=True)
    model.encoder.to(device).eval()
    return model.encoder


@torch.no_grad()
def plain_encoder_spatial_tokens(
    encoder: nn.Module, batch: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    """
    BeatGANsEncoderModel: forward to middle_block, then Norm->SiLU->1x1
    skipping AdaptiveAvgPool2d in encoder.out Sequential.

    Returns:
        tokens (B, H'*W', C_out); H', W' (bottleneck spatial size).
        Flatten order: NCHW contiguous (row-major over H', W').
    """
    if not hasattr(encoder, "out") or not isinstance(encoder.out, nn.Sequential):
        raise TypeError("Expected BeatGANsEncoderModel.encoder with Sequential `out` head")
    if len(encoder.out) < 4:
        raise ValueError(f"Unexpected encoder.out depth {len(encoder.out)} (need pool head with >=4 modules)")

    emb = None
    h = batch.type(encoder.dtype)
    for module in encoder.input_blocks:
        h = module(h, emb=emb)
    h = encoder.middle_block(h, emb=emb)
    h = h.type(batch.dtype)

    seq = encoder.out
    h = seq[0](h)
    h = seq[1](h)
    # Index 2: AdaptiveAvgPool2d — skipped
    h = seq[3](h)
    _b, _c, hp, wp = h.shape
    tok = h.flatten(2).transpose(1, 2).contiguous()
    return tok, int(hp), int(wp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage B (Plain AE): spatial grid tokens + spherical k-means — schema v2",
    )
    parser.add_argument("--normal_root", type=str, required=True)
    parser.add_argument("--tts_root", type=str, default="")
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument(
        "--assign_only_csvs",
        type=str,
        nargs="*",
        default=None,
        help="e.g. test.csv — assign-only, excluded from k-means fit",
    )
    parser.add_argument(
        "--extra_csvs",
        type=str,
        nargs="*",
        default=None,
        help="[DEPRECATED] train + extra_csvs used for fit (legacy).",
    )
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_slices", type=int, default=32)
    parser.add_argument("--k_clusters", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--style_ch", type=int, default=512)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--apply_lv_mask", action="store_true")
    parser.add_argument("--ct_suffix", type=str, default="_roi.nii.gz")
    parser.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz")
    parser.add_argument("--verify_nifti_slice_geometry", action="store_true")
    args = parser.parse_args()

    if args.verify_nifti_slice_geometry:
        print("  NIfTI slice geometry verification ON.", flush=True)

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    expected_shape_xyz = tuple(args.expected_shape)

    def _ids_from_csv(csv_path: str) -> List[str]:
        df = pd.read_csv(csv_path)
        if "patient_id" in df.columns:
            return df["patient_id"].astype(str).tolist()
        if "ID" in df.columns:
            return df["ID"].astype(str).tolist()
        return df[df.columns[0]].astype(str).tolist()

    fit_ids: List[str] = []
    assign_only_ids: List[str] = []

    if args.extra_csvs:
        for csv_path in [args.train_csv] + list(args.extra_csvs):
            fit_ids.extend(_ids_from_csv(csv_path))
        fit_ids = sorted(set([i.strip() for i in fit_ids]))
        assign_only_ids = []
        print(f"  [legacy] K-means fit on train+extra_csvs (n={len(fit_ids)})", flush=True)
    else:
        fit_ids.extend(_ids_from_csv(args.train_csv))
        if args.val_csv and os.path.isfile(args.val_csv):
            fit_ids.extend(_ids_from_csv(args.val_csv))
        fit_ids = sorted(set([i.strip() for i in fit_ids]))
        if args.assign_only_csvs:
            for csv_path in args.assign_only_csvs:
                assign_only_ids.extend(_ids_from_csv(csv_path))
            assign_only_ids = sorted(set([i.strip() for i in assign_only_ids]))
        print(
            f"  K-means fit: train+val (n={len(fit_ids)}), assign-only: n={len(assign_only_ids)}",
            flush=True,
        )

    all_ids = sorted(set(fit_ids) | set(assign_only_ids))
    os.makedirs(args.out_dir, exist_ok=True)
    include_csv = os.path.join(args.out_dir, "combined_include_ids.csv")
    pd.DataFrame({"patient_id": all_ids}).to_csv(include_csv, index=False)
    print(f"✅ Include CSV: {include_csv} (n={len(all_ids)})", flush=True)

    fit_set = set(fit_ids)
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
        raise RuntimeError("Dataset is empty.")

    encoder = _build_plain_ae_encoder(
        checkpoint=args.checkpoint,
        device=device,
        style_ch=args.style_ch,
        use_ema=args.use_ema,
    )

    fit_embeddings: List[np.ndarray] = []
    fit_meta: List[Tuple[str, int]] = []
    # fit_meta tracks global ordering: (patient_id, row_index_in_that_patient_emb_all)
    per_patient: Dict[str, Dict[str, Any]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pid = sample["patient_id"]
        x = sample["x"].to(device)
        x_slices = x.permute(1, 0, 2, 3).contiguous()
        d = x_slices.shape[0]

        token_rows_list: List[np.ndarray] = []
        slice_indices_parts: List[np.ndarray] = []
        spatial_idx_parts: List[np.ndarray] = []

        bottleneck_hw: Tuple[int, int] | None = None

        for i in range(0, d, args.batch_slices):
            batch = x_slices[i : i + args.batch_slices]
            with torch.no_grad():
                bt, hp, wp = plain_encoder_spatial_tokens(encoder, batch)
            if bottleneck_hw is None:
                bottleneck_hw = (hp, wp)
            elif bottleneck_hw != (hp, wp):
                raise RuntimeError(f"Inconsistent bottleneck {bottleneck_hw} vs {(hp, wp)} for pid={pid}")

            bt_np = bt.float().cpu().numpy()
            Bb = bt_np.shape[0]
            HW = bottleneck_hw[0] * bottleneck_hw[1]
            assert bt_np.shape[1] == HW, (bt_np.shape, HW, bottleneck_hw)

            spatial_flat = np.arange(HW, dtype=np.int32)
            for j in range(Bb):
                s_idx = i + j
                token_rows_list.append(bt_np[j])
                slice_indices_parts.append(np.full(HW, s_idx, dtype=np.int32))
                spatial_idx_parts.append(spatial_flat.copy())

        emb_all = np.concatenate(token_rows_list, axis=0)
        embeddings_l2 = l2_normalize_rows(emb_all)
        slice_indices_arr = np.concatenate(slice_indices_parts, axis=0)
        spatial_flat_idx = np.concatenate(spatial_idx_parts, axis=0)

        ct_path = sample.get("ct_path")
        if not ct_path:
            raise RuntimeError(f"[{pid}] missing ct_path for slice_z_mm")
        cw, _, _, _, shape3 = nifti_per_slice_positions(
            str(ct_path), axis=2, n_slices_expected=d, eps=1e-8
        )
        slice_z_mm_slice = cw[:, 2].astype(np.float32)
        slice_z_mm_tokens = slice_z_mm_slice[slice_indices_arr]

        if args.verify_nifti_slice_geometry and cw.shape[0] >= 2:
            step = np.linalg.norm(np.diff(cw, axis=0), axis=1)
            print(
                f"  [verify] {pid} NIfTI={shape3} slices={d} mean_step_mm={float(step.mean()):.5f}",
                flush=True,
            )

        per_patient[pid] = {
            "embeddings": emb_all,
            "embeddings_l2": embeddings_l2,
            "slice_indices": slice_indices_arr,
            "spatial_flat_idx": spatial_flat_idx,
            "bottleneck_hw": np.array(bottleneck_hw, dtype=np.int32),
            "slice_z_per_slice": slice_z_mm_slice,
            "slice_z_mm": slice_z_mm_tokens,
            "n_slices": d,
            "token_dim": emb_all.shape[1],
        }

        if pid in fit_set:
            for row_i in range(emb_all.shape[0]):
                fit_embeddings.append(embeddings_l2[row_i])
                fit_meta.append((pid, row_i))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} patients", flush=True)

    fit_embeddings_arr = np.stack(fit_embeddings, axis=0)
    prototypes, fit_assignments = spherical_kmeans(
        fit_embeddings_arr,
        k=args.k_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    cluster_ids_by_pid: Dict[str, np.ndarray] = {}
    for pid, pdata in per_patient.items():
        if pid in fit_set:
            cluster_ids_by_pid[pid] = np.zeros(pdata["embeddings"].shape[0], dtype=np.int64)

    for (pid, row_i), c_id in zip(fit_meta, fit_assignments):
        cluster_ids_by_pid[pid][row_i] = c_id

    for pid, pdata in per_patient.items():
        emb_l2 = pdata["embeddings_l2"]
        if pid in fit_set:
            pdata["cluster_ids"] = cluster_ids_by_pid[pid]
        else:
            sims = emb_l2 @ prototypes.T
            pdata["cluster_ids"] = sims.argmax(axis=1).astype(np.int64)

    out_patients = os.path.join(args.out_dir, "patients")
    os.makedirs(out_patients, exist_ok=True)
    np.save(os.path.join(args.out_dir, "prototypes.npy"), prototypes)

    for pid, data in per_patient.items():
        emb = data["embeddings"]
        emb_l2 = data["embeddings_l2"]
        cids = data["cluster_ids"]
        ntok = emb.shape[0]
        d_sl = data["n_slices"]
        hw_sz = int(data["bottleneck_hw"][0]) * int(data["bottleneck_hw"][1])

        slice_pos_rep = np.repeat(
            np.linspace(0.0, 1.0, d_sl, dtype=np.float32),
            repeats=hw_sz,
        )
        assert slice_pos_rep.shape[0] == ntok

        mask = np.ones(ntok, dtype=np.int64)

        hw_arr = data["bottleneck_hw"].astype(np.int32)
        hp_, wp_ = int(hw_arr[0]), int(hw_arr[1])
        flat = data["spatial_flat_idx"].astype(np.int32)
        spatial_y = (flat // wp_).astype(np.int32)
        spatial_x = (flat % wp_).astype(np.int32)

        np.savez_compressed(
            os.path.join(out_patients, f"{pid}.npz"),
            schema_version=np.int32(SCHEMA_VERSION),
            embeddings=emb.astype(np.float32),
            embeddings_l2=emb_l2.astype(np.float32),
            cluster_ids=cids,
            mask=mask,
            patient_id=np.array([pid], dtype=object),
            slice_indices=data["slice_indices"].astype(np.int32),
            spatial_flat_idx=flat,
            spatial_y=spatial_y,
            spatial_x=spatial_x,
            bottleneck_hw=hw_arr,
            slice_z_per_slice=data["slice_z_per_slice"].astype(np.float32),
            slice_z_mm=data["slice_z_mm"].astype(np.float32),
            slice_position_norm=slice_pos_rep,
        )

    print(f"✅ prototypes: {os.path.join(args.out_dir, 'prototypes.npy')}", flush=True)
    print(f"✅ patients:   {out_patients}  (schema_version={SCHEMA_VERSION})", flush=True)


if __name__ == "__main__":
    main()