#!/usr/bin/env python3
"""
Stage B (MONAI AutoEncoder) — spatial bottleneck tokens without GAP.

Stage A checkpoints unchanged; only extraction differs from extract_embeddings_and_kmeans_monai.py.

Per slice: ``encode()`` + ``intermediate()`` gives (B, C, H', W').
No pooling; flatten spatial grid -> H'*W' rows of dimension C each.

NPZ schema matches Plain StageB_new/spatial_tokens (schema_version=2).
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
_AE_TTS = os.path.abspath(os.path.join(script_dir, "..", ".."))
_MONAI_DIR = os.path.join(_AE_TTS, "MONAI")
_PLAIN_STAGEA = os.path.join(_AE_TTS, "plain_AE_stageA")
_DIFFAE = os.path.join(_AE_TTS, "diffae-master")
_DIFF3D_STAGE_B = os.path.join(_AE_TTS, "Diff_AE_STAGE_B_for_Diff3Dformer")
for p in (_AE_TTS, _DIFFAE, _PLAIN_STAGEA, _DIFF3D_STAGE_B):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import CTROIVolumeDataset
from nifti_slice_position import nifti_per_slice_positions

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


def _build_monai_ae(checkpoint: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args") or {}
    use_local = bool(saved.get("use_local_module", False))
    if use_local:
        from AE import AutoEncoder  # type: ignore

    else:
        from monai.networks.nets import AutoEncoder  # type: ignore

    chs = tuple(int(x) for x in str(saved.get("channels", "16,32,64,128")).split(","))
    sts = tuple(int(x) for x in str(saved.get("strides", "2,2,2,2")).split(","))
    if len(chs) != len(sts):
        raise ValueError("Checkpoint args: channels and strides length mismatch")

    net = AutoEncoder(
        spatial_dims=2,
        in_channels=int(saved.get("in_channels", 1)),
        out_channels=int(saved.get("out_channels", 1)),
        channels=chs,
        strides=sts,
        num_res_units=int(saved.get("num_res_units", 0)),
    )
    state = ckpt.get("model")
    if state is None:
        raise KeyError("Checkpoint missing 'model'")
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    print(f"  Loaded MONAI AE {checkpoint} channels={chs} strides={sts}", flush=True)
    return net


@torch.no_grad()
def monai_bottleneck_spatial_tokens(model: nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    z = model.encode(batch)
    z = model.intermediate(z)
    if z.dim() != 4:
        raise RuntimeError(f"Expected (B,C,H,W), got {tuple(z.shape)}")
    _b, _c, hp, wp = z.shape
    tok = z.flatten(2).transpose(1, 2).contiguous()
    return tok, int(hp), int(wp)


def main() -> None:
    p = argparse.ArgumentParser(description="MONAI AE Stage B: spatial tokens + k-means (schema v2)")
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--tts_root", type=str, default="")
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--assign_only_csvs", type=str, nargs="*", default=None)
    p.add_argument("--extra_csvs", type=str, nargs="*", default=None)
    p.add_argument("--expected_shape", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_slices", type=int, default=32)
    p.add_argument("--k_clusters", type=int, default=64)
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--apply_lv_mask", action="store_true")
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz")
    p.add_argument("--verify_nifti_slice_geometry", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

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
        print(f"  [legacy] K-means fit n={len(fit_ids)}", flush=True)
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
            f"  fit train+val n={len(fit_ids)}, assign-only n={len(assign_only_ids)}",
            flush=True,
        )

    all_ids = sorted(set(fit_ids) | set(assign_only_ids))
    os.makedirs(args.out_dir, exist_ok=True)
    include_csv = os.path.join(args.out_dir, "combined_include_ids.csv")
    pd.DataFrame({"patient_id": all_ids}).to_csv(include_csv, index=False)

    fit_set = set(fit_ids)
    dataset = _build_dataset(
        normal_root=args.normal_root,
        tts_root=args.tts_root.strip() or None,
        metadata_csv=args.metadata_csv,
        expected_xyz=tuple(args.expected_shape),
        include_csv=include_csv,
        apply_lv_mask=args.apply_lv_mask,
        ct_suffix=args.ct_suffix,
        mask_filename=args.mask_filename,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset.")

    model = _build_monai_ae(args.checkpoint, device)

    fit_embeddings: List[np.ndarray] = []
    fit_meta: List[Tuple[str, int]] = []
    per_patient: Dict[str, Dict[str, Any]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pid = sample["patient_id"]
        x = sample["x"].to(device)
        x_slices = x.permute(1, 0, 2, 3).contiguous()
        d = x_slices.shape[0]

        token_rows_list: List[np.ndarray] = []
        slice_parts: List[np.ndarray] = []
        spatial_parts: List[np.ndarray] = []
        bottleneck_hw: Tuple[int, int] | None = None

        for i in range(0, d, args.batch_slices):
            batch = x_slices[i : i + args.batch_slices]
            bt, hp, wp = monai_bottleneck_spatial_tokens(model, batch)
            if bottleneck_hw is None:
                bottleneck_hw = (hp, wp)
            elif bottleneck_hw != (hp, wp):
                raise RuntimeError(f"bottleneck {bottleneck_hw} vs {(hp, wp)} pid={pid}")

            bt_np = bt.float().cpu().numpy()
            Bb = bt_np.shape[0]
            HW = hp * wp
            assert bt_np.shape[1] == HW
            spat = np.arange(HW, dtype=np.int32)

            for j in range(Bb):
                token_rows_list.append(bt_np[j])
                slice_parts.append(np.full(HW, i + j, dtype=np.int32))
                spatial_parts.append(spat.copy())

        emb_all = np.concatenate(token_rows_list, axis=0)
        embeddings_l2 = l2_normalize_rows(emb_all)
        slice_indices_arr = np.concatenate(slice_parts, axis=0)
        spatial_flat_idx = np.concatenate(spatial_parts, axis=0)

        ct_path = sample.get("ct_path")
        if not ct_path:
            raise RuntimeError(f"[{pid}] missing ct_path")
        cw, _, _, _, shape3 = nifti_per_slice_positions(
            str(ct_path), axis=2, n_slices_expected=d, eps=1e-8
        )
        slice_z_mm_slice = cw[:, 2].astype(np.float32)
        slice_z_mm_tokens = slice_z_mm_slice[slice_indices_arr]

        if args.verify_nifti_slice_geometry and cw.shape[0] >= 2:
            step = np.linalg.norm(np.diff(cw, axis=0), axis=1)
            print(
                f"  [verify] {pid} NIfTI={shape3} dz_mean={float(step.mean()):.5f}",
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
            print(f"  processed {idx + 1}/{len(dataset)}", flush=True)

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
            pdata["cluster_ids"] = (emb_l2 @ prototypes.T).argmax(axis=1).astype(np.int64)

    out_patients = os.path.join(args.out_dir, "patients")
    os.makedirs(out_patients, exist_ok=True)
    np.save(os.path.join(args.out_dir, "prototypes.npy"), prototypes)

    for pid, data in per_patient.items():
        emb = data["embeddings"]
        emb_l2 = data["embeddings_l2"]
        cids = data["cluster_ids"]
        ntok = emb.shape[0]
        d_sl = data["n_slices"]
        hw_arr = data["bottleneck_hw"].astype(np.int32)
        hp_, wp_ = int(hw_arr[0]), int(hw_arr[1])
        hw_sz = hp_ * wp_

        flat = data["spatial_flat_idx"].astype(np.int32)
        spatial_y = (flat // wp_).astype(np.int32)
        spatial_x = (flat % wp_).astype(np.int32)

        slice_pos_rep = np.repeat(np.linspace(0.0, 1.0, d_sl, dtype=np.float32), repeats=hw_sz)
        assert slice_pos_rep.shape[0] == ntok

        np.savez_compressed(
            os.path.join(out_patients, f"{pid}.npz"),
            schema_version=np.int32(SCHEMA_VERSION),
            embeddings=emb.astype(np.float32),
            embeddings_l2=emb_l2.astype(np.float32),
            cluster_ids=cids,
            mask=np.ones(ntok, dtype=np.int64),
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

    print(f"✅ Saved prototypes + schema v{SCHEMA_VERSION} NPZs -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
