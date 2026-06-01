#!/usr/bin/env python3
"""
analyze_cluster_vit_explain.py

Explainability analysis for ClusterViT (Stage C):
1) Attention rollout (last block) -> slice attention heatmap and top-K slices.
2) Prototype contribution analysis using attention-weighted cluster IDs.
3) Evidence sampling: map top-K slices back to CT and save visualizations.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# Add AE_TTS root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)
models_dir = os.path.join(ae_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from data.dataset import load_nii, hu_clip_and_scale, apply_mask
from cluster_vit import ClusterViT


def load_labels(metadata_csv: str) -> Dict[str, int]:
    labels = {}
    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("ID", "")).strip()
            case = row.get("case", "")
            if not pid:
                continue
            v = str(case).strip().lower()
            if v in {"0", "normal", "control", "healthy"}:
                lbl = 0
            elif v in {"1", "tts", "disease", "abnormal"}:
                lbl = 1
            else:
                try:
                    lbl = int(float(v))
                except Exception:
                    lbl = 0
            labels[pid] = lbl
    return labels


def resolve_label(pid: str, labels: Dict[str, int]) -> int | None:
    pid_clean = pid.strip()
    if pid_clean in labels:
        return labels[pid_clean]
    labels_lower = {k.lower(): v for k, v in labels.items()}
    pid_lower = pid_clean.lower()
    if pid_lower in labels_lower:
        return labels_lower[pid_lower]
    matches = [labels[k] for k in labels if pid_clean.startswith(k) or k.startswith(pid_clean)]
    if matches:
        return matches[0]
    return None


def load_split_ids(csv_path: str) -> List[str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if "patient_id" in reader.fieldnames:
            key = "patient_id"
        elif "ID" in reader.fieldnames:
            key = "ID"
        else:
            key = reader.fieldnames[0]
        ids = [str(row.get(key, "")).strip() for row in reader if str(row.get(key, "")).strip()]
    return ids


def find_ct_path(root_dir: str, pid: str, ct_suffix: str) -> str | None:
    if not root_dir:
        return None
    ct_path = os.path.join(root_dir, pid, f"{pid}{ct_suffix}")
    return ct_path if os.path.isfile(ct_path) else None


def load_ct_volume(
    pid: str,
    normal_root: str,
    tts_root: str,
    ct_suffix: str,
    mask_filename: str,
    apply_lv_mask: bool,
) -> np.ndarray | None:
    ct_path = find_ct_path(normal_root, pid, ct_suffix)
    if ct_path is None:
        ct_path = find_ct_path(tts_root, pid, ct_suffix)
    if ct_path is None:
        return None
    ct_xyz = load_nii(ct_path)
    ct_xyz = hu_clip_and_scale(ct_xyz)
    if apply_lv_mask and mask_filename:
        mask_path = os.path.join(os.path.dirname(ct_path), mask_filename)
        if os.path.isfile(mask_path):
            mask_xyz = load_nii(mask_path)
            ct_xyz = apply_mask(ct_xyz, mask_xyz)
    # (X, Y, Z) -> (Z, Y, X)
    ct_zyx = np.transpose(ct_xyz, (2, 1, 0))
    return ct_zyx


def split_region(index: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    s1 = total // 3
    s2 = 2 * total // 3
    if index < s1:
        return "base"
    if index < s2:
        return "mid"
    return "apex"


def compute_slice_attention(
    attn: torch.Tensor,
    cluster_weights: torch.Tensor,
    cluster_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    attn: (B, H, K, N) from last block (cluster -> token)
    cluster_weights: (K,)
    cluster_ids: (B, N)
    mask: (B, N)
    """
    # Mean over heads
    attn_mean = attn.mean(dim=1)  # (B, K, N)
    # Cluster frequency q_k per patient
    b, n = cluster_ids.shape
    k = cluster_weights.shape[0]
    mask_f = mask.float()
    counts = torch.zeros(b, k, device=cluster_ids.device, dtype=attn_mean.dtype)
    counts = counts.scatter_add(1, cluster_ids, mask_f)
    q_k = counts / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    A_k = torch.softmax(cluster_weights, dim=0)[None, :, None]  # (1, K, 1)
    # Weighted slice attention
    slice_attn = (A_k * q_k[:, :, None] * attn_mean).sum(dim=1)  # (B, N)
    slice_attn = slice_attn * mask_f
    # Normalize per patient
    slice_attn = slice_attn / slice_attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return slice_attn


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> ClusterViT:
    state = torch.load(ckpt_path, map_location=device)
    args = state.get("args", {})
    model = ClusterViT(
        input_dim=int(args.get("input_dim", 512)),
        model_dim=int(args.get("model_dim", 512)),
        k_clusters=int(args.get("k_clusters", 64)),
        depth=int(args.get("depth", 6)),
        heads=int(args.get("heads", 8)),
        dropout=float(args.get("dropout", 0.1)),
        max_len=int(args.get("max_len", 256)),
    ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Explain ClusterViT attention and prototype contributions")
    p.add_argument("--npz_dir", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="Stage C checkpoint (cluster_vit_best.pth)")
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--tts_root", type=str, required=True)
    p.add_argument("--ct_suffix", type=str, default="_roi.nii.gz")
    p.add_argument("--mask_filename", type=str, default="heart_ventricle_left_roi.nii.gz")
    p.add_argument("--apply_lv_mask", action="store_true")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.metadata_csv)
    train_ids = load_split_ids(args.train_csv)
    val_ids = load_split_ids(args.val_csv)
    if args.split == "train":
        target_ids = train_ids
    elif args.split == "val":
        target_ids = val_ids
    else:
        target_ids = sorted(set(train_ids + val_ids))

    model = load_model_from_ckpt(args.ckpt, device=device)
    k_clusters = model.k_clusters

    npz_dir = Path(args.npz_dir)
    available = {p.stem: p for p in npz_dir.glob("*.npz")}
    available_lower = {k.lower(): v for k, v in available.items()}

    # Aggregates
    attn_sum_tts = None
    attn_sum_norm = None
    attn_cnt_tts = 0
    attn_cnt_norm = 0
    proto_contrib_tts = np.zeros((k_clusters,), dtype=np.float64)
    proto_contrib_norm = np.zeros((k_clusters,), dtype=np.float64)
    region_counts = {r: np.zeros((k_clusters,), dtype=np.int64) for r in ["base", "mid", "apex"]}

    topk_rows = []
    fingerprint_rows = []

    for pid in target_ids:
        pid_clean = pid.strip()
        npz_path = None
        if pid_clean in available:
            npz_path = available[pid_clean]
        elif pid_clean.lower() in available_lower:
            npz_path = available_lower[pid_clean.lower()]
        else:
            matches = [v for k, v in available.items() if k.startswith(pid_clean)]
            if len(matches) >= 1:
                npz_path = sorted(matches)[0]
        if npz_path is None:
            continue

        data = np.load(npz_path, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
        cids = data["cluster_ids"].astype(np.int64)
        mask = data["mask"].astype(np.int64)
        lbl = resolve_label(pid_clean, labels)
        if lbl is None:
            continue

        # Build batch tensors
        x = torch.tensor(emb[None, ...], device=device)
        cluster_ids = torch.tensor(cids[None, ...], device=device)
        mask_t = torch.tensor(mask[None, ...], device=device)

        with torch.no_grad():
            _ = model(x, cluster_ids, mask_t)
            last_attn = model.blocks[-1].attn.last_attn  # (B, H, K, N)
            if last_attn is None:
                continue
            slice_attn = compute_slice_attention(
                last_attn, model.cluster_weights, cluster_ids, mask_t
            )[0].cpu().numpy()

        valid_idx = np.where(mask > 0)[0]
        if len(valid_idx) == 0:
            continue

        # Top-K slices
        order = np.argsort(-slice_attn[valid_idx])
        top_idx = valid_idx[order[: args.top_k]]
        top_scores = slice_attn[top_idx]

        topk_rows.append({
            "patient_id": pid_clean,
            "label": int(lbl),
            "topk_indices": ",".join([str(int(i)) for i in top_idx.tolist()]),
            "topk_scores": ",".join([f"{float(s):.6f}" for s in top_scores.tolist()]),
        })

        # Prototype contribution per patient (sum attention for slices per proto)
        contrib = np.zeros((k_clusters,), dtype=np.float64)
        for i in valid_idx:
            contrib[int(cids[i])] += float(slice_attn[i])

        # Top-5 prototypes for fingerprint
        top_proto = np.argsort(-contrib)[:5]
        fingerprint_rows.append({
            "patient_id": pid_clean,
            "label": int(lbl),
            "top5_proto": ",".join([str(int(p)) for p in top_proto.tolist()]),
            "top5_scores": ",".join([f"{float(contrib[p]):.6f}" for p in top_proto.tolist()]),
            "topk_slices": ",".join([str(int(i)) for i in top_idx.tolist()]),
        })

        # Aggregate contributions
        if lbl == 1:
            proto_contrib_tts += contrib
        else:
            proto_contrib_norm += contrib

        # Region counts (top prototypes by contribution for this patient)
        for p in top_proto:
            for i in valid_idx:
                if int(cids[i]) == int(p):
                    region = split_region(int(i), len(valid_idx))
                    region_counts[region][int(p)] += 1

        # Mean attention per depth
        if lbl == 1:
            if attn_sum_tts is None:
                attn_sum_tts = slice_attn.copy()
            else:
                attn_sum_tts = attn_sum_tts + slice_attn
            attn_cnt_tts += 1
        else:
            if attn_sum_norm is None:
                attn_sum_norm = slice_attn.copy()
            else:
                attn_sum_norm = attn_sum_norm + slice_attn
            attn_cnt_norm += 1

        # Evidence sampling
        if HAS_MPL:
            ct = load_ct_volume(
                pid_clean,
                args.normal_root,
                args.tts_root,
                args.ct_suffix,
                args.mask_filename,
                args.apply_lv_mask,
            )
            if ct is not None:
                fig, axes = plt.subplots(1, len(top_idx), figsize=(3 * len(top_idx), 3))
                if len(top_idx) == 1:
                    axes = [axes]
                for ax, idx in zip(axes, top_idx):
                    if idx < ct.shape[0]:
                        ax.imshow(ct[idx], cmap="gray")
                        ax.set_title(f"slice {idx}")
                        ax.axis("off")
                fig.suptitle(f"{pid_clean} (label={lbl})")
                fig.tight_layout()
                out_path = out_dir / "evidence"
                out_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path / f"{pid_clean}_topk.png", dpi=200)
                plt.close(fig)

    # Save attention mean plots
    if HAS_MPL and attn_sum_tts is not None and attn_sum_norm is not None:
        mean_tts = attn_sum_tts / max(attn_cnt_tts, 1)
        mean_norm = attn_sum_norm / max(attn_cnt_norm, 1)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(mean_norm, label="Normal")
        ax.plot(mean_tts, label="TTS")
        ax.set_xlabel("Slice index (depth)")
        ax.set_ylabel("Mean attention")
        ax.set_title("Mean attention over depth (last block)")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "attention_mean_curve.png", dpi=200)
        plt.close(fig)

        if HAS_SNS:
            sns.set(style="white")
            heat = np.stack([mean_norm, mean_tts], axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(10, 2))
            sns.heatmap(heat, cmap="viridis", ax=ax, cbar=True)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["Normal", "TTS"])
            ax.set_xlabel("Slice index")
            ax.set_title("Attention heatmap (mean)")
            fig.tight_layout()
            fig.savefig(out_dir / "attention_heatmap.png", dpi=200)
            plt.close(fig)

    # Prototype contribution summary
    contrib_rows = []
    for pid in range(k_clusters):
        contrib_rows.append({
            "prototype_id": pid,
            "attn_sum_tts": float(proto_contrib_tts[pid]),
            "attn_sum_normal": float(proto_contrib_norm[pid]),
            "delta_tts_minus_normal": float(proto_contrib_tts[pid] - proto_contrib_norm[pid]),
        })

    contrib_rows.sort(key=lambda r: r["attn_sum_tts"], reverse=True)
    top5 = [r["prototype_id"] for r in contrib_rows[:5]]

    region_rows = []
    for p in top5:
        for region in ["base", "mid", "apex"]:
            region_rows.append({
                "prototype_id": int(p),
                "region": region,
                "count": int(region_counts[region][p]),
            })

    # Save CSVs
    with open(out_dir / "topk_slices.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()) if topk_rows else [])
        if topk_rows:
            writer.writeheader()
            writer.writerows(topk_rows)

    with open(out_dir / "disease_fingerprint.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fingerprint_rows[0].keys()) if fingerprint_rows else [])
        if fingerprint_rows:
            writer.writeheader()
            writer.writerows(fingerprint_rows)

    with open(out_dir / "prototype_contribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(contrib_rows[0].keys()) if contrib_rows else [])
        if contrib_rows:
            writer.writeheader()
            writer.writerows(contrib_rows)

    with open(out_dir / "prototype_region_distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()) if region_rows else [])
        if region_rows:
            writer.writeheader()
            writer.writerows(region_rows)

    # Save summary
    summary = {
        "top5_prototypes_by_tts_attention": top5,
        "n_patients": len(topk_rows),
        "split": args.split,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("# ClusterViT Explainability Report\\n\\n")
        f.write(f"- Patients analyzed: {len(topk_rows)}\\n")
        f.write(f"- Split: {args.split}\\n")
        f.write(f"- Top-5 prototypes (TTS attention): {top5}\\n")
        f.write("\\nOutputs:\\n")
        f.write("- attention_mean_curve.png / attention_heatmap.png\\n")
        f.write("- topk_slices.csv\\n")
        f.write("- prototype_contribution.csv\\n")
        f.write("- prototype_region_distribution.csv\\n")
        f.write("- disease_fingerprint.csv\\n")
        f.write("- evidence/*.png\\n")


if __name__ == "__main__":
    main()
