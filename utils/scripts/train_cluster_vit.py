#!/usr/bin/env python3
"""
train_cluster_vit.py
- Stage C: Train clustering ViT on slice embeddings for patient classification.
"""

import os
import json
import sys
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add AE_TTS root and models to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ae_root = os.path.dirname(script_dir)
if ae_root not in sys.path:
    sys.path.insert(0, ae_root)
models_dir = os.path.join(ae_root, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from cluster_vit import ClusterViT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_case_value(val) -> int:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"0", "normal", "control", "healthy"}:
            return 0
        if v in {"1", "tts", "disease", "abnormal"}:
            return 1
        try:
            return int(float(v))
        except ValueError:
            return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def load_labels(metadata_csv: str) -> Dict[str, int]:
    df = pd.read_csv(metadata_csv)
    if "ID" not in df.columns or "case" not in df.columns:
        raise ValueError("metadata_csv must include columns: ID, case")
    labels = {str(row["ID"]).strip(): _normalize_case_value(row["case"]) for _, row in df.iterrows()}
    return labels


def load_split_ids(csv_path: str, normal_only: bool = False) -> List[str]:
    df = pd.read_csv(csv_path)
    id_col = None
    if "patient_id" in df.columns:
        id_col = "patient_id"
    elif "ID" in df.columns:
        id_col = "ID"
    else:
        id_col = df.columns[0]

    if normal_only and "case" in df.columns:
        df = df[df["case"].astype(int) == 0]

    return df[id_col].astype(str).str.strip().tolist()


def resolve_label(pid: str, labels: Dict[str, int], labels_lower: Dict[str, int]) -> int | None:
    pid_clean = pid.strip()
    if pid_clean in labels:
        return labels[pid_clean]
    pid_lower = pid_clean.lower()
    if pid_lower in labels_lower:
        return labels_lower[pid_lower]

    # Prefix matching in either direction
    matches = [labels[k] for k in labels if pid_clean.startswith(k) or k.startswith(pid_clean)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]
    return None


def filter_ids_by_label(ids: List[str], labels: Dict[str, int], target_label: int) -> List[str]:
    """
    Keep only IDs that exist in labels dict and match target_label.
    IDs with missing labels are skipped.
    """
    labels_lower = {k.lower(): v for k, v in labels.items()}
    filtered = []
    for pid in ids:
        lbl = resolve_label(pid, labels, labels_lower)
        if lbl is None:
            continue
        if int(lbl) == target_label:
            filtered.append(pid)
    return filtered


class PatientEmbeddingDataset(Dataset):
    def __init__(
        self,
        npz_dir: str,
        ids: List[str],
        labels: Dict[str, int],
        allow_missing: bool = True,
        missing_csv: str = None,
    ):
        self.npz_dir = npz_dir
        self.labels = labels
        self.labels_lower = {k.lower(): v for k, v in labels.items()}
        self.allow_missing = allow_missing

        # Build lookup table of available files
        available = {}
        for name in os.listdir(npz_dir):
            if name.endswith(".npz"):
                key = name[:-4]
                available[key] = name

        available_lower = {k.lower(): v for k, v in available.items()}

        # Resolve each patient ID to an existing npz file
        resolved = []
        missing_ids = []
        for pid in ids:
            pid_clean = pid.strip()
            file_name = None

            # 1) Exact match
            if pid_clean in available:
                file_name = available[pid_clean]
            # 2) Case-insensitive match
            elif pid_clean.lower() in available_lower:
                file_name = available_lower[pid_clean.lower()]
            else:
                # 3) Prefix match (if CSV has shorter ID)
                matches = [v for k, v in available.items() if k.startswith(pid_clean)]
                if len(matches) == 1:
                    file_name = matches[0]
                elif len(matches) > 1:
                    matches.sort()
                    file_name = matches[0]
                    print(f"⚠️  Multiple matches for {pid_clean}. Using {file_name}.")

            if file_name is None:
                missing_ids.append(pid)
            else:
                resolved.append((pid, file_name))

        if missing_ids and not allow_missing:
            sample = ", ".join(missing_ids[:5])
            raise FileNotFoundError(f"Missing {len(missing_ids)} npz files. Example: {sample}")

        if missing_ids and missing_csv:
            os.makedirs(os.path.dirname(missing_csv) or ".", exist_ok=True)
            with open(missing_csv, "w") as f:
                f.write("patient_id\n")
                for pid in missing_ids:
                    f.write(f"{pid}\n")

        if missing_ids:
            print(f"⚠️  Skipping {len(missing_ids)} patients without npz files.")

        self.samples = resolved

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pid, file_name = self.samples[idx]
        npz_path = os.path.join(self.npz_dir, file_name)
        data = np.load(npz_path, allow_pickle=True)
        # Each patient file stores variable-length slice sequence
        emb = data["embeddings"].astype(np.float32)  # (D, 512)
        cids = data["cluster_ids"].astype(np.int64)  # (D,)
        mask = data["mask"].astype(np.int64)         # (D,)
        label = resolve_label(pid, self.labels, self.labels_lower)
        if label is None:
            label = 0
        return {
            "embeddings": emb,
            "cluster_ids": cids,
            "mask": mask,
            "label": label,
            "patient_id": pid,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Pad sequences to the max length in this batch and build a mask
    max_len = max(item["embeddings"].shape[0] for item in batch)
    b = len(batch)
    dim = batch[0]["embeddings"].shape[1]

    emb = np.zeros((b, max_len, dim), dtype=np.float32)
    cids = np.zeros((b, max_len), dtype=np.int64)
    mask = np.zeros((b, max_len), dtype=np.int64)
    labels = np.zeros((b,), dtype=np.float32)
    pids = []

    for i, item in enumerate(batch):
        d = item["embeddings"].shape[0]
        emb[i, :d] = item["embeddings"]
        cids[i, :d] = item["cluster_ids"]
        mask[i, :d] = item["mask"]
        labels[i] = item["label"]
        pids.append(item["patient_id"])

    return {
        "embeddings": torch.tensor(emb),
        "cluster_ids": torch.tensor(cids),
        "mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
        "patient_ids": pids,
    }


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple AUC via trapezoidal integration (no sklearn dependency)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return np.trapz(tpr, fpr)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Precision-recall AUC (no sklearn dependency)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(tps[-1], 1)
    return np.trapz(precision, recall)


def run_epoch(
    model: ClusterViT,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = 0.0
    n = 0
    all_logits = []
    all_labels = []

    for batch in loader:
        x = batch["embeddings"].to(device)
        cids = batch["cluster_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Model returns patient-level logits (before sigmoid)
        logits = model(x, cids, mask)
        loss = loss_fn(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if len(all_logits) == 0:
        return 0.0, np.array([]), np.array([])
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return total / max(n, 1), logits, labels


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ClusterViT for patient classification")
    p.add_argument("--npz_dir", type=str, required=True, help="Directory with per-patient .npz files")
    p.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV with labels (ID, case)")
    p.add_argument("--train_csv", type=str, required=True, help="Train split CSV")
    p.add_argument("--val_csv", type=str, required=True, help="Val split CSV")
    p.add_argument("--test_csv", type=str, default=None, help="Optional test split CSV")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")

    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--allow_missing", action="store_true", help="Skip patients with missing npz files")
    p.add_argument("--missing_csv", type=str, default=None, help="Optional CSV path to save missing IDs")
    p.add_argument("--normal_only", action="store_true", help="Use only normal (case=0) patients")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 to disable)")

    p.add_argument("--input_dim", type=int, default=512, help="Input embedding dim")
    p.add_argument("--model_dim", type=int, default=512, help="Transformer model dim")
    p.add_argument("--k_clusters", type=int, default=64, help="Number of clusters")
    p.add_argument("--depth", type=int, default=6, help="Transformer layers")
    p.add_argument("--heads", type=int, default=8, help="Attention heads")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    p.add_argument("--max_len", type=int, default=256, help="Max sequence length for positional emb")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_labels(args.metadata_csv)
    train_ids = load_split_ids(args.train_csv, normal_only=args.normal_only)
    val_ids = load_split_ids(args.val_csv, normal_only=args.normal_only)
    test_ids = load_split_ids(args.test_csv, normal_only=args.normal_only) if args.test_csv else []

    if args.normal_only:
        train_ids = filter_ids_by_label(train_ids, labels, target_label=0)
        val_ids = filter_ids_by_label(val_ids, labels, target_label=0)
        print(f"⚠️  normal_only enabled: train={len(train_ids)}, val={len(val_ids)}")

    ds_train = PatientEmbeddingDataset(
        args.npz_dir,
        train_ids,
        labels,
        allow_missing=args.allow_missing,
        missing_csv=args.missing_csv,
    )
    ds_val = PatientEmbeddingDataset(
        args.npz_dir,
        val_ids,
        labels,
        allow_missing=args.allow_missing,
        missing_csv=args.missing_csv,
    )
    ds_test = None
    if args.test_csv:
        ds_test = PatientEmbeddingDataset(
            args.npz_dir,
            test_ids,
            labels,
            allow_missing=args.allow_missing,
            missing_csv=args.missing_csv,
        )

    if len(ds_train) == 0:
        raise RuntimeError("Train dataset is empty after resolving npz files.")
    if len(ds_val) == 0:
        print("⚠️  Val dataset is empty after resolving npz files. Skipping validation.")

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    loader_test = None
    if ds_test is not None:
        loader_test = DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    model = ClusterViT(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        k_clusters=args.k_clusters,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    loss_csv = os.path.join(args.out_dir, "loss_history.csv")
    best_ckpt = os.path.join(args.out_dir, "cluster_vit_best.pth")

    best_val_auc = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loader_train, device, loss_fn, optimizer)
        if len(ds_val) > 0:
            val_loss, val_logits, val_labels = run_epoch(model, loader_val, device, loss_fn, optimizer=None)
            val_probs = 1 / (1 + np.exp(-val_logits))
            val_auc = compute_roc_auc(val_labels, val_probs)
            val_pr = compute_pr_auc(val_labels, val_probs)
        else:
            val_loss, val_auc, val_pr = 0.0, 0.0, 0.0

        print(f"[epoch {epoch:03d}/{args.epochs}] "
              f"train={train_loss:.6f} val={val_loss:.6f} "
              f"ROC-AUC={val_auc:.4f} PR-AUC={val_pr:.4f}")

        history.append((epoch, train_loss, val_loss, val_auc, val_pr))

        if len(ds_val) > 0 and val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_auc": best_val_auc,
            }, best_ckpt)
            print(f"💾 saved best checkpoint: {best_ckpt}")
        else:
            epochs_no_improve += 1

        with open(loss_csv, "w") as f:
            f.write("epoch,train_loss,val_loss,val_auc,val_pr\n")
            for e, tr, va, auc, pr in history:
                f.write(f"{e},{tr},{va},{auc},{pr}\n")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"⏹️  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print("✅ Training complete!")
    print(f"Best checkpoint: {best_ckpt}")

    # Optional test evaluation using best checkpoint (kept separate from tuning)
    if loader_test is not None and os.path.exists(best_ckpt):
        print("🔎 Evaluating on test split with best checkpoint...")
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        _, test_logits, test_labels = run_epoch(model, loader_test, device, loss_fn, optimizer=None)
        if len(test_labels) > 0:
            test_probs = 1 / (1 + np.exp(-test_logits))
            test_auc = compute_roc_auc(test_labels, test_probs)
            test_pr = compute_pr_auc(test_labels, test_probs)
        else:
            test_auc, test_pr = 0.0, 0.0
        test_path = os.path.join(args.out_dir, "test_metrics.json")
        with open(test_path, "w") as f:
            json.dump(
                {
                    "test_auc": float(test_auc),
                    "test_pr": float(test_pr),
                    "test_n": int(len(test_labels)),
                    "ckpt": best_ckpt,
                },
                f,
                indent=2,
            )
        print(f"✅ Test metrics saved: {test_path}")


if __name__ == "__main__":
    main()
