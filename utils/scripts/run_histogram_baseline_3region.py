#!/usr/bin/env python3
"""
run_histogram_baseline_3region.py

Baseline (upgraded): patient-level 3-region prototype histograms.
Split slices into base/mid/apex by depth index, compute 64-d histogram per region,
concatenate into 192-d feature, and train logistic regression.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return float(np.trapz(tpr, fpr))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(tps[-1], 1)
    return float(np.trapz(precision, recall))


def build_histogram(cids: np.ndarray, k: int) -> np.ndarray:
    hist = np.zeros((k,), dtype=np.float32)
    for cid in cids:
        if 0 <= int(cid) < k:
            hist[int(cid)] += 1.0
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def split_indices(n_slices: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Depth order is preserved in the slice sequence; split into 3 equal parts.
    idx = np.arange(n_slices)
    s1 = n_slices // 3
    s2 = 2 * n_slices // 3
    return idx[:s1], idx[s1:s2], idx[s2:]


def build_3region_feature(cids: np.ndarray, k: int) -> np.ndarray:
    n = cids.shape[0]
    i1, i2, i3 = split_indices(n)
    h1 = build_histogram(cids[i1], k) if len(i1) > 0 else np.zeros((k,), dtype=np.float32)
    h2 = build_histogram(cids[i2], k) if len(i2) > 0 else np.zeros((k,), dtype=np.float32)
    h3 = build_histogram(cids[i3], k) if len(i3) > 0 else np.zeros((k,), dtype=np.float32)
    return np.concatenate([h1, h2, h3], axis=0)


def load_patient_features(
    npz_dir: Path,
    ids: List[str],
    labels: Dict[str, int],
    k_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    resolved_ids = []
    available = {p.stem: p for p in npz_dir.glob("*.npz")}
    available_lower = {k.lower(): v for k, v in available.items()}

    for pid in ids:
        pid_clean = pid.strip()
        path = None
        if pid_clean in available:
            path = available[pid_clean]
        elif pid_clean.lower() in available_lower:
            path = available_lower[pid_clean.lower()]
        else:
            matches = [v for k, v in available.items() if k.startswith(pid_clean)]
            if len(matches) >= 1:
                path = sorted(matches)[0]

        if path is None:
            continue
        data = np.load(path, allow_pickle=True)
        cids = data["cluster_ids"].astype(np.int64)
        feat = build_3region_feature(cids, k_clusters)
        lbl = resolve_label(pid_clean, labels)
        if lbl is None:
            continue
        X.append(feat)
        y.append(int(lbl))
        resolved_ids.append(pid_clean)

    if not X:
        return np.zeros((0, k_clusters * 3), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.stack(X, axis=0), np.array(y, dtype=np.int64), resolved_ids


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    epochs: int,
    device: torch.device,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict[str, float]]:
    model = nn.Linear(X_train.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_v).view(-1).cpu().numpy()
        train_logits = model(X_t).view(-1).cpu().numpy()

    train_auc = compute_roc_auc(y_train, train_logits)
    train_pr = compute_pr_auc(y_train, train_logits)
    val_auc = compute_roc_auc(y_val, val_logits)
    val_pr = compute_pr_auc(y_val, val_logits)

    metrics = {
        "train_auc": train_auc,
        "train_pr": train_pr,
        "val_auc": val_auc,
        "val_pr": val_pr,
    }
    return model, train_logits, val_logits, metrics


def save_feature_csv(out_path: Path, X: np.ndarray, y: np.ndarray, ids: List[str], k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["patient_id", "label"]
    for region in ["base", "mid", "apex"]:
        header += [f"{region}_p{i:02d}" for i in range(k)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pid, lbl, row in zip(ids, y, X):
            writer.writerow([pid, int(lbl)] + [float(v) for v in row.tolist()])


def plot_region_means(X: np.ndarray, y: np.ndarray, k: int, out_path: Path) -> None:
    if not HAS_MPL or X.size == 0:
        return
    # X shape: (N, 3k)
    def split_region(mat, r):
        return mat[:, r * k:(r + 1) * k]

    mean0 = X[y == 0]
    mean1 = X[y == 1]

    regions = ["base", "mid", "apex"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for r, ax in enumerate(axes):
        m0 = split_region(mean0, r).mean(axis=0) if len(mean0) else np.zeros((k,))
        m1 = split_region(mean1, r).mean(axis=0) if len(mean1) else np.zeros((k,))
        ax.plot(m0, label="Normal (0)")
        ax.plot(m1, label="TTS (1)")
        ax.set_title(f"{regions[r]} mean histogram")
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Prototype ID")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_region_diff(X: np.ndarray, y: np.ndarray, k: int, out_path: Path) -> None:
    if not HAS_MPL or X.size == 0:
        return
    regions = ["base", "mid", "apex"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for r in range(3):
        region_x = X[:, r * k:(r + 1) * k]
        mean0 = region_x[y == 0].mean(axis=0) if (y == 0).any() else np.zeros((k,))
        mean1 = region_x[y == 1].mean(axis=0) if (y == 1).any() else np.zeros((k,))
        ax.plot(mean1 - mean0, label=f"{regions[r]} (TTS - Normal)")
    ax.set_title("Region-wise mean histogram difference")
    ax.set_xlabel("Prototype ID")
    ax.set_ylabel("Delta ratio")
    ax.grid(True, alpha=0.2)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="3-region histogram baseline (prototype ratios)")
    p.add_argument("--npz_dir", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--k_clusters", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.metadata_csv)
    train_ids = load_split_ids(args.train_csv)
    val_ids = load_split_ids(args.val_csv)
    test_ids = load_split_ids(args.test_csv) if args.test_csv else []

    X_train, y_train, train_resolved = load_patient_features(
        Path(args.npz_dir), train_ids, labels, args.k_clusters
    )
    X_val, y_val, val_resolved = load_patient_features(
        Path(args.npz_dir), val_ids, labels, args.k_clusters
    )
    X_test = np.zeros((0, args.k_clusters * 3), dtype=np.float32)
    y_test = np.zeros((0,), dtype=np.int64)
    test_resolved: List[str] = []
    if args.test_csv:
        X_test, y_test, test_resolved = load_patient_features(
            Path(args.npz_dir), test_ids, labels, args.k_clusters
        )

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("Empty train/val after resolving npz files.")

    model, train_logits, val_logits, metrics = train_logreg(
        X_train, y_train, X_val, y_val, args.lr, args.epochs, device
    )

    test_metrics = {}
    if args.test_csv and len(X_test) > 0:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            test_logits = model(X_t).view(-1).cpu().numpy()
        test_auc = compute_roc_auc(y_test, test_logits)
        test_pr = compute_pr_auc(y_test, test_logits)
        test_metrics = {"test_auc": test_auc, "test_pr": test_pr, "test_n": int(len(X_test))}
        save_feature_csv(out_dir / "test_features.csv", X_test, y_test, test_resolved, args.k_clusters)

    # Save outputs
    save_feature_csv(out_dir / "train_features.csv", X_train, y_train, train_resolved, args.k_clusters)
    save_feature_csv(out_dir / "val_features.csv", X_val, y_val, val_resolved, args.k_clusters)

    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    plot_region_means(X_all, y_all, args.k_clusters, out_dir / "mean_histogram_3region.png")
    plot_region_diff(X_all, y_all, args.k_clusters, out_dir / "region_diff.png")

    summary = {
        "npz_dir": args.npz_dir,
        "k_clusters": args.k_clusters,
        "train_n": int(len(X_train)),
        "val_n": int(len(X_val)),
        "metrics": metrics,
    }
    if test_metrics:
        summary["test_n"] = test_metrics["test_n"]
        summary["test_metrics"] = {
            "test_auc": test_metrics["test_auc"],
            "test_pr": test_metrics["test_pr"],
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("# 3-Region Histogram Baseline\\n\\n")
        f.write(f"- Train N: {len(X_train)}\\n")
        f.write(f"- Val N: {len(X_val)}\\n")
        f.write(f"- AUC (train): {metrics['train_auc']:.4f}\\n")
        f.write(f"- PR  (train): {metrics['train_pr']:.4f}\\n")
        f.write(f"- AUC (val): {metrics['val_auc']:.4f}\\n")
        f.write(f"- PR  (val): {metrics['val_pr']:.4f}\\n")
        if test_metrics:
            f.write(f"- AUC (test): {test_metrics['test_auc']:.4f}\\n")
            f.write(f"- PR  (test): {test_metrics['test_pr']:.4f}\\n")
        f.write("\\nOutputs:\\n")
        f.write("- train_features.csv\\n")
        f.write("- val_features.csv\\n")
        if test_metrics:
            f.write("- test_features.csv\\n")
        f.write("- mean_histogram_3region.png\\n")
        f.write("- region_diff.png\\n")


if __name__ == "__main__":
    main()
