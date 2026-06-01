#!/usr/bin/env python3
"""
run_continuity_baseline.py

Continuity feature baseline:
- Compute prototype run-length stats per patient from slice-ordered cluster_ids.
- Features: histogram (k) + continuity (3k: max_run, mean_run, has_run>=L).
- Compare histogram-only vs histogram+continuity with logistic regression.
"""

import argparse
import csv
import json
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


def _runs_for_proto(cids: np.ndarray, proto_id: int) -> List[int]:
    runs = []
    current = 0
    for cid in cids:
        if int(cid) == proto_id:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)
    return runs


def continuity_features(cids: np.ndarray, k: int, run_min_len: int) -> np.ndarray:
    max_runs = np.zeros((k,), dtype=np.float32)
    mean_runs = np.zeros((k,), dtype=np.float32)
    has_runs = np.zeros((k,), dtype=np.float32)
    for pid in range(k):
        runs = _runs_for_proto(cids, pid)
        if runs:
            max_runs[pid] = float(max(runs))
            mean_runs[pid] = float(sum(runs) / len(runs))
            has_runs[pid] = 1.0 if max(runs) >= run_min_len else 0.0
    # normalize run lengths by sequence length to keep scale stable
    denom = max(len(cids), 1)
    max_runs /= denom
    mean_runs /= denom
    return np.concatenate([max_runs, mean_runs, has_runs], axis=0)


def load_patient_features(
    npz_dir: Path,
    ids: List[str],
    labels: Dict[str, int],
    k_clusters: int,
    run_min_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_hist = []
    X_cont = []
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
        hist = build_histogram(cids, k_clusters)
        cont = continuity_features(cids, k_clusters, run_min_len)
        lbl = resolve_label(pid_clean, labels)
        if lbl is None:
            continue
        X_hist.append(hist)
        X_cont.append(cont)
        y.append(int(lbl))
        resolved_ids.append(pid_clean)

    if not X_hist:
        return (
            np.zeros((0, k_clusters), dtype=np.float32),
            np.zeros((0, k_clusters * 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return (
        np.stack(X_hist, axis=0),
        np.stack(X_cont, axis=0),
        np.array(y, dtype=np.int64),
        resolved_ids,
    )


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
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

    return {
        "train_auc": compute_roc_auc(y_train, train_logits),
        "train_pr": compute_pr_auc(y_train, train_logits),
        "val_auc": compute_roc_auc(y_val, val_logits),
        "val_pr": compute_pr_auc(y_val, val_logits),
    }


def save_feature_csv(out_path: Path, X: np.ndarray, y: np.ndarray, ids: List[str], prefix: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["patient_id", "label"] + [f"{prefix}_{i:03d}" for i in range(X.shape[1])]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pid, lbl, row in zip(ids, y, X):
            writer.writerow([pid, int(lbl)] + [float(v) for v in row.tolist()])


def plot_top_proto_runs(
    cont_feats: np.ndarray,
    y: np.ndarray,
    k: int,
    out_path: Path,
    top_n: int = 10,
) -> None:
    if not HAS_MPL or cont_feats.size == 0:
        return
    max_runs = cont_feats[:, :k]
    mean0 = max_runs[y == 0].mean(axis=0) if (y == 0).any() else np.zeros((k,))
    mean1 = max_runs[y == 1].mean(axis=0) if (y == 1).any() else np.zeros((k,))
    delta = mean1 - mean0
    idx = np.argsort(-np.abs(delta))[:top_n]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(np.arange(len(idx)) - 0.2, mean0[idx], width=0.4, label="Normal (0)")
    ax.bar(np.arange(len(idx)) + 0.2, mean1[idx], width=0.4, label="TTS (1)")
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels([str(i) for i in idx])
    ax.set_xlabel("Prototype ID (top |delta|)")
    ax.set_ylabel("Mean max run (normalized)")
    ax.set_title("Prototypes with longest run differences")
    ax.legend()
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Continuity feature baseline")
    p.add_argument("--npz_dir", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--k_clusters", type=int, default=64)
    p.add_argument("--run_min_len", type=int, default=2)
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

    Xh_train, Xc_train, y_train, train_resolved = load_patient_features(
        Path(args.npz_dir), train_ids, labels, args.k_clusters, args.run_min_len
    )
    Xh_val, Xc_val, y_val, val_resolved = load_patient_features(
        Path(args.npz_dir), val_ids, labels, args.k_clusters, args.run_min_len
    )

    if len(Xh_train) == 0 or len(Xh_val) == 0:
        raise RuntimeError("Empty train/val after resolving npz files.")

    # Histogram-only
    metrics_hist = train_logreg(Xh_train, y_train, Xh_val, y_val, args.lr, args.epochs, device)

    # Histogram + continuity
    Xhc_train = np.concatenate([Xh_train, Xc_train], axis=1)
    Xhc_val = np.concatenate([Xh_val, Xc_val], axis=1)
    metrics_hist_cont = train_logreg(Xhc_train, y_train, Xhc_val, y_val, args.lr, args.epochs, device)

    # Save features
    save_feature_csv(out_dir / "train_hist.csv", Xh_train, y_train, train_resolved, "hist")
    save_feature_csv(out_dir / "val_hist.csv", Xh_val, y_val, val_resolved, "hist")
    save_feature_csv(out_dir / "train_continuity.csv", Xc_train, y_train, train_resolved, "cont")
    save_feature_csv(out_dir / "val_continuity.csv", Xc_val, y_val, val_resolved, "cont")
    save_feature_csv(out_dir / "train_hist_cont.csv", Xhc_train, y_train, train_resolved, "hc")
    save_feature_csv(out_dir / "val_hist_cont.csv", Xhc_val, y_val, val_resolved, "hc")

    # Visualization
    Xc_all = np.concatenate([Xc_train, Xc_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    plot_top_proto_runs(Xc_all, y_all, args.k_clusters, out_dir / "top_proto_runs.png", top_n=10)

    summary = {
        "npz_dir": args.npz_dir,
        "k_clusters": args.k_clusters,
        "run_min_len": args.run_min_len,
        "train_n": int(len(Xh_train)),
        "val_n": int(len(Xh_val)),
        "metrics_hist": metrics_hist,
        "metrics_hist_continuity": metrics_hist_cont,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("# Continuity Baseline\\n\\n")
        f.write(f"- Train N: {len(Xh_train)}\\n")
        f.write(f"- Val N: {len(Xh_val)}\\n")
        f.write("\\n## Histogram only\\n")
        f.write(f"- AUC (val): {metrics_hist['val_auc']:.4f}\\n")
        f.write(f"- PR  (val): {metrics_hist['val_pr']:.4f}\\n")
        f.write("\\n## Histogram + continuity\\n")
        f.write(f"- AUC (val): {metrics_hist_cont['val_auc']:.4f}\\n")
        f.write(f"- PR  (val): {metrics_hist_cont['val_pr']:.4f}\\n")
        f.write("\\nOutputs:\\n")
        f.write("- train_hist.csv / val_hist.csv\\n")
        f.write("- train_continuity.csv / val_continuity.csv\\n")
        f.write("- train_hist_cont.csv / val_hist_cont.csv\\n")
        f.write("- top_proto_runs.png\\n")


if __name__ == "__main__":
    main()
