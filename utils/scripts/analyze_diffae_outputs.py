#!/usr/bin/env python3
"""
analyze_diffae_outputs.py

Comprehensive analysis for DiffAE Stage B/C outputs:
- Stage B coverage, cluster stats, prototype similarity
- Embedding visualization (PCA; UMAP/TSNE if available)
- Prototype-distance anomaly score (per-slice and per-patient)
- Stage C metrics summary (best/last)
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN_METRICS = True
except Exception:
    HAS_SKLEARN_METRICS = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def list_npz_ids(npz_dir: Path) -> List[Path]:
    return sorted([p for p in npz_dir.glob("*.npz")])


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def pca_2d(x: np.ndarray) -> np.ndarray:
    # center
    x = x - x.mean(axis=0, keepdims=True)
    # SVD
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def save_scatter(points: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mask1 = labels == 1
    mask0 = labels == 0
    ax.scatter(points[mask0, 0], points[mask0, 1], s=8, alpha=0.6, c="tab:blue", label="Normal (0)")
    ax.scatter(points[mask1, 0], points[mask1, 1], s=8, alpha=0.6, c="tab:red", label="TTS (1)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_loss_history(loss_csv: Path) -> Dict:
    if not loss_csv.exists():
        return {"path": str(loss_csv), "missing": True}
    rows = []
    with open(loss_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in list(r.keys()):
                try:
                    r[k] = float(r[k])
                except Exception:
                    pass
            rows.append(r)
    if not rows:
        return {"path": str(loss_csv), "empty": True}
    if "val_auc" in rows[0]:
        best = max(rows, key=lambda r: r.get("val_auc", float("-inf")))
    else:
        best = min(rows, key=lambda r: r.get("val_loss", float("inf")))
    last = rows[-1]
    return {
        "path": str(loss_csv),
        "best_epoch": int(best.get("epoch", 0)),
        "best_val_auc": best.get("val_auc"),
        "best_val_pr": best.get("val_pr"),
        "best_val_loss": best.get("val_loss"),
        "last_epoch": int(last.get("epoch", 0)),
        "last_val_auc": last.get("val_auc"),
        "last_val_pr": last.get("val_pr"),
        "last_val_loss": last.get("val_loss"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze DiffAE Stage B/C outputs")
    p.add_argument("--stageb_dir", type=str, required=True, help="Stage B output dir (contains patients/)")
    p.add_argument("--stagec_dir", type=str, required=True, help="Stage C output dir (loss_history.csv)")
    p.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV (ID, case)")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_points", type=int, default=5000, help="Max points for plots")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.metadata_csv)
    train_ids = load_split_ids(args.train_csv)
    val_ids = load_split_ids(args.val_csv)

    stageb_dir = Path(args.stageb_dir)
    npz_dir = stageb_dir / "patients"
    npz_files = list_npz_ids(npz_dir)

    # Coverage stats
    npz_ids = {p.stem for p in npz_files}
    train_missing = [pid for pid in train_ids if pid not in npz_ids]
    val_missing = [pid for pid in val_ids if pid not in npz_ids]

    # Load embeddings
    all_embeddings = []
    all_labels = []
    all_pids = []
    per_patient_stats = []
    cluster_counts = None
    proto_path = stageb_dir / "prototypes.npy"
    prototypes = np.load(proto_path) if proto_path.exists() else None

    for p in npz_files:
        data = np.load(p, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
        cids = data["cluster_ids"].astype(np.int64)
        pid = str(data.get("patient_id", p.stem))
        lbl = resolve_label(pid, labels)
        if lbl is None:
            continue

        if cluster_counts is None:
            cluster_counts = np.zeros(int(cids.max()) + 1, dtype=np.int64)
        for cid in cids:
            if cid < len(cluster_counts):
                cluster_counts[cid] += 1

        all_embeddings.append(emb)
        all_labels.append(np.full((emb.shape[0],), int(lbl), dtype=np.int64))
        all_pids.extend([pid] * emb.shape[0])

        # per-patient stats
        uniq, counts = np.unique(cids, return_counts=True)
        probs = counts / max(counts.sum(), 1)
        entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        per_patient_stats.append(
            {
                "patient_id": pid,
                "label": int(lbl),
                "n_slices": int(emb.shape[0]),
                "cluster_entropy": entropy,
            }
        )

    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_embeddings = np.zeros((0, 1), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)

    # Prototype similarity stats
    proto_stats = {}
    if prototypes is not None and prototypes.size > 0:
        proto_norm = l2_normalize(prototypes)
        sims = proto_norm @ proto_norm.T
        # exclude diagonal
        mask = ~np.eye(sims.shape[0], dtype=bool)
        sim_vals = sims[mask]
        proto_stats = {
            "min_cos_sim": float(sim_vals.min()),
            "max_cos_sim": float(sim_vals.max()),
            "mean_cos_sim": float(sim_vals.mean()),
        }

    # Embedding visualization
    plot_dir = out_dir / "plots"
    if all_embeddings.shape[0] > 0:
        n = min(args.max_points, all_embeddings.shape[0])
        idx = np.random.choice(all_embeddings.shape[0], size=n, replace=False)
        emb_sample = all_embeddings[idx]
        lbl_sample = all_labels[idx]

        # PCA
        emb_pca = pca_2d(emb_sample)
        save_scatter(emb_pca, lbl_sample, plot_dir / "pca.png", "PCA (slice embeddings)")

        # UMAP
        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=args.seed, n_neighbors=15, min_dist=0.1)
            emb_umap = reducer.fit_transform(emb_sample)
            save_scatter(emb_umap, lbl_sample, plot_dir / "umap.png", "UMAP (slice embeddings)")

        # t-SNE
        if HAS_TSNE:
            tsne = TSNE(n_components=2, random_state=args.seed, init="pca", perplexity=30)
            emb_tsne = tsne.fit_transform(emb_sample)
            save_scatter(emb_tsne, lbl_sample, plot_dir / "tsne.png", "t-SNE (slice embeddings)")

    # Prototype-distance anomaly score
    scores = []
    if prototypes is not None and all_embeddings.shape[0] > 0:
        proto_norm = l2_normalize(prototypes)
        emb_norm = l2_normalize(all_embeddings)
        sims = emb_norm @ proto_norm.T
        min_dist = 1.0 - sims.max(axis=1)
        # aggregate by patient
        pid_scores = {}
        pid_labels = {}
        for s, pid, lbl in zip(min_dist, all_pids, all_labels):
            pid_scores.setdefault(pid, []).append(float(s))
            pid_labels[pid] = int(lbl)
        for pid, vals in pid_scores.items():
            scores.append(
                {
                    "patient_id": pid,
                    "label": pid_labels.get(pid, 0),
                    "mean_min_proto_dist": float(np.mean(vals)),
                    "median_min_proto_dist": float(np.median(vals)),
                }
            )

        # compute AUC/PR if possible
        if HAS_SKLEARN_METRICS:
            y_true = np.array([s["label"] for s in scores], dtype=np.int64)
            y_score = np.array([s["mean_min_proto_dist"] for s in scores], dtype=np.float32)
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_score))
                pr = float(average_precision_score(y_true, y_score))
            else:
                auc, pr = None, None
        else:
            auc, pr = None, None
    else:
        auc, pr = None, None

    # Save CSVs
    if per_patient_stats:
        with open(out_dir / "per_patient_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_patient_stats[0].keys()))
            writer.writeheader()
            writer.writerows(per_patient_stats)
    if scores:
        with open(out_dir / "prototype_distance_scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
            writer.writeheader()
            writer.writerows(scores)

    # Stage C summary
    stagec_summary = summarize_loss_history(Path(args.stagec_dir) / "loss_history.csv")

    # Write summary JSON
    summary = {
        "stageb_dir": str(stageb_dir),
        "npz_count": len(npz_files),
        "train_missing_npz": len(train_missing),
        "val_missing_npz": len(val_missing),
        "train_missing_examples": train_missing[:5],
        "val_missing_examples": val_missing[:5],
        "embedding_dim": int(all_embeddings.shape[1]) if all_embeddings.size > 0 else None,
        "prototypes_shape": tuple(prototypes.shape) if prototypes is not None else None,
        "prototype_similarity": proto_stats,
        "prototype_distance_auc": auc,
        "prototype_distance_pr": pr,
        "stagec_summary": stagec_summary,
        "plots_dir": str(plot_dir) if HAS_MPL else None,
        "has_umap": HAS_UMAP,
        "has_tsne": HAS_TSNE,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write markdown report
    report_path = out_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("# DiffAE Output Analysis\\n\\n")
        f.write("## Stage B coverage\\n")
        f.write(f"- NPZ count: {len(npz_files)}\\n")
        f.write(f"- Train missing NPZ: {len(train_missing)}\\n")
        f.write(f"- Val missing NPZ: {len(val_missing)}\\n\\n")
        f.write("## Prototype stats\\n")
        f.write(f"- prototypes: {summary['prototypes_shape']}\\n")
        if proto_stats:
            f.write(f"- cos_sim min/mean/max: {proto_stats['min_cos_sim']:.4f} / ")
            f.write(f"{proto_stats['mean_cos_sim']:.4f} / {proto_stats['max_cos_sim']:.4f}\\n")
        f.write("\\n## Prototype-distance anomaly score\\n")
        f.write(f"- AUC: {auc}\\n")
        f.write(f"- PR: {pr}\\n\\n")
        f.write("## Stage C summary\\n")
        for k, v in stagec_summary.items():
            f.write(f"- {k}: {v}\\n")
        f.write("\\n## Plots\\n")
        if HAS_MPL:
            f.write(f"- {plot_dir}\\n")
        else:
            f.write("- matplotlib not available; no plots generated.\\n")


if __name__ == "__main__":
    main()
