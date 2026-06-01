#!/usr/bin/env python3
"""
Latent space analysis: slice-level NPZ -> patient-level aggregation -> PCA & UMAP.

Designed for medical imaging AE / DiffAE embeddings per patient:
  each .npz: embeddings (N_slices, D), optional cluster_ids, mask.

Usage examples
--------------
  # Single source (point --npz_dir at the folder that contains *.npz, e.g. .../patients)
  python latent_space_pca_umap.py \\
    --npz_dir /Volumes/Chanho_PhD_Project/latent_data/diff3dformer_stageB_plain_ae/patients \\
    --labels_csv ../../data/train.csv --labels_csv ../../data/val.csv --labels_csv ../../data/test.csv \\
    --out_dir ./latent_viz_plain_ae \\
    --pooling mean

  # Compare Plain AE vs DiffAE (side-by-side PCA / UMAP)
  python latent_space_pca_umap.py \\
    --compare_npz_dirs \\
      /Volumes/.../latent_data/diff3dformer_stageB_plain_ae/patients \\
      /Volumes/.../latent_data/diff3dformer_stageB_diffae/patients \\
    --compare_names PlainAE DiffAE \\
    --labels_csv ../../data/train.csv --labels_csv ../../data/val.csv --labels_csv ../../data/test.csv \\
    --out_dir ./latent_viz_compare

  # Sweep UMAP hyperparameters
  python latent_space_pca_umap.py --npz_dir .../patients --labels_csv ... --out_dir ./out \\
    --umap_sweep

  # Mean+std pooling + LogReg / LinearSVM / RBF-SVM (same split as CSVs) + bootstrap + PCA & UMAP (if umap-learn)
  python latent_space_pca_umap.py --eval_classifiers \\
    --split_dir ../../data \\
    --labels_csv ../../data/train.csv --labels_csv ../../data/val.csv --labels_csv ../../data/test.csv \\
    --latent_source plain_ae /path/.../diff3dformer_stageB_plain_ae/patients \\
    --latent_source diffae /path/.../diff3dformer_stageB_diffae/patients \\
    --latent_source plain_mse_ssim /path/.../diff3dformer_stageB_plain_ae_mse_ssim/patients \\
    --latent_source monai_ae /path/.../diff3dformer_stageB_monai_ae/patients \\
    --out_dir ./latent_eval_out --bootstrap_n 1000

  # Or run the bundled helper (defaults: monai_ae only + merge root CSVs; AE_TTS/latent_data → latent_data/out_new):
  #   bash utils/scripts/run_out_new_latent_eval.sh
  #   MODE=full MERGE_EXISTING_ROOT_METRICS=0 bash ...  # recompute all sources; replace root CSVs entirely

  # Diagnostics: label-shuffle null, mean vs std vs mean+std ablation, score histograms, misclassified list
  python latent_space_pca_umap.py --eval_classifiers --eval_diagnostics --label_shuffle_n 500 \\
    --split_dir ../../data --labels_csv ... (x3) --latent_source plain_ae .../patients \\
    --out_dir ./out

Dependencies: numpy, pandas, scikit-learn, matplotlib, umap-learn (optional for UMAP).
  If ``import umap`` fails (often NumPy 2.x vs older TensorFlow), the script still runs; use numpy<2 or upgrade TF.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# umap-learn's top-level __init__ may import ParametricUMAP → TensorFlow; a broken TF/NumPy pair
# (e.g. NumPy 2.x + TF wheels built for NumPy 1.x) raises AttributeError/TypeError, not ImportError.
UMAP_IMPORT_ERROR: str | None = None
try:
    import umap
except Exception as _umap_exc:
    umap = None
    UMAP_IMPORT_ERROR = f"{type(_umap_exc).__name__}: {_umap_exc}"

RANDOM_STATE = 42


def _umap_unavailable_message() -> str:
    base = "UMAP skipped."
    if UMAP_IMPORT_ERROR:
        return (
            f"{base} Import failed: {UMAP_IMPORT_ERROR}\n"
            "  Common fix: use NumPy 1.x for this env, e.g. pip install 'numpy<2.0'\n"
            "  then reinstall TensorFlow / umap-learn, or upgrade TF to a NumPy 2–compatible wheel."
        )
    return f"{base} pip install umap-learn"

# ---------------------------------------------------------------------------
# 1. Load clinical CSV(s) -> patient_id, label, age, sex
# ---------------------------------------------------------------------------


def _merge_labels_and_clinical(paths: list[str]) -> pd.DataFrame:
    """Merge CSVs; expect patient_id + label or case + age + sex."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        id_col = "patient_id" if "patient_id" in df.columns else None
        if id_col is None:
            for c in ("ID", "id"):
                if c in df.columns:
                    id_col = c
                    break
        if id_col is None:
            raise ValueError(f"No patient id column in {p}")

        lab_col = None
        for c in ("label", "case", "Label", "TTS"):
            if c in df.columns:
                lab_col = c
                break
        if lab_col is None:
            raise ValueError(f"No label/case column in {p}")

        if "age" not in df.columns:
            raise ValueError(f"Column 'age' missing in {p}")
        if "sex" not in df.columns:
            raise ValueError(f"Column 'sex' missing in {p}")

        sub = df[[id_col, lab_col, "age", "sex"]].copy()
        sub.columns = ["patient_id", "label_raw", "age", "sex"]
        sub["patient_id"] = sub["patient_id"].astype(str)
        # Normalize label to int 0/1
        def _to_label(v):
            if isinstance(v, (int, np.integer)):
                return int(v)
            s = str(v).strip().lower()
            if s in ("0", "normal", "n"):
                return 0
            if s in ("1", "tts", "t"):
                return 1
            if "tts" in s:
                return 1
            return int(float(v))

        sub["label"] = sub["label_raw"].map(_to_label)
        sub["age"] = pd.to_numeric(sub["age"], errors="coerce")
        sub["sex"] = sub["sex"].astype(str).str.strip().str.upper()
        frames.append(sub[["patient_id", "label", "age", "sex"]])

    out = pd.concat(frames, ignore_index=True)
    # Last occurrence wins on duplicate patient_id (same as training merge)
    out = out.drop_duplicates(subset=["patient_id"], keep="last")
    return out


def discover_npz_files(npz_dir: str) -> list[str]:
    """Return sorted paths to all .npz under npz_dir (non-recursive)."""
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    return [os.path.join(npz_dir, f) for f in names]


@dataclass
class PatientRecord:
    patient_id: str
    embeddings: np.ndarray  # (N, D) slice-level — kept only for optional cluster analysis
    patient_vec: np.ndarray  # (D_agg,) after aggregation
    label: int
    age: float
    sex: str
    dominant_cluster: Optional[int] = None


def load_data(
    npz_dir: str,
    clinical: pd.DataFrame,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[PatientRecord], list[str]]:
    """
    Load each NPZ, match patient_id to clinical table, aggregate slices -> patient vector.

    Returns
    -------
    records : list of PatientRecord
    skipped : list of reasons (missing label, missing file, etc.) for logging
    """
    paths = discover_npz_files(npz_dir)
    clinical = clinical.set_index("patient_id", drop=False)
    records: list[PatientRecord] = []
    skipped: list[str] = []

    for path in paths:
        stem = os.path.basename(path).replace(".npz", "")
        if stem not in clinical["patient_id"].values:
            skipped.append(f"no clinical row: {stem}")
            continue

        row = clinical.loc[stem]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            skipped.append(f"nan age: {stem}")
            continue

        z = np.load(path, allow_pickle=True)
        if "embeddings" not in z.files:
            skipped.append(f"no embeddings key: {stem}")
            continue
        emb = np.asarray(z["embeddings"], dtype=np.float64)
        if emb.ndim != 2:
            skipped.append(f"bad embeddings shape {emb.shape}: {stem}")
            continue

        patient_vec = aggregate_fn(emb)

        dom_c: Optional[int] = None
        if "cluster_ids" in z.files:
            cid = np.asarray(z["cluster_ids"]).reshape(-1)
            if cid.shape[0] == emb.shape[0]:
                vals, counts = np.unique(cid, return_counts=True)
                dom_c = int(vals[np.argmax(counts)])

        rec = PatientRecord(
            patient_id=stem,
            embeddings=emb,
            patient_vec=patient_vec,
            label=int(row["label"]),
            age=float(row["age"]),
            sex=str(row["sex"]),
            dominant_cluster=dom_c,
        )
        records.append(rec)

    return records, skipped


# ---------------------------------------------------------------------------
# 2. Patient-level aggregation (modular)
# ---------------------------------------------------------------------------


def aggregate_mean(emb: np.ndarray) -> np.ndarray:
    """Mean over slices: (N, D) -> (D,)."""
    return emb.mean(axis=0)


def aggregate_max(emb: np.ndarray) -> np.ndarray:
    """Max over slices (per dimension): (N, D) -> (D,)."""
    return emb.max(axis=0)


def aggregate_mean_std_concat(emb: np.ndarray) -> np.ndarray:
    """Concatenate mean and std along feature axis: (N, D) -> (2D,)."""
    m = emb.mean(axis=0)
    s = emb.std(axis=0)
    return np.concatenate([m, s], axis=0)


def aggregate_std_only(emb: np.ndarray) -> np.ndarray:
    """Per-feature std over slices (ddof=0), (N, D) -> (D,)."""
    return emb.std(axis=0, ddof=0)


def _cluster_hist_from_npz(path: str, k: int) -> Optional[np.ndarray]:
    """
    Build a normalized histogram over cluster_ids (length k) for one patient NPZ.
    Uses `mask` if present (expects same length as cluster_ids); otherwise counts all slices.
    Returns None if cluster_ids missing or invalid.
    """
    z = np.load(path, allow_pickle=True)
    if "cluster_ids" not in z.files:
        return None
    cid = np.asarray(z["cluster_ids"]).reshape(-1)
    if cid.ndim != 1 or cid.size == 0:
        return None
    if np.any(cid < 0):
        return None

    if "mask" in z.files:
        m = np.asarray(z["mask"]).reshape(-1)
        if m.shape[0] == cid.shape[0]:
            cid = cid[m.astype(bool)]

    if cid.size == 0:
        return np.zeros((k,), dtype=np.float64)

    cid = np.clip(cid.astype(np.int64), 0, k - 1)
    h = np.bincount(cid, minlength=k).astype(np.float64)
    s = float(h.sum())
    return h / s if s > 0 else h


def build_aligned_dataset_cluster_hist(
    npz_dir: str,
    clinical: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like build_aligned_dataset, but features are cluster histogram vectors (K,) per patient.
    K is inferred from the maximum cluster_id observed across patients in train∪val∪test.
    """
    clinical = clinical.set_index("patient_id", drop=False)
    split_ok = set(train_ids) | set(val_ids) | set(test_ids)

    # infer K from available NPZ files in split
    k_max = -1
    pid_to_path: dict[str, str] = {}
    for path in discover_npz_files(npz_dir):
        pid = os.path.basename(path).replace(".npz", "")
        if pid not in split_ok:
            continue
        if pid not in clinical.index:
            continue
        z = np.load(path, allow_pickle=True)
        if "cluster_ids" not in z.files:
            continue
        cid = np.asarray(z["cluster_ids"]).reshape(-1)
        if cid.size == 0:
            continue
        try:
            mx = int(np.max(cid))
        except Exception:
            continue
        if mx > k_max:
            k_max = mx
        pid_to_path[pid] = path

    k = int(k_max + 1)
    if k <= 0:
        # fall back to empty dataset; caller will skip due to low n_train / n_test
        return [], np.zeros((0, 1), dtype=np.float64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64), np.array([], dtype=str), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    want = sorted(pid_to_path.keys())
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    age_list: list[float] = []
    sex_list: list[str] = []
    for pid in want:
        row = clinical.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if pd.isna(row["age"]):
            continue
        h = _cluster_hist_from_npz(pid_to_path[pid], k=k)
        if h is None:
            continue
        X_list.append(h)
        y_list.append(int(row["label"]))
        age_list.append(float(row["age"]))
        sex_list.append(str(row["sex"]))

    if not X_list:
        return [], np.zeros((0, k), dtype=np.float64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64), np.array([], dtype=str), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    age = np.array(age_list, dtype=np.float64)
    sex = np.array(sex_list)
    tr_set, va_set, te_set = set(train_ids), set(val_ids), set(test_ids)
    train_m = np.array([pid in tr_set for pid in want], dtype=bool)
    val_m = np.array([pid in va_set for pid in want], dtype=bool)
    test_m = np.array([pid in te_set for pid in want], dtype=bool)
    return want, X, y, age, sex, train_m, val_m, test_m


POOLING_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": aggregate_mean,
    "max": aggregate_max,
    "mean_std": aggregate_mean_std_concat,
    "std": aggregate_std_only,
}


def get_aggregate_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in POOLING_REGISTRY:
        raise ValueError(f"pooling must be one of {list(POOLING_REGISTRY.keys())}")
    return POOLING_REGISTRY[name]


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------


def preprocess_features(
    X: np.ndarray,
    standardize: bool = True,
    l2_normalize: bool = False,
) -> tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Preprocess **patient-level** matrix X (n_patients, dim).

    Standardization (zero mean, unit variance per feature) puts features on a comparable
    scale so PCA/UMAP are not dominated by a few high-variance dimensions.

    Optional L2 row normalization makes each patient vector unit length; this emphasizes
    direction (cosine-like geometry) and can reduce scale confounds. Use with care: it
    removes magnitude information that might carry signal.

    For exploratory visualization (no held-out test), we fit on all patients. For
    publication-grade generalization claims, fit scaler only on training IDs.
    """
    scaler = None
    Xp = X.astype(np.float64, copy=True)
    if standardize:
        scaler = StandardScaler()
        Xp = scaler.fit_transform(Xp)
    if l2_normalize:
        norms = np.linalg.norm(Xp, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Xp = Xp / norms
    return Xp, scaler


# ---------------------------------------------------------------------------
# 4. PCA
# ---------------------------------------------------------------------------


def run_pca(X: np.ndarray, n_components: int = 2, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca


# ---------------------------------------------------------------------------
# 5. UMAP
# ---------------------------------------------------------------------------


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    if umap is None:
        raise RuntimeError(_umap_unavailable_message())
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def umap_fit_train_transform_query(
    X_train: np.ndarray,
    X_query: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, int]:
    """
    Fit UMAP on training rows only, then embed X_query (e.g. all patients).
    Returns (embedding_2d, n_neighbors_effective). Raises if umap-learn missing.
    """
    if umap is None:
        raise RuntimeError(_umap_unavailable_message())
    n_tr = int(X_train.shape[0])
    nn_eff = min(int(n_neighbors), max(1, n_tr - 1))
    reducer = umap.UMAP(
        n_neighbors=nn_eff,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    reducer.fit(X_train)
    return reducer.transform(X_query), nn_eff


# ---------------------------------------------------------------------------
# 6. Plotting (matplotlib only)
# ---------------------------------------------------------------------------


def _sex_to_marker(sex: str) -> str:
    s = str(sex).upper().strip()
    if s in ("M", "1", "MALE"):
        return "^"
    if s in ("F", "0", "FEMALE"):
        return "o"
    return "s"


def _sex_to_numeric(sex: str) -> float:
    s = str(sex).upper().strip()
    if s in ("M", "1", "MALE"):
        return 1.0
    if s in ("F", "0", "FEMALE"):
        return 0.0
    return 0.5


def plot_scatter_2d(
    Z: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    title: str,
    color_mode: Literal["label", "age", "sex", "logistic_prob"],
    out_path: str,
    xlabel: str = "dim-1",
    ylabel: str = "dim-2",
    prob_positive: Optional[np.ndarray] = None,
) -> None:
    """Scatter in 2D; color by label, age, sex, or logistic P(TTS); marker by sex when not sex-colored."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    if color_mode == "label":
        mask0 = labels == 0
        mask1 = labels == 1
        for mask, name, c in [(mask0, "Normal (0)", "tab:blue"), (mask1, "TTS (1)", "tab:red")]:
            if not np.any(mask):
                continue
            for sex in np.unique(sexes[mask]):
                sm = _sex_to_marker(str(sex))
                m = mask & (sexes == sex)
                if not np.any(m):
                    continue
                ax.scatter(
                    Z[m, 0],
                    Z[m, 1],
                    c=c,
                    marker=sm,
                    s=36,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.25,
                    label=f"{name} sex={sex}",
                )
        ax.legend(loc="best", fontsize=8, ncol=2)

    elif color_mode == "age":
        # Continuous age colormap; still vary marker by sex
        norm = mcolors.Normalize(vmin=np.nanmin(ages), vmax=np.nanmax(ages))
        cmap = plt.cm.viridis
        for sex in np.unique(sexes):
            m = sexes == sex
            if not np.any(m):
                continue
            sc = ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=ages[m],
                cmap=cmap,
                norm=norm,
                marker=_sex_to_marker(str(sex)),
                s=40,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.25,
                label=f"sex={sex}",
            )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("age")

    elif color_mode == "sex":
        sex_codes = np.array([_sex_to_numeric(s) for s in sexes], dtype=np.float64)
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=sex_codes,
            cmap="coolwarm",
            vmin=-0.1,
            vmax=1.1,
            s=40,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.25,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["F / 0", "M / 1"])
        cbar.set_label("sex (encoded)")

    elif color_mode == "logistic_prob":
        if prob_positive is None:
            raise ValueError("logistic_prob requires prob_positive")
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=prob_positive,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            s=40,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.25,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(TTS) logistic")

    else:
        raise ValueError(f"Unknown color_mode: {color_mode}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_side_by_side(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    name_a: str,
    name_b: str,
    method: str,
    color_mode: Literal["label", "age"],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def draw(ax, Z, panel_title: str):
        if color_mode == "label":
            for lab, cname, c in [(0, "Normal", "tab:blue"), (1, "TTS", "tab:red")]:
                m = labels == lab
                if not np.any(m):
                    continue
                ax.scatter(
                    Z[m, 0],
                    Z[m, 1],
                    c=c,
                    s=32,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.2,
                    label=cname,
                )
            ax.legend(fontsize=8)
        else:
            sc = ax.scatter(
                Z[:, 0],
                Z[:, 1],
                c=ages,
                cmap="viridis",
                s=36,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.2,
            )
            fig.colorbar(sc, ax=ax, fraction=0.046, label="age")
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(f"{method}-1")
    draw(axes[0], Z_a, name_a)
    draw(axes[1], Z_b, name_b)
    axes[0].set_ylabel(f"{method}-2")
    fig.suptitle(f"{method}: {color_mode}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional: cluster purity (dominant k-means id vs clinical label)
# ---------------------------------------------------------------------------


def report_cluster_purity(records: list[PatientRecord]) -> None:
    """If dominant_cluster is set, print label distribution per dominant cluster."""
    have = [r for r in records if r.dominant_cluster is not None]
    if not have:
        print("[cluster_purity] No cluster_ids in NPZ or length mismatch; skip.")
        return
    rows = []
    for r in have:
        rows.append({"dominant_cluster": r.dominant_cluster, "label": r.label})
    df = pd.DataFrame(rows)
    print("\n[cluster_purity] Contingency: dominant_cluster (from slices) vs label")
    ct = pd.crosstab(df["dominant_cluster"], df["label"], margins=True)
    print(ct.to_string())
    # Purity per cluster: fraction of majority class among patients with that dominant id
    print("\n[cluster_purity] Majority-label fraction per dominant_cluster (patient-level):")
    for k in sorted(df["dominant_cluster"].unique()):
        sub = df[df["dominant_cluster"] == k]["label"]
        vc = sub.value_counts()
        maj = vc.max() / len(sub) if len(sub) else 0.0
        print(f"  cluster {k}: n_patients={len(sub)}  majority_frac={maj:.3f}  counts:\n{vc.to_string()}")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def records_to_arrays(records: list[PatientRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.stack([r.patient_vec for r in records], axis=0)
    y = np.array([r.label for r in records], dtype=np.int64)
    age = np.array([r.age for r in records], dtype=np.float64)
    sex = np.array([r.sex for r in records])
    return X, y, age, sex


# ---------------------------------------------------------------------------
# 7. Train/val/test split classification + PCA colored by logistic prob
# ---------------------------------------------------------------------------


def load_split_patient_ids(split_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load patient_id lists from train.csv, val.csv, test.csv (same split as training)."""
    sd = os.path.abspath(split_dir)
    train = pd.read_csv(os.path.join(sd, "train.csv"))["patient_id"].astype(str).to_numpy()
    val = pd.read_csv(os.path.join(sd, "val.csv"))["patient_id"].astype(str).to_numpy()
    test = pd.read_csv(os.path.join(sd, "test.csv"))["patient_id"].astype(str).to_numpy()
    return train, val, test


def verify_disjoint_patient_splits(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    strict: bool = True,
) -> None:
    """
    Ensure no patient_id appears in more than one of train / val / test.

    If strict=True, raises ValueError on any overlap (true leakage for patient-level tasks).
    Duplicates *within* the same CSV are reported but do not fail (harmless for masking).
    """
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    o_tv = tr & va
    o_tt = tr & te
    o_vt = va & te
    if o_tv or o_tt or o_vt:
        msg = (
            f"[leakage] patient_id overlap: train∩val={len(o_tv)} train∩test={len(o_tt)} val∩test={len(o_vt)}. "
            f"Examples: {list(o_tv | o_tt | o_vt)[:8]}"
        )
        if strict:
            raise ValueError(msg)
        print("WARNING", msg, file=sys.stderr)
    else:
        print("[audit] OK: train / val / test patient_id sets are pairwise disjoint.")

    def _dup_report(name: str, arr: np.ndarray) -> None:
        u, c = np.unique(arr, return_counts=True)
        d = u[c > 1]
        if len(d):
            print(f"[audit] note: {name}.csv has {len(d)} duplicate patient_id row(s) (mask still valid).")

    _dup_report("train", train_ids)
    _dup_report("val", val_ids)
    _dup_report("test", test_ids)


def build_aligned_dataset(
    npz_dir: str,
    clinical: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Patient-level aggregation; keep patients in train∪val∪test with NPZ + clinical.

    Returns
    -------
    patient_ids, X, y, age, sex, mask_train, mask_val, mask_test (boolean masks aligned to rows)
    """
    rec, _skip = load_data(npz_dir, clinical, aggregate_fn)
    id_set = {r.patient_id for r in rec}
    split_ok = set(train_ids) | set(val_ids) | set(test_ids)
    want = sorted(id_set & split_ok)
    rec_map = {r.patient_id: r for r in rec}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    age_list: list[float] = []
    sex_list: list[str] = []
    for pid in want:
        r = rec_map[pid]
        X_list.append(r.patient_vec)
        y_list.append(int(r.label))
        age_list.append(float(r.age))
        sex_list.append(str(r.sex))
    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    age = np.array(age_list, dtype=np.float64)
    sex = np.array(sex_list)
    tr_set, va_set, te_set = set(train_ids), set(val_ids), set(test_ids)
    train_m = np.array([pid in tr_set for pid in want], dtype=bool)
    val_m = np.array([pid in va_set for pid in want], dtype=bool)
    test_m = np.array([pid in te_set for pid in want], dtype=bool)
    return want, X, y, age, sex, train_m, val_m, test_m


def build_aligned_dataset_mean_std(
    npz_dir: str,
    clinical: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_aligned_dataset(
        npz_dir, clinical, train_ids, val_ids, test_ids, aggregate_mean_std_concat
    )


def _scores_for_auc(model, X: np.ndarray) -> np.ndarray:
    """Higher score = more predicted toward positive class (TTS = 1)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise TypeError(f"No scores from {type(model)}")


def fit_three_classifiers(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, object]:
    """Logistic regression, linear SVM, RBF SVM — fit on scaled training features."""
    models: dict[str, object] = {}
    models["logistic"] = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    models["logistic"].fit(X_train, y_train)

    models["linear_svc"] = LinearSVC(
        max_iter=20000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        dual=False,
    )
    models["linear_svc"].fit(X_train, y_train)

    models["rbf_svc"] = SVC(
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    models["rbf_svc"].fit(X_train, y_train)
    return models


def bootstrap_roc_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap resamples of test set; return (roc_percentiles[3], pr_percentiles[3])."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    roc_vals: list[float] = []
    pr_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        sc = scores[idx]
        if np.unique(yt).size < 2:
            continue
        roc_vals.append(roc_auc_score(yt, sc))
        pr_vals.append(average_precision_score(yt, sc))
    ra = np.asarray(roc_vals, dtype=np.float64)
    pa = np.asarray(pr_vals, dtype=np.float64)
    if ra.size == 0:
        q = np.array([np.nan, np.nan, np.nan])
        return q, q
    return np.percentile(ra, [2.5, 50.0, 97.5]), np.percentile(pa, [2.5, 50.0, 97.5])


def label_shuffle_null_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_shuffles: int,
    seed: int = RANDOM_STATE,
) -> tuple[float, np.ndarray, float]:
    """
    Permutation / label-shuffle null for logistic regression: shuffle train labels only,
    refit, measure test ROC. Real labels on test are unchanged.

    Returns true test ROC (correct labels on train), array of null test ROCs, one-sided p-value
    P(null_ROC >= true_ROC) under the shuffle null (low p => true signal unlikely by chance).
    """
    true_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    true_model.fit(X_train, y_train)
    true_scores = true_model.predict_proba(X_test)[:, 1]
    if np.unique(y_test).size < 2:
        return float("nan"), np.array([]), float("nan")
    true_roc = float(roc_auc_score(y_test, true_scores))

    rng = np.random.RandomState(seed)
    null_rocs: list[float] = []
    for _ in range(n_shuffles):
        idx = rng.permutation(len(y_train))
        yt_perm = y_train[idx]
        m = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        m.fit(X_train, yt_perm)
        st = m.predict_proba(X_test)[:, 1]
        if np.unique(y_test).size >= 2:
            null_rocs.append(roc_auc_score(y_test, st))
    null_arr = np.asarray(null_rocs, dtype=np.float64)
    p_one_sided = float(np.mean(null_arr >= true_roc)) if null_arr.size else float("nan")
    return true_roc, null_arr, p_one_sided


def plot_logistic_score_histograms_splits(
    y: np.ndarray,
    scores_all: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    test_m: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """Overlaid histograms of P(TTS) for class 0 vs 1 on train / val / test."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (sn, mask) in zip(
        axes,
        [("train", train_m), ("val", val_m), ("test", test_m)],
    ):
        if not np.any(mask):
            ax.set_title(f"{sn} (empty)")
            continue
        ys = y[mask]
        sc = scores_all[mask]
        for lab, c, lbl in [(0, "tab:blue", "y=0 Normal"), (1, "tab:red", "y=1 TTS")]:
            m = ys == lab
            if np.any(m):
                ax.hist(sc[m], bins=14, alpha=0.55, color=c, label=lbl, density=True)
        ax.set_xlabel("predicted P(TTS)")
        ax.set_ylabel("density")
        ax.set_title(sn)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def misclassified_rows_for_split(
    split_name: str,
    patient_ids: list[str],
    mask: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    out: list[dict] = []
    for i in np.where(mask)[0]:
        yt = int(y[i])
        sc = float(scores[i])
        yp = 1 if sc >= threshold else 0
        if yp != yt:
            out.append(
                {
                    "split": split_name,
                    "patient_id": patient_ids[i],
                    "y_true": yt,
                    "predicted_probability_tts": sc,
                    "y_pred": yp,
                }
            )
    return out


def save_logistic_mean_std_coef_csv(
    logistic_model: LogisticRegression,
    out_csv: str,
) -> tuple[int, np.ndarray]:
    """
    For mean||std concatenated features (2D dims), coef order is [mean_0..D-1, std_0..D-1]
    after StandardScaler — coefficients are w.r.t. **scaled** inputs (per 1 SD of feature).

    Saves all dims sorted by |coef on std block (descending). Returns (D, latent_dim order).
    """
    coef = logistic_model.coef_.ravel()
    if coef.size % 2 != 0:
        raise ValueError(f"Expected even n_features for mean||std layout, got {coef.size}")
    d = coef.size // 2
    c_mean, c_std = coef[:d], coef[d:]
    rows = []
    for j in range(d):
        rows.append(
            {
                "latent_dim": j,
                "coef_mean_block_scaled_feature": float(c_mean[j]),
                "coef_std_block_scaled_feature": float(c_std[j]),
                "abs_coef_std_block": float(abs(c_std[j])),
            }
        )
    df = pd.DataFrame(rows).sort_values("abs_coef_std_block", ascending=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    rank = df["latent_dim"].to_numpy()
    return d, rank


def plot_logistic_std_block_coef_visualizations(
    logistic_model: LogisticRegression,
    diag_dir: str,
    name: str,
    bar_top_k: int,
) -> None:
    """
    Bar chart: top bar_top_k latent dims by |coef| on the std block.
    Heatmap: all D dimensions as a single row (magnitude of std-block coefficients).

    Coefs apply to **StandardScaler-transformed** σ (train-fitted), not raw slice-std.
    """
    coef = logistic_model.coef_.ravel()
    if coef.size % 2 != 0:
        raise ValueError(f"Expected mean‖std layout (even n_features), got {coef.size}")
    d = coef.size // 2
    c_std = coef[d:]
    abs_std = np.abs(c_std)
    k_bar = max(1, min(int(bar_top_k), d))

    top_j = np.argsort(abs_std)[-k_bar:][::-1]
    vals = abs_std[top_j]

    fig, ax = plt.subplots(figsize=(min(18, 5.0 + k_bar * 0.22), 5.0))
    xpos = np.arange(k_bar)
    ax.bar(xpos, vals, color="steelblue", edgecolor="black", linewidth=0.35)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(int(j)) for j in top_j], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(r"|coef| on scaled $\sigma_j$ (std block)")
    ax.set_xlabel("latent_dim j")
    ax.set_title(
        f"{name} — logistic: top-{k_bar} std-block |coef|\n"
        "(mean‖std features; coef = change in log-odds per 1 SD of scaled σ_j)"
    )
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "logistic_std_coef_topk_bar.png"), dpi=150)
    plt.close(fig)

    fig_w = max(12.0, min(28.0, d / 35.0))
    fig2, ax2 = plt.subplots(figsize=(fig_w, 1.35))
    im = ax2.imshow(
        abs_std.reshape(1, -1),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
    )
    ax2.set_yticks([])
    tick_step = max(1, d // 40)
    ax2.set_xticks(np.arange(0, d, tick_step))
    ax2.set_xlabel("latent_dim j (std block)")
    ax2.set_title(f"{name} — |coef| on scaled σ: all {d} dimensions")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
    cbar.set_label("|coef|")
    fig2.tight_layout()
    fig2.savefig(os.path.join(diag_dir, "logistic_std_coef_full_heatmap.png"), dpi=150)
    plt.close(fig2)


def run_std_latent_diagnostics(
    name: str,
    npz_dir: str,
    clinical: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    logistic_model: LogisticRegression,
    y: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    tr_m: np.ndarray,
    va_m: np.ndarray,
    te_m: np.ndarray,
    diag_dir: str,
    topk: int,
    umap_neighbors: int = 10,
    umap_min_dist: float = 0.1,
) -> None:
    """
    1) CSV + bar/heatmap: std-block |coef| (top-k bar + full 1×D heatmap).
    2) PCA on **std-only** patient vectors (scaler+PCA fit on train).
    3) Optional UMAP (same std features, train-fit) if umap-learn is installed.
    4) Boxplots: Normal vs TTS for ||std||_2 and L1 mass on top-k std dims (raw σ, not z-scored).
    """
    os.makedirs(diag_dir, exist_ok=True)

    d, rank_by_std_coef = save_logistic_mean_std_coef_csv(
        logistic_model,
        os.path.join(diag_dir, "logistic_coef_importance_mean_vs_std_blocks.csv"),
    )
    k_use = max(1, min(int(topk), d))
    top_idx = rank_by_std_coef[:k_use]

    plot_logistic_std_block_coef_visualizations(
        logistic_model,
        diag_dir,
        name,
        bar_top_k=k_use,
    )

    _, X_std, _, _, _, _, _, _ = build_aligned_dataset(
        npz_dir, clinical, train_ids, val_ids, test_ids, aggregate_std_only
    )
    if X_std.shape[1] != d:
        raise RuntimeError(
            f"std feature dim {X_std.shape[1]} != half of mean+std ({d}); check aggregation."
        )

    sc_s = StandardScaler()
    Xtr_s = sc_s.fit_transform(X_std[tr_m])
    pca_s = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_s.fit(Xtr_s)
    Zs = pca_s.transform(sc_s.transform(X_std))
    evs = pca_s.explained_variance_ratio_
    plot_scatter_2d(
        Zs,
        y,
        age,
        sex,
        title=f"{name} — PCA std-only (slice-σ per dim); train-fit scaler+PCA\n"
        f"expl_var={evs[0]:.3f}, {evs[1]:.3f}",
        color_mode="label",
        out_path=os.path.join(diag_dir, "pca_std_only_label.png"),
        xlabel="PC1",
        ylabel="PC2",
    )
    plot_scatter_2d(
        Zs,
        y,
        age,
        sex,
        title=f"{name} — PCA std-only by age",
        color_mode="age",
        out_path=os.path.join(diag_dir, "pca_std_only_age.png"),
        xlabel="PC1",
        ylabel="PC2",
    )

    if umap is not None:
        X_std_scaled_all = sc_s.transform(X_std)
        try:
            Zsu, nn_u = umap_fit_train_transform_query(
                Xtr_s,
                X_std_scaled_all,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
            )
            umap_note = f"n_neighbors={nn_u}, min_dist={umap_min_dist}; fit on train, transform all"
            plot_scatter_2d(
                Zsu,
                y,
                age,
                sex,
                title=f"{name} — UMAP std-only\n{umap_note}",
                color_mode="label",
                out_path=os.path.join(diag_dir, "umap_std_only_label.png"),
                xlabel="UMAP-1",
                ylabel="UMAP-2",
            )
            plot_scatter_2d(
                Zsu,
                y,
                age,
                sex,
                title=f"{name} — UMAP std-only by age",
                color_mode="age",
                out_path=os.path.join(diag_dir, "umap_std_only_age.png"),
                xlabel="UMAP-1",
                ylabel="UMAP-2",
            )
        except Exception as exc:
            print(f"  [std_diag umap] skipped: {exc}")

    l2 = np.linalg.norm(X_std, axis=1)
    topk_l1 = np.sum(np.abs(X_std[:, top_idx]), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].boxplot([l2[y == 0], l2[y == 1]])
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Normal (0)", "TTS (1)"])
    axes[0].set_ylabel(r"$\|\sigma\|_2$ (raw slice-std vector)")
    axes[0].set_title("L2 norm of per-patient std vector")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([topk_l1[y == 0], topk_l1[y == 1]])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Normal (0)", "TTS (1)"])
    axes[1].set_ylabel(f"$\\sum_j |\\sigma_j|$ over top-{k_use} dims (by |logistic coef| on std)")
    axes[1].set_title("L1 mass on most important std dimensions")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"{name} — inter-slice variability: Normal vs TTS (raw σ, not StandardScaler)"
    )
    fig.tight_layout()
    bp_path = os.path.join(diag_dir, "boxplot_std_magnitude_normal_vs_tts.png")
    fig.savefig(bp_path, dpi=150)
    plt.close(fig)

    print(
        f"  [std_diag] top-{k_use} latent dims by |coef_std|: {top_idx[: min(12, k_use)].tolist()} …\n"
        f"           plots: logistic_std_coef_topk_bar.png, logistic_std_coef_full_heatmap.png"
    )


def _merge_root_metrics_csv(
    path: str,
    df_new: pd.DataFrame,
    replace_latent_sources: set[str],
) -> pd.DataFrame:
    """
    If path exists, drop rows whose latent_source is in replace_latent_sources, then append df_new.
    Used so a monai-only re-run can append to an existing classifier_metrics / diagnostics table.
    """
    if not os.path.exists(path):
        return df_new
    if "latent_source" not in df_new.columns:
        return df_new
    try:
        df_old = pd.read_csv(path)
    except Exception as exc:
        print(f"  [merge] could not read {path}: {exc}; overwriting")
        return df_new
    if "latent_source" not in df_old.columns:
        print(f"  [merge] no latent_source in existing {path}; overwriting")
        return df_new
    cols_old, cols_new = set(df_old.columns), set(df_new.columns)
    if cols_old != cols_new:
        print(
            f"  [merge] warning: column mismatch old={sorted(cols_old - cols_new)} "
            f"new={sorted(cols_new - cols_old)}"
        )
    kept = df_old[~df_old["latent_source"].isin(replace_latent_sources)]
    out = pd.concat([kept, df_new], ignore_index=True)
    print(f"  [merge] {path}: kept {len(kept)} rows, added {len(df_new)} rows (replaced sources={replace_latent_sources})")
    return out


def run_latent_classifier_eval(
    latent_sources: list[tuple[str, str]],
    split_dir: str,
    clinical: pd.DataFrame,
    out_dir: str,
    bootstrap_n: int,
    eval_diagnostics: bool = False,
    label_shuffle_n: int = 200,
    std_diag_topk: int = 16,
    umap_neighbors: int = 10,
    umap_min_dist: float = 0.1,
    merge_existing_root_metrics: bool = False,
) -> None:
    """
    For each latent NPZ directory: mean/std pooling, same train/val/test split,
    StandardScaler fit on train, three classifiers, ROC/PR on val & test + bootstrap on test.
    PCA fit on train only, transform all; plots colored by label, age, sex, logistic P(TTS).
    If umap-learn is installed, same for UMAP (train-fit, transform all) with --umap_* args.

    If eval_diagnostics: mean vs std-only vs mean+std ablation; label-shuffle null (logistic);
    score histograms (logistic); misclassified patient CSVs (logistic, 0.5 threshold);
    std-block logistic importance, std-only PCA/UMAP, std magnitude boxplots (uses std_diag_topk).
    """
    train_ids, val_ids, test_ids = load_split_patient_ids(split_dir)
    verify_disjoint_patient_splits(train_ids, val_ids, test_ids, strict=True)
    os.makedirs(out_dir, exist_ok=True)
    summary_rows: list[dict] = []
    pooling_ablation_rows: list[dict] = []
    shuffle_rows: list[dict] = []
    source_names_this_run = {str(n) for n, _ in latent_sources}

    for name, npz_dir in latent_sources:
        npz_dir = os.path.abspath(npz_dir)
        print(f"\n{'=' * 60}\n[eval] source={name}  npz_dir={npz_dir}\n{'=' * 60}")

        pids, X, y, age, sex, tr_m, va_m, te_m = build_aligned_dataset_mean_std(
            npz_dir, clinical, train_ids, val_ids, test_ids
        )
        n_tr, n_va, n_te = int(tr_m.sum()), int(va_m.sum()), int(te_m.sum())
        print(f"  patients aligned to split: total={len(pids)}  train={n_tr}  val={n_va}  test={n_te}")
        if n_tr < 2 or n_te < 2:
            print("  [skip] need at least 2 train and 2 test patients.")
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[tr_m])
        X_val = scaler.transform(X[va_m]) if n_va else None
        X_test = scaler.transform(X[te_m])
        y_train, y_val, y_test = y[tr_m], y[va_m], y[te_m]

        models = fit_three_classifiers(X_train, y_train)

        # --- metrics ---
        for mname, model in models.items():
            row_base = {"latent_source": name, "model": mname, "n_train": n_tr, "n_val": n_va, "n_test": n_te}
            if n_va:
                sv = _scores_for_auc(model, X_val)
                row_base["val_roc_auc"] = float(roc_auc_score(y_val, sv))
                row_base["val_pr_auc"] = float(average_precision_score(y_val, sv))
            else:
                row_base["val_roc_auc"] = np.nan
                row_base["val_pr_auc"] = np.nan

            st = _scores_for_auc(model, X_test)
            row_base["test_roc_auc"] = float(roc_auc_score(y_test, st))
            row_base["test_pr_auc"] = float(average_precision_score(y_test, st))

            r_q, p_q = bootstrap_roc_pr(y_test, st, n_boot=bootstrap_n, seed=RANDOM_STATE)
            row_base["test_roc_boot_q025"] = r_q[0]
            row_base["test_roc_boot_q50"] = r_q[1]
            row_base["test_roc_boot_q975"] = r_q[2]
            row_base["test_pr_boot_q025"] = p_q[0]
            row_base["test_pr_boot_q50"] = p_q[1]
            row_base["test_pr_boot_q975"] = p_q[2]

            summary_rows.append(row_base)
            print(
                f"  {mname}: test ROC={row_base['test_roc_auc']:.4f} PR={row_base['test_pr_auc']:.4f}  "
                f"boot ROC median={r_q[1]:.4f} PR median={p_q[1]:.4f}"
            )

        # --- PCA visualization (fit on train, transform all) ---
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        pca2.fit(X_train)
        X_all_scaled = scaler.transform(X)
        Z = pca2.transform(X_all_scaled)
        ev = pca2.explained_variance_ratio_
        log_model = models["logistic"]
        prob_all = log_model.predict_proba(X_all_scaled)[:, 1]

        sub = os.path.join(out_dir, name.replace(" ", "_").replace("/", "_"))
        os.makedirs(sub, exist_ok=True)

        plot_scatter_2d(
            Z, y, age, sex,
            title=f"{name} — PCA (mean+std pool); scaler+PCA fit on train only\n"
            f"expl_var={ev[0]:.3f}, {ev[1]:.3f}",
            color_mode="label",
            out_path=os.path.join(sub, "pca_label.png"),
            xlabel="PC1",
            ylabel="PC2",
        )
        plot_scatter_2d(
            Z, y, age, sex,
            title=f"{name} — PCA by age",
            color_mode="age",
            out_path=os.path.join(sub, "pca_age.png"),
            xlabel="PC1",
            ylabel="PC2",
        )
        plot_scatter_2d(
            Z, y, age, sex,
            title=f"{name} — PCA by sex",
            color_mode="sex",
            out_path=os.path.join(sub, "pca_sex.png"),
            xlabel="PC1",
            ylabel="PC2",
        )
        plot_scatter_2d(
            Z, y, age, sex,
            title=f"{name} — PCA by logistic P(TTS) (train-fit model, all points)",
            color_mode="logistic_prob",
            out_path=os.path.join(sub, "pca_logistic_prob.png"),
            xlabel="PC1",
            ylabel="PC2",
            prob_positive=prob_all,
        )

        if umap is not None:
            try:
                Zu, nn_u = umap_fit_train_transform_query(
                    X_train,
                    X_all_scaled,
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                )
                umap_note = f"n_neighbors={nn_u}, min_dist={umap_min_dist}; fit on train, transform all"
                plot_scatter_2d(
                    Zu,
                    y,
                    age,
                    sex,
                    title=f"{name} — UMAP (mean+std pool); same scaler as PCA\n{umap_note}",
                    color_mode="label",
                    out_path=os.path.join(sub, "umap_label.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
                plot_scatter_2d(
                    Zu,
                    y,
                    age,
                    sex,
                    title=f"{name} — UMAP by age",
                    color_mode="age",
                    out_path=os.path.join(sub, "umap_age.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
                plot_scatter_2d(
                    Zu,
                    y,
                    age,
                    sex,
                    title=f"{name} — UMAP by sex",
                    color_mode="sex",
                    out_path=os.path.join(sub, "umap_sex.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
                plot_scatter_2d(
                    Zu,
                    y,
                    age,
                    sex,
                    title=f"{name} — UMAP by logistic P(TTS) (train-fit model, all points)",
                    color_mode="logistic_prob",
                    out_path=os.path.join(sub, "umap_logistic_prob.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                    prob_positive=prob_all,
                )
            except Exception as exc:
                print(f"  [umap] skipped: {exc}")
        else:
            print(f"  [umap] {_umap_unavailable_message()}")

        if eval_diagnostics:
            diag = os.path.join(sub, "diagnostics")
            os.makedirs(diag, exist_ok=True)

            # 1) mean-only vs std-only vs mean+std (same scaler: train fit only)
            for ptag, agg_fn in [
                ("mean", aggregate_mean),
                ("std", aggregate_std_only),
                ("mean_std", aggregate_mean_std_concat),
            ]:
                _, Xp, _, _, _, _, _, _ = build_aligned_dataset(
                    npz_dir, clinical, train_ids, val_ids, test_ids, agg_fn
                )
                sc_p = StandardScaler()
                Xtr_p = sc_p.fit_transform(Xp[tr_m])
                Xte_p = sc_p.transform(Xp[te_m])
                ytr, yte = y[tr_m], y[te_m]
                ms_p = fit_three_classifiers(Xtr_p, ytr)
                for mn, mdl in ms_p.items():
                    st = _scores_for_auc(mdl, Xte_p)
                    pooling_ablation_rows.append(
                        {
                            "latent_source": name,
                            "pooling": ptag,
                            "model": mn,
                            "test_roc_auc": float(roc_auc_score(yte, st)),
                            "test_pr_auc": float(average_precision_score(yte, st)),
                        }
                    )

            # 1b) cluster histogram (uses cluster_ids; per-patient feature dim = K inferred from NPZs)
            # If a source doesn't have cluster_ids (or cannot infer K), it will be skipped implicitly
            # by the low-n check.
            _pids_c, Xc, yc, _age_c, _sex_c, trc, _vac, tec = build_aligned_dataset_cluster_hist(
                npz_dir, clinical, train_ids, val_ids, test_ids
            )
            if Xc.shape[0] and int(trc.sum()) >= 2 and int(tec.sum()) >= 2 and np.unique(yc[tec]).size >= 2:
                sc_c = StandardScaler()
                Xtr_c = sc_c.fit_transform(Xc[trc])
                Xte_c = sc_c.transform(Xc[tec])
                ytr_c, yte_c = yc[trc], yc[tec]
                ms_c = fit_three_classifiers(Xtr_c, ytr_c)
                for mn, mdl in ms_c.items():
                    st = _scores_for_auc(mdl, Xte_c)
                    pooling_ablation_rows.append(
                        {
                            "latent_source": name,
                            "pooling": "cluster_hist",
                            "model": mn,
                            "test_roc_auc": float(roc_auc_score(yte_c, st)),
                            "test_pr_auc": float(average_precision_score(yte_c, st)),
                        }
                    )

            # 2) Label shuffle null (logistic; features = mean+std pipeline)
            if label_shuffle_n > 0 and np.unique(y_test).size >= 2:
                tro, nullr, pv = label_shuffle_null_logistic(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_shuffles=label_shuffle_n,
                    seed=RANDOM_STATE,
                )
                shuffle_rows.append(
                    {
                        "latent_source": name,
                        "true_test_roc_logistic": tro,
                        "null_roc_mean": float(np.mean(nullr)) if nullr.size else np.nan,
                        "null_roc_std": float(np.std(nullr)) if nullr.size else np.nan,
                        "p_value_one_sided": pv,
                        "n_shuffles": label_shuffle_n,
                    }
                )
                print(
                    f"  [shuffle] logistic test ROC={tro:.4f}  null mean={np.mean(nullr):.4f}  "
                    f"p(one-sided)={pv:.4f}"
                )

            # 3) Prediction score histograms (logistic, train/val/test)
            plot_logistic_score_histograms_splits(
                y,
                prob_all,
                tr_m,
                va_m,
                te_m,
                title=f"{name} — logistic P(TTS) (mean+std pool)",
                out_path=os.path.join(diag, "logistic_score_histograms.png"),
            )

            # 4) Misclassified patients (logistic, threshold 0.5)
            bad: list[dict] = []
            bad.extend(misclassified_rows_for_split("train", pids, tr_m, y, prob_all))
            bad.extend(misclassified_rows_for_split("val", pids, va_m, y, prob_all))
            bad.extend(misclassified_rows_for_split("test", pids, te_m, y, prob_all))
            pd.DataFrame(bad).to_csv(
                os.path.join(diag, "misclassified_logistic.csv"), index=False
            )
            print(f"  [misclass] logistic threshold=0.5: {len(bad)} errors (train+val+test)")

            # 5) Std-focused analysis: coef on std block, std-only PCA, ||std|| boxplots
            run_std_latent_diagnostics(
                name,
                npz_dir,
                clinical,
                train_ids,
                val_ids,
                test_ids,
                log_model,
                y,
                age,
                sex,
                tr_m,
                va_m,
                te_m,
                diag,
                topk=std_diag_topk,
                umap_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
            )

    if summary_rows:
        df_out = pd.DataFrame(summary_rows)
        csv_path = os.path.join(out_dir, "classifier_metrics_mean_std_pooling.csv")
        if merge_existing_root_metrics:
            df_out = _merge_root_metrics_csv(csv_path, df_out, source_names_this_run)
        df_out.to_csv(csv_path, index=False)
        print(f"\n[eval] Wrote {csv_path}")

    if pooling_ablation_rows:
        ppath = os.path.join(out_dir, "diagnostics_pooling_ablation.csv")
        df_p = pd.DataFrame(pooling_ablation_rows)
        if merge_existing_root_metrics:
            df_p = _merge_root_metrics_csv(ppath, df_p, source_names_this_run)
        df_p.to_csv(ppath, index=False)
        print(f"[eval] Wrote {ppath}")
    if shuffle_rows:
        spath = os.path.join(out_dir, "diagnostics_label_shuffle_logistic.csv")
        df_s = pd.DataFrame(shuffle_rows)
        if merge_existing_root_metrics:
            df_s = _merge_root_metrics_csv(spath, df_s, source_names_this_run)
        df_s.to_csv(spath, index=False)
        print(f"[eval] Wrote {spath}")


def run_full_pipeline(
    name: str,
    npz_dir: str,
    clinical: pd.DataFrame,
    out_dir: str,
    pooling: str,
    standardize: bool,
    l2_normalize: bool,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_sweep: bool,
    run_cluster_purity: bool,
) -> None:
    agg = get_aggregate_fn(pooling)
    records, skipped = load_data(npz_dir, clinical, agg)
    print(f"\n[{name}] loaded patients: {len(records)}  skipped: {len(skipped)}")
    if skipped[:5]:
        print("  sample skips:", skipped[:5])
    if len(records) < 5:
        print("Too few patients; abort.", file=sys.stderr)
        sys.exit(1)

    if run_cluster_purity:
        report_cluster_purity(records)

    X, y, age, sex = records_to_arrays(records)
    X_proc, _ = preprocess_features(X, standardize=standardize, l2_normalize=l2_normalize)

    # PCA
    Z_pca, pca = run_pca(X_proc, n_components=2)
    ev = pca.explained_variance_ratio_
    print(f"\n[{name}] PCA explained variance ratio (2D): {ev}  cumulative={ev.sum():.4f}")

    sub = os.path.join(out_dir, name.replace(" ", "_"))
    os.makedirs(sub, exist_ok=True)

    plot_scatter_2d(
        Z_pca, y, age, sex,
        title=f"{name} — PCA (pooling={pooling})\nexpl_var={ev[0]:.3f}, {ev[1]:.3f}",
        color_mode="label",
        out_path=os.path.join(sub, "pca_label.png"),
        xlabel="PC1",
        ylabel="PC2",
    )
    plot_scatter_2d(
        Z_pca, y, age, sex,
        title=f"{name} — PCA colored by age",
        color_mode="age",
        out_path=os.path.join(sub, "pca_age.png"),
        xlabel="PC1",
        ylabel="PC2",
    )

    if umap is None:
        print(_umap_unavailable_message())
        return

    def umap_one(nn: int, md: float, tag: str) -> np.ndarray:
        print(f"  UMAP n_neighbors={nn} min_dist={md}")
        return run_umap(X_proc, n_neighbors=nn, min_dist=md, metric="cosine", random_state=RANDOM_STATE)

    if umap_sweep:
        for nn in (5, 10, 20):
            for md in (0.1, 0.3, 0.5):
                tag = f"nn{nn}_md{md}"
                Zu = umap_one(nn, md, tag)
                plot_scatter_2d(
                    Zu, y, age, sex,
                    title=f"{name} — UMAP ({tag})",
                    color_mode="label",
                    out_path=os.path.join(sub, f"umap_label_{tag}.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
                plot_scatter_2d(
                    Zu, y, age, sex,
                    title=f"{name} — UMAP ({tag}) age",
                    color_mode="age",
                    out_path=os.path.join(sub, f"umap_age_{tag}.png"),
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
    else:
        Zu = umap_one(umap_neighbors, umap_min_dist, "single")
        plot_scatter_2d(
            Zu, y, age, sex,
            title=f"{name} — UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})",
            color_mode="label",
            out_path=os.path.join(sub, "umap_label.png"),
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )
        plot_scatter_2d(
            Zu, y, age, sex,
            title=f"{name} — UMAP age",
            color_mode="age",
            out_path=os.path.join(sub, "umap_age.png"),
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )


def run_compare(
    npz_dirs: list[str],
    names: list[str],
    clinical: pd.DataFrame,
    out_dir: str,
    pooling: str,
    standardize: bool,
    l2_normalize: bool,
) -> None:
    agg = get_aggregate_fn(pooling)
    Z_list = []
    common_ids: Optional[set[str]] = None
    rec_per_source: list[list[PatientRecord]] = []

    for d, name in zip(npz_dirs, names):
        rec, _ = load_data(d, clinical, agg)
        ids = {r.patient_id for r in rec}
        common_ids = ids if common_ids is None else common_ids & ids
        rec_per_source.append(rec)

    assert common_ids is not None
    print(f"\n[compare] patients common to all sources: {len(common_ids)}")

    Z_pca_list = []
    Z_umap_list = []
    y = age = sex = None

    for rec in rec_per_source:
        rec_f = [r for r in rec if r.patient_id in common_ids]
        rec_f = sorted(rec_f, key=lambda r: r.patient_id)
        X, y_arr, age_arr, sex_arr = records_to_arrays(rec_f)
        if y is None:
            y, age, sex = y_arr, age_arr, sex_arr
        else:
            if not np.array_equal(y, y_arr):
                print("[warn] label mismatch for same patient order; using first source labels")

        X_proc, _ = preprocess_features(X, standardize=standardize, l2_normalize=l2_normalize)
        Z_pca, _ = run_pca(X_proc, n_components=2)
        Z_pca_list.append(Z_pca)
        if umap is not None:
            Z_umap_list.append(run_umap(X_proc, n_neighbors=10, min_dist=0.1, metric="cosine"))

    cmp_dir = os.path.join(out_dir, "compare")
    os.makedirs(cmp_dir, exist_ok=True)
    plot_compare_side_by_side(
        Z_pca_list[0], Z_pca_list[1], y, age, sex, names[0], names[1], "PCA", "label",
        os.path.join(cmp_dir, "pca_compare_label.png"),
    )
    plot_compare_side_by_side(
        Z_pca_list[0], Z_pca_list[1], y, age, sex, names[0], names[1], "PCA", "age",
        os.path.join(cmp_dir, "pca_compare_age.png"),
    )
    if len(Z_umap_list) == 2:
        plot_compare_side_by_side(
            Z_umap_list[0], Z_umap_list[1], y, age, sex, names[0], names[1], "UMAP", "label",
            os.path.join(cmp_dir, "umap_compare_label.png"),
        )
        plot_compare_side_by_side(
            Z_umap_list[0], Z_umap_list[1], y, age, sex, names[0], names[1], "UMAP", "age",
            os.path.join(cmp_dir, "umap_compare_age.png"),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Patient-level PCA/UMAP for slice NPZ embeddings")
    ap.add_argument(
        "--eval_classifiers",
        action="store_true",
        help="Mean+std patient pooling; LogReg + LinearSVM + RBF-SVM; ROC/PR + bootstrap; PCA & UMAP plots (if umap-learn)",
    )
    ap.add_argument(
        "--split_dir",
        default=None,
        help="Folder with train.csv, val.csv, test.csv (patient_id) — required with --eval_classifiers",
    )
    ap.add_argument(
        "--latent_source",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ_DIR"),
        default=None,
        help="Repeat: display_name and directory of per-patient .npz (embeddings). Used with --eval_classifiers.",
    )
    ap.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap resamples for test ROC/PR (eval mode)")
    ap.add_argument(
        "--eval_diagnostics",
        action="store_true",
        help="With --eval_classifiers: pooling ablation (mean/std/mean_std), label-shuffle null, "
        "score histograms, misclassified CSV",
    )
    ap.add_argument(
        "--label_shuffle_n",
        type=int,
        default=200,
        help="Permutation repeats for label-shuffle test (0 to skip)",
    )
    ap.add_argument(
        "--merge_existing_root_metrics",
        action="store_true",
        help="With --eval_classifiers: if out_dir already has classifier_metrics_mean_std_pooling.csv "
        "(and diagnostic CSVs), keep rows for other latent_source values and replace/append only "
        "rows for latent_source names used in this run.",
    )
    ap.add_argument(
        "--std_diag_topk",
        type=int,
        default=16,
        help="With --eval_diagnostics: top-|coef| std dims for boxplot L1 mass (and rank head print)",
    )
    ap.add_argument("--npz_dir", default=None, help="Directory containing one .npz per patient")
    ap.add_argument(
        "--compare_npz_dirs",
        nargs="+",
        default=None,
        help="Two directories to compare (side-by-side); runs compare mode only",
    )
    ap.add_argument(
        "--compare_names",
        nargs="+",
        default=None,
        help="Names for compare mode (same length as --compare_npz_dirs)",
    )
    ap.add_argument(
        "--labels_csv",
        action="append",
        default=None,
        metavar="PATH",
        help="Clinical CSV with patient_id, label or case, age, sex (repeat to merge)",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--run_name", default="run", help="Subfolder name for single-source run")
    ap.add_argument(
        "--pooling",
        default="mean",
        choices=list(POOLING_REGISTRY.keys()),
        help="Patient-level aggregation of slice embeddings",
    )
    ap.add_argument("--no_standardize", action="store_true", help="Disable StandardScaler")
    ap.add_argument("--l2_normalize", action="store_true", help="L2-normalize each patient vector after scaling")
    ap.add_argument("--umap_neighbors", type=int, default=10)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument(
        "--umap_sweep",
        action="store_true",
        help="Run UMAP for n_neighbors in {5,10,20} and min_dist in {0.1,0.3,0.5}",
    )
    ap.add_argument("--cluster_purity", action="store_true", help="Report dominant slice cluster vs label")
    args = ap.parse_args()

    if not args.labels_csv:
        print("Provide at least one --labels_csv", file=sys.stderr)
        sys.exit(1)

    np.random.seed(RANDOM_STATE)
    clinical = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
    print(f"Clinical table: {len(clinical)} unique patient_id rows")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.eval_classifiers:
        if not args.split_dir:
            print("--eval_classifiers requires --split_dir", file=sys.stderr)
            sys.exit(1)
        if not args.latent_source:
            print("--eval_classifiers requires at least one --latent_source NAME DIR", file=sys.stderr)
            sys.exit(1)
        sources = [(str(a[0]), os.path.abspath(str(a[1]))) for a in args.latent_source]
        run_latent_classifier_eval(
            sources,
            os.path.abspath(args.split_dir),
            clinical,
            os.path.abspath(args.out_dir),
            bootstrap_n=args.bootstrap_n,
            eval_diagnostics=args.eval_diagnostics,
            label_shuffle_n=args.label_shuffle_n,
            std_diag_topk=args.std_diag_topk,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            merge_existing_root_metrics=args.merge_existing_root_metrics,
        )
        return

    if args.compare_npz_dirs:
        dirs = [os.path.abspath(d) for d in args.compare_npz_dirs]
        if len(dirs) != 2:
            print("compare mode currently supports exactly 2 directories", file=sys.stderr)
            sys.exit(1)
        names = args.compare_names or [os.path.basename(d.rstrip("/")) for d in dirs]
        if len(names) != 2:
            print("--compare_names must have 2 entries", file=sys.stderr)
            sys.exit(1)
        run_compare(
            dirs, names, clinical, args.out_dir,
            pooling=args.pooling,
            standardize=not args.no_standardize,
            l2_normalize=args.l2_normalize,
        )
        return

    if not args.npz_dir:
        print("Provide --npz_dir or --compare_npz_dirs", file=sys.stderr)
        sys.exit(1)

    run_full_pipeline(
        name=args.run_name,
        npz_dir=os.path.abspath(args.npz_dir),
        clinical=clinical,
        out_dir=args.out_dir,
        pooling=args.pooling,
        standardize=not args.no_standardize,
        l2_normalize=args.l2_normalize,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_sweep=args.umap_sweep,
        run_cluster_purity=args.cluster_purity,
    )


if __name__ == "__main__":
    main()
