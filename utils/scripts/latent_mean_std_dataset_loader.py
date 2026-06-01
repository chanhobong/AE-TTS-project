"""
Load NPZ slice embeddings -> patient-level mean||std vectors + train/val/test alignment.

Duplicated from latent_space_pca_umap (subset only). This module does **not** import umap,
so scripts that only need MIL-style features avoid TensorFlow / NumPy-2 wheel issues.

Keep behavior identical to latent_space_pca_umap.build_aligned_dataset_mean_std.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def _merge_labels_and_clinical(paths: list[str]) -> pd.DataFrame:
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
    out = out.drop_duplicates(subset=["patient_id"], keep="last")
    return out


def discover_npz_files(npz_dir: str) -> list[str]:
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)
    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    return [os.path.join(npz_dir, f) for f in names]


@dataclass
class PatientRecord:
    patient_id: str
    embeddings: np.ndarray
    patient_vec: np.ndarray
    label: int
    age: float
    sex: str
    dominant_cluster: Optional[int] = None


def aggregate_mean_std_concat(emb: np.ndarray) -> np.ndarray:
    m = emb.mean(axis=0)
    s = emb.std(axis=0, ddof=0)
    return np.concatenate([m, s], axis=0)


def load_data(
    npz_dir: str,
    clinical: pd.DataFrame,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[PatientRecord], list[str]]:
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


def build_aligned_dataset(
    npz_dir: str,
    clinical: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def load_split_patient_ids(split_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
