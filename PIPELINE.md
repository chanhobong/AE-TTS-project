# Pipeline: Stage A → B → C

Official entry points: **`stage_a/`**, **`stage_b/`**, and **`stage_c/`**.

Environment (all stages):

```bash
export AE_TTS_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export SPLIT_DIR="${AE_TTS_ROOT}/data/splits"   # local only; not in git
```

---

## Data (local, not in repository)

- ROI NIfTI: external disk (e.g. `TTS_RM_V2/`, `normal_RM_V2/`) — see [data/README.md](data/README.md).
- Splits: `data/splits/{train,val,test}.csv` with columns at minimum `patient_id`, `case` (or `label`), `age`, `sex`.
- **No patient imaging or CSV files are committed.**

---

## Stage A — slice autoencoders

**Directory:** [stage_a/](stage_a/)

| Item | Policy |
|------|--------|
| Input | Axial CT slices from **train + val** patient IDs |
| Excluded | **test.csv** patients — never used for AE weight updates |
| Outputs | `checkpoint_best.pth` (Plain), `monai_ae_best.pth` (MONAI) |
| Loss | L1 reconstruction (both towers by default) |

```text
train.csv + val.csv  →  CTSliceDataset  →  Plain AE / MONAI AE  →  .pth
```

Run scripts: `stage_a/plain_ae/`, `stage_a/monai_ae/` (see [stage_a/README.md](stage_a/README.md)).

---

## Stage B — spatial tokens (schema v2) + k-means

**Directory:** [stage_b/](stage_b/)  
*(Renamed from `stageB_spatial_v2/`; spatial v2 = NPZ keys `spatial_y`, `spatial_x`, `slice_position_norm`, etc.)*

| Step | Description |
|------|-------------|
| 1 | Load **frozen** Stage A encoder |
| 2 | Extract slice embeddings + grid metadata per patient |
| 3 | Fit spherical k-means (K=64) on **train∪val** slice vectors (L2-normalized) |
| 4 | Assign test slices to nearest prototype (**no centroid refit**) |
| 5 | Write `patients/<patient_id>.npz` |

```text
checkpoint .pth  →  encoder(slide)  →  NPZ schema v2  →  downstream pooling
```

Run scripts: [stage_b/README.md](stage_b/README.md).  
Legacy Stage B (non-spatial): [stage_b/DEPRECATED.md](stage_b/DEPRECATED.md).

---

## Stage C — downstream

**Directory:** [stage_c/](stage_c/)

Patient-level features from NPZ (mean / std / cluster_hist / …) → logistic or SVC → optional late fusion (Plain + MONAI probabilities).

```bash
export LATENT_VOL=/Volumes/Chanho_PhD_Project/latent_data
export RUN_TAG=thesis_v1
bash stage_c/run/run_paired_ensemble.sh
```

| Component | Path |
|-----------|------|
| Repeated eval | `stage_c/eval/repeated_stratified_shuffle_eval.py` |
| Paired ensemble | `stage_c/run/run_paired_ensemble.sh` |
| Clinical baseline | `stage_c/eval/clinical_only_repeated_eval.py` |
| Spatial viz (appendix) | `stage_c/viz/` |

Split policy: pool all labelled patients with NPZ; **patient-level** split after pooling; scaler fit on train fold only.

Legacy exploratory scripts: `utils/scripts/` (see [utils/scripts/README.md](utils/scripts/README.md)).

---

## What not to commit

See [.gitignore](.gitignore): `outputs/`, `*.pth`, `*.npz`, `data/splits/*.csv`, NIfTI, legacy trees.

---

## Migration from old layout

| Old (local/HPC) | GitHub repo |
|-----------------|-------------|
| `stageA/` | `stage_a/` |
| `stageB_spatial_v2/` | `stage_b/` |
| `data/train.csv` | `data/splits/train.csv` (local) |
| `plain_AE_stageA/`, `MONAI/` at repo root | `_archive/` or omit |

After rename, set `AE_TTS_ROOT` and update any `ROOT=` in run scripts to `"${AE_TTS_ROOT}"`.
