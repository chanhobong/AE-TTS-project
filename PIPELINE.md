# Pipeline: Stage A → B → C

Official entry points: **`stage_a/`** and **`stage_b/`**.  
Stage C (classification, repeated splits, ensemble) currently lives in **`utils/scripts/`** until a `stage_c/` package is split out.

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

## Stage C — downstream (current: `utils/scripts/`)

Patient-level features from NPZ (mean / std / cluster_hist / …) → logistic or SVC → optional late fusion (Plain + MONAI probabilities).

Key scripts:

| Script | Purpose |
|--------|---------|
| `repeated_stratified_shuffle_eval.py` | 100× patient-level stratified splits |
| `ensemble_oos_same_split_eval.py` | Matched-split Plain/MONAI ensemble |
| `clinical_vs_clinical_plus_ensemble_repeat.py` | Clinical-only vs clinical+ensemble |
| `plot_spatial_coeff_grid_overlay.py` | Spatial coef overlays (schema v2) |

Split policy for repeated eval: pool **all** labelled patients with NPZ; split is **patient-level** after pooling; scaler fit on train fold only.

Paths helper: `utils/scripts/spatial_viz_paths.example.sh` (set `VOL_PHD_PROJECT`, `PATIENT_ID`, etc.).

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
