# Stage C — downstream evaluation

**Input:** per-patient `.npz` from [Stage B](../stage_b/README.md) (schema v2).  
**Output:** CSV summaries (ROC/PR over repeated patient-level splits), optional ensemble tables.

Split CSVs are **local only** → `data/splits/{train,val,test}.csv` (not in git).

---

## Layout

```text
stage_c/
  README.md           ← this file
  eval/               ← core Python
  run/                ← bash entry points
  analysis/           ← confounding, benchmark tables, plots
  viz/                ← spatial coef overlays (appendix)
  docs/               ← repeated-eval notes
```

---

## Quick start

### Thesis canonical (`clinical_2`, `RUN_TAG=mytag`)

Uses Plain **p90_p10_vol_mm + Elastic-Net LR** and MONAI **cluster_hist + RBF-SVC** on the repeat_2 NPZ layout. Frozen numbers: [`../results/README.md`](../results/README.md).

```bash
export AE_TTS_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${AE_TTS_ROOT}"

mkdir -p data/splits
# cp .../train.csv data/splits/  etc.

export LATENT_VOL=/Volumes/Chanho_PhD_Project/latent_data
export RUN_TAG=mytag

bash stage_c/run/run_paired_ensemble_mytag.sh
```

Skip tower re-run if OOS CSVs already exist: `SKIP_TOWER_RERUN=1 bash stage_c/run/run_paired_ensemble_mytag.sh`

| Tower | Pooling | Classifier |
|-------|---------|------------|
| Plain | `p90_p10_vol_mm` | `logistic_en_C0.1_r0.2` |
| MONAI | `cluster_hist` | `rbf_svc` |

Outputs:

```text
${LATENT_VOL}/out_plain_p90_p10_vol_mm_head_repeat_combo_plain_slice_meta_ae2/run_${RUN_TAG}/...
${LATENT_VOL}/out_monai_ae_cluster_hist_rbf_repeat_2/run_${RUN_TAG}/...
${LATENT_VOL}/ensemble_fix_mytag_clinical_2/run_${RUN_TAG}/summary_methods.csv
```

### Generic demo (spatial_v2-friendly defaults)

```bash
export RUN_TAG=thesis_v1
bash stage_c/run/run_paired_ensemble.sh
```

Defaults inside `run_paired_ensemble.sh`:

| Tower | Pooling | Classifier |
|-------|---------|------------|
| Plain | `std` | logistic |
| MONAI | `cluster_hist` | rbf_svc |

Outputs:

```text
${LATENT_VOL}/out_plain_ae_std_repeat/run_${RUN_TAG}/...
${LATENT_VOL}/out_monai_ae_cluster_hist_repeat/run_${RUN_TAG}/...
${LATENT_VOL}/ensemble_${RUN_TAG}/run_${RUN_TAG}/summary_methods.csv
```

---

## Run scripts

| Script | Purpose |
|--------|---------|
| [run/run_plain_repeat.sh](run/run_plain_repeat.sh) | Plain NPZ only |
| [run/run_monai_repeat.sh](run/run_monai_repeat.sh) | MONAI NPZ only |
| [run/run_paired_ensemble.sh](run/run_paired_ensemble.sh) | Generic: std + cluster_hist + fusion |
| [run/run_paired_ensemble_mytag.sh](run/run_paired_ensemble_mytag.sh) | **Thesis canonical** (`clinical_2`) |

Environment:

| Variable | Meaning |
|----------|---------|
| `LATENT_VOL` | Directory containing NPZ trees |
| `NPZ_DIR` | Override auto-detected patients folder |
| `SPLIT_DIR` | Default `${AE_TTS_ROOT}/data/splits` |
| `POOLING` | `mean` \| `std` \| `mean_std` \| `cluster_hist` |
| `N_SPLITS` | Default 100 |
| `WRITE_TEST_PRED=1` | Export OOS probs (set by paired script) |

NPZ search order (Plain): `stage_b/plain_ae/patients` → legacy `diff3dformer_*` paths.

---

## Eval modules (`eval/`)

| File | Role |
|------|------|
| `repeated_stratified_shuffle_eval.py` | Patient-level StratifiedShuffleSplit × N |
| `ensemble_oos_same_split_eval.py` | Matched-split Plain/MONAI late fusion |
| `clinical_only_repeated_eval.py` | Age + sex baseline |
| `clinical_vs_clinical_plus_ensemble_repeat.py` | Clinical vs clinical+ensemble |

Example (manual):

```bash
python3 stage_c/eval/repeated_stratified_shuffle_eval.py \
  --labels_csv data/splits/train.csv \
  --labels_csv data/splits/val.csv \
  --labels_csv data/splits/test.csv \
  --latent_source plain_ae /path/to/npz/patients \
  --pooling std --classifier logistic \
  --n_splits 100 --out_dir /path/out --run_tag manual_001
```

---

## Analysis (`analysis/`) — optional

| File | Role |
|------|------|
| `plain_ae_metadata_robustness.py` | Age/sex confounding, subgroup AUC |
| `build_benchmark_comparison_table.py` | Merge method summaries into one table |
| `analyze_ensemble_paired_splits.py` | Paired split diagnostics |
| `plot_ensemble_per_split_auc_distributions.py` | Violin/box per-split AUC |

---

## Visualization (`viz/`) — appendix

Spatial coef overlays on Stage B NPZ (schema v2):

```bash
source stage_c/viz/spatial_viz_paths.example.sh
# see stage_c/docs/SPATIAL_VIZ_COMMANDS.md
```

---

## Split / leakage policy

- Evaluation uses **frozen** NPZ from Stage B.
- Repeated splits are **patient-level** after pooling (not slice-level).
- Test patients were **not** in Stage A AE training (see [PIPELINE.md](../PIPELINE.md)).

---

## Legacy

Older exploratory scripts remain under `utils/scripts/` (DiffAE, flow matching, MIL, etc.).  
**Canonical Stage C = this folder.** See [../utils/scripts/README.md](../utils/scripts/README.md).
