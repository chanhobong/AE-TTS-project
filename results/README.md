# Results (thesis canonical)

Frozen downstream metrics for **100 repeated patient-level splits** (`RUN_TAG=mytag`, `StratifiedShuffleSplit`, test fraction 0.2, `random_state=42`).

No patient identifiers are stored here — only split-level and summary aggregates.

## Main results (`clinical_2`)

| # | Method | ROC-AUC (mean ± std) | PR-AUC |
|---|--------|---------------------|--------|
| 0 | **Clinical · age + sex** (LR) | **0.750 ± 0.081** | 0.756 |
| 1 | Plain · p90_p10_vol_mm + EN-LR | 0.874 ± 0.056 | 0.877 |
| 2 | MONAI · cluster_hist + RBF-SVC | 0.796 ± 0.071 | 0.831 |
| 3 | Late ensemble w=0.5 | **0.891 ± 0.053** | 0.908 |
| 4 | Ensemble + age/sex LR stack | **0.915 ± 0.048** | 0.930 |

Source: [`main_summary_methods.csv`](main_summary_methods.csv) (rows 1–4); clinical baseline from [`towers/clinical_age_sex_lr_summary.csv`](towers/clinical_age_sex_lr_summary.csv) (100 splits, same protocol as imaging towers).

### Tower configuration

| Tower | Pooling / readout | Classifier | NPZ layout (local) |
|-------|-------------------|------------|-------------------|
| Plain | `p90_p10_vol_mm` | `logistic_en_C0.1_r0.2` | `…/diff3dformer_stageB_plain_ae_slice_meta/patients` |
| MONAI | `cluster_hist` (K=64) | `rbf_svc` | `…/diff3dformer_stageB_monai_ae/patients` (repeat_2 eval tree) |

Imaging-only ensemble (**0.891**) vs clinical baseline (**0.750**) → **+0.14 ROC** incremental signal (same repeated-split protocol).

The **+ age/sex** row fits `LogisticRegression(class_weight=balanced)` on MinMax-scaled `[ens_w0.5]`, age, and sex. The ensemble score at train time uses a **leave-one-split-out** mean of fixed-w ensemble probs from other splits (see [`ensemble_clinical2_meta.json`](ensemble_clinical2_meta.json)).

## Files

| File | Contents |
|------|----------|
| `main_summary_methods.csv` | Primary method summary (plain / monai / ensemble / +clinical) |
| `ensemble_clinical2_per_split_metrics.csv` | Per-split ROC/PR for tail quantiles |
| `ensemble_clinical2_meta.json` | Run metadata (classifiers, join counts; paths redacted for git) |
| `towers/clinical_age_sex_lr_summary.csv` | Age + sex baseline (100 splits) |
| `towers/plain_p90_p10_vol_mm_en_summary.csv` | Plain single-tower summary |
| `benchmark_manifest.json` | Machine-readable row spec |
| `benchmark_comparison_table.csv` | LaTeX/Excel-friendly merged table |

## Reproduce (requires local NPZ + split CSVs)

```bash
export AE_TTS_ROOT="$(git rev-parse --show-toplevel)"
cd "${AE_TTS_ROOT}"

# Splits (never commit): data/splits/{train,val,test}.csv
export LATENT_VOL=/path/to/latent_data
export RUN_TAG=mytag

bash stage_c/run/run_paired_ensemble_mytag.sh
```

Ensemble-only if OOS CSVs already exist:

```bash
SKIP_TOWER_RERUN=1 bash stage_c/run/run_paired_ensemble_mytag.sh
```

## Regenerate comparison table

```bash
python3 stage_c/analysis/build_benchmark_comparison_table.py \
  --manifest results/benchmark_manifest.json \
  --out_csv results/benchmark_comparison_table.csv
```

## Not in this folder

- Stage A checkpoints, Stage B NPZ volumes, and split CSVs → GitHub **Release** or private storage (see [GITHUB_SETUP.md](../GITHUB_SETUP.md)).
- Exploratory runs (`ensemble_fix_mytag_clinical` without `_2`, spatial_v2 mean pooling) → supplementary only.
