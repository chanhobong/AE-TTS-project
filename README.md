# AT-TTS-pipeline

Dual slice-level autoencoders (Plain + MONAI) → spatial Stage B (schema v2) → patient-level downstream evaluation on cardiac CT (TTS vs control).

**This repository contains code and documentation only.** NIfTI volumes, checkpoints, NPZ embeddings, and train/val/test CSV splits are **not** included.

## Main results (thesis canonical)

100 repeated **patient-level** splits (`RUN_TAG=mytag`). Details: [`results/README.md`](results/README.md).

| Method | ROC-AUC (mean ± std) | PR-AUC |
|--------|---------------------|--------|
| Plain · p90_p10_vol_mm + EN-LR | 0.874 ± 0.056 | 0.877 |
| MONAI · cluster_hist + RBF-SVC | 0.796 ± 0.071 | 0.831 |
| Late ensemble w=0.5 | **0.891 ± 0.053** | 0.908 |
| Ensemble + age/sex LR stack | **0.915 ± 0.048** | 0.930 |

Reproduce (local NPZ + splits): `bash stage_c/run/run_paired_ensemble_mytag.sh`

## Quick start

```bash
export AE_TTS_ROOT="$(pwd)"   # repo root after clone
cd "${AE_TTS_ROOT}"

# 1) Stage A — train AEs on train+val patients only (see stage_a/README.md)
# 2) Stage B — frozen encoder → spatial tokens + k-means → patient .npz (see stage_b/README.md)
# 3) Stage C — classifiers / repeated splits (stage_c/; see PIPELINE.md)
```

Place your split files locally (not in git):

```text
${AE_TTS_ROOT}/data/splits/train.csv
${AE_TTS_ROOT}/data/splits/val.csv
${AE_TTS_ROOT}/data/splits/test.csv
```

## Pipeline (30 s)

```text
data/splits (train+val) ──► Stage A: AE train ──► checkpoint .pth
                                      │
                                      ▼
                         Stage B: spatial tokens + k-means
                                      │
                                      ▼
                         patient .npz (schema v2)
                                      │
                                      ▼
                         Stage C: pooling + classifier + (optional) ensemble
```

## Split policy (summary)

| Stage | Patients in fit |
|-------|-----------------|
| Stage A AE training | **train + val** only |
| Stage B k-means fit | **train + val** slice embeddings |
| Stage B test / external | assign-only to fixed prototypes |
| Stage C evaluation | patient-level CV; **test never in AE fit** |

Details: [PIPELINE.md](PIPELINE.md)

## Directory map

| Path | Role |
|------|------|
| [stage_a/](stage_a/) | Stage A — Plain AE & MONAI AE training |
| [stage_b/](stage_b/) | Stage B — embeddings, spatial v2 NPZ, k-means |
| [stage_c/](stage_c/) | Stage C — patient pooling, classifiers, ensemble |
| [results/](results/) | Frozen thesis metrics (`clinical_2`, no patient IDs) |
| [data/](data/) | Split CSV **schema** (files stay local) |

Legacy trees (`plain_AE_stageA/`, old Stage B, etc.) stay **outside** this repo; see [stage_b/DEPRECATED.md](stage_b/DEPRECATED.md).

## Reproducibility assets (external)

Host on **GitHub Release**, Zenodo, or Hugging Face; link SHA/path in README:

- Stage A checkpoints (`plain_ae_best.pth`, `monai_ae_best.pth`)
- Stage B NPZ directories + `prototypes.npy`
- Optional: slurm logs under `stage_b/logs/` (small; prove train+val-only AE)

## Releases (suggested tags)

- `v0.1-stage-a` — Stage A scripts frozen
- `v0.2-stage-b-spatial-v2` — Stage B + schema v2 NPZ layout
- `v0.3-stage-c` — Stage C eval + `clinical_2` main results ([`results/`](results/))

## Topics

`medical-imaging`, `autoencoder`, `ct`, `representation-learning`, `takotsubo`

## Citation

TBD — MPhil thesis, University of Manchester.
