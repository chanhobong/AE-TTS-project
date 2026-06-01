# ae-tts-pipeline

Dual slice-level autoencoders (Plain + MONAI) → spatial Stage B (schema v2) → patient-level downstream evaluation on cardiac CT (TTS vs control).

**This repository contains code and documentation only.** NIfTI volumes, checkpoints, NPZ embeddings, and train/val/test CSV splits are **not** included.

## Quick start

```bash
export AE_TTS_ROOT="$(pwd)"   # repo root after clone
cd "${AE_TTS_ROOT}"

# 1) Stage A — train AEs on train+val patients only (see stage_a/README.md)
# 2) Stage B — frozen encoder → spatial tokens + k-means → patient .npz (see stage_b/README.md)
# 3) Stage C — classifiers / repeated splits (utils/scripts/; see PIPELINE.md)
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
| [utils/scripts/](utils/scripts/) | Stage C — repeated eval, ensemble, spatial viz |
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

## Topics

`medical-imaging`, `autoencoder`, `ct`, `representation-learning`, `takotsubo`

## Citation

TBD — MPhil thesis, University of Manchester.
