# Stage A (canonical bundle)

**Purpose:** Minimal files to **train** Plain AE and MONAI AE (Stage A only).  
Split: **`train.csv` + `val.csv` only** — no test in weights.

Checkpoints used by spatial v2 Stage B:

| Track | Default `OUT_DIR` | Stage B bundle |
|-------|-------------------|----------------|
| Plain | `AE_TTS/outputs/train_plain_ae/checkpoint_best.pth` | `stageB_spatial_v2/plain_ae/` |
| MONAI | `AE_TTS/MONAI/outputs/monai_ae_default/monai_ae_best.pth` | `stageB_spatial_v2/monai_ae/` |

---

## Layout

```
stageA/
  README.md
  plain_ae/
    train_plain_ae.py
    plain_ae.py
    run_train.sh
    run_train_mse_ssim.sh
  monai_ae/
    train_monai_ae.py
    AE.py              # reference; default uses installed monai
    run_train.sh
  shared/
    data/dataset.py    # CTROIVolumeDataset
    diffae/            # minimal BeatGANs encoder + CTSliceDataset wrapper
      dataset.py
      choices.py
      config_base.py
      model/{unet,blocks,nn}.py
```

**Not copied** (paths in run scripts):

- Patient CSVs: `AE_TTS/data/{train,val,test}.csv`
- NIfTI: `Dataset/normal_RM_V2`, labels `Combined_Labels.csv`
- Checkpoints / tensorboard (written to `outputs/`)

---

## Re-run Stage A

```bash
cd /mnt/iusers01/eee01/g09698cb/scratch
sbatch AE_TTS/stageA/plain_ae/run_train.sh
sbatch AE_TTS/stageA/monai_ae/run_train.sh

# Plain MSE+SSIM (separate ckpt dir):
sbatch AE_TTS/stageA/plain_ae/run_train_mse_ssim.sh
```

Override output:

```bash
PLAIN_AE_OUT_DIR=/path/to/new_ckpt sbatch AE_TTS/stageA/plain_ae/run_train.sh
OUT_DIR=/path/to/monai_ckpt sbatch AE_TTS/stageA/monai_ae/run_train.sh
```

---

## Defaults (match published spatial_v2 runs)

| | Plain | MONAI |
|---|-------|-------|
| Loss | L1 | L1 |
| LR | 1e-4 | 1e-4 |
| Batch | 4 | 4 |
| Epochs | 100 | 100 |
| Early stop | patience 20 (Plain) | patience 10 |
| EMA | Plain default on | none |

---

## Pipeline order

1. **Stage A** → this folder  
2. **Stage B spatial v2** → `stageB_spatial_v2/`  
3. Stage C (later) → fixed `test.csv`

---

## Sync note

Duplicates remain at `plain_AE_stageA/`, `MONAI/`, full `diffae-master/`.  
**Edit here first** for cleanup; keep originals until legacy dirs are removed.

See also `stageB_spatial_v2/DEPRECATED.md` for legacy scripts to retire.
