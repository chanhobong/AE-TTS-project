# Stage B spatial v2 (canonical bundle)

**Purpose:** One folder for the **clean** Plain + MONAI spatial-token Stage B runs  
(`schema_version=2`, train+val k-means fit, test assign-only).

Outputs (keep on disk; do not duplicate NPZ here):

| Track | `OUT_DIR` |
|-------|-----------|
| Plain AE | `AE_TTS/outputs/diff3dformer_stageB_plain_ae_spatial_v2/` |
| MONAI AE | `AE_TTS/outputs/diff3dformer_stageB_monai_ae_spatial_v2/` |

Slurm logs proving split policy: `logs/plain_slurm-15560735.out`, `logs/monai_slurm-15560789.out`.

---

## What lives here

```
stageB_spatial_v2/
  README.md                 ← this file
  DEPRECATED.md             ← old Stage B / eval to retire
  plain_ae/
    extract_embeddings_and_kmeans_spatial_tokens.py
    run_stageB.sh
  monai_ae/
    extract_embeddings_and_kmeans_spatial_tokens.py
    run_stageB.sh
  logs/
```

**Not copied** (still under `AE_TTS/` — too large / shared):

- Stage A training: use **`AE_TTS/stageA/`** bundle (`plain_ae/`, `monai_ae/`, `shared/`)
- Full tree also at `plain_AE_stageA/`, `MONAI/`, `diffae-master/` (legacy layout)
- Checkpoints: see below

---

## Stage A checkpoints (frozen for these NPZ)

| Track | Default checkpoint | Train script |
|-------|-------------------|--------------|
| Plain | `outputs/train_plain_ae/checkpoint_best.pth` | `stageA/plain_ae/run_train.sh` |
| MONAI | `MONAI/outputs/monai_ae_default/monai_ae_best.pth` | `stageA/monai_ae/run_train.sh` |

Both Stage A runs use **`data/train.csv` + `data/val.csv` only** (no test). Val = early stop / best ckpt.

---

## Split policy (Stage B)

- **K-means fit:** `train.csv` ∪ `val.csv` (115 patients in May 2025 run)
- **Assign-only:** `test.csv` (29 patients)
- **Do not use** `--extra_csvs` (legacy: fits test into k-means)

---

## Re-run

```bash
cd /mnt/iusers01/eee01/g09698cb/scratch
sbatch AE_TTS/stageB_spatial_v2/plain_ae/run_stageB.sh
sbatch AE_TTS/stageB_spatial_v2/monai_ae/run_stageB.sh
```

Use a **new** `OUT_DIR` if you change k or checkpoint so old NPZ are not overwritten.

---

## NPZ schema (v2)

Per patient: token rows `(n_slices × H'×W') × d` with `embeddings`, `embeddings_l2`, `cluster_ids`, `slice_indices`, `spatial_flat_idx`, `spatial_x/y`, `slice_z_per_slice`, `slice_z_mm`, …

Plain: bottleneck typically **4×4**, **d=512** (GAP skipped; Norm→SiLU→1×1 conv on spatial map).  
MONAI: bottleneck from `encode()+intermediate()`, **no GAP**.

---

## Downstream (not in this bundle)

- **`Loss_experiment/eval.py`** — 100 random splits; **not** used for these two OUT_DIRs. Do not use for primary test metrics.
- Stage C: `scripts/train_cluster_vit.py` + fixed `test.csv` when ready.

---

## Sync note

Duplicates also exist at:

- `plain_AE_stageA/StageB_new/` (Plain — edit **either** here or there, keep in sync)
- `MONAI/extract_embeddings_and_kmeans_spatial_tokens.py` + `MONAI/run_stageB_monai_ae_spatial_tokens.sh`

**Canonical for documentation / cleanup:** prefer **`AE_TTS/stageB_spatial_v2/`**.
