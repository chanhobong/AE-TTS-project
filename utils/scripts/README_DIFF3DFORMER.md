## Diff3Dformer (Plan-1: 2D axial slice sequence)

This README describes the run order for Stage A → B → C.

### Stage A — Train DAE2D (slice diffusion autoencoder)
Outputs: `encoder_checkpoint.pth`

```bash
python /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/scripts/train_dae2d.py \
  --normal_root /path/to/normal_RM_V2 \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/train.csv \
  --val_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/val.csv \
  --out_dir /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/train_dae2d \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4
```

### Stage B — Extract embeddings + spherical k-means
Outputs: `prototypes.npy` and per-patient `.npz` files

```bash
python /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/scripts/extract_embeddings_and_kmeans.py \
  --normal_root /path/to/normal_RM_V2 \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/train.csv \
  --checkpoint /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/train_dae2d/encoder_checkpoint.pth \
  --out_dir /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/diff3dformer_stageB \
  --k_clusters 64
```

### Stage C — Train clustering ViT
Outputs: `cluster_vit_best.pth` and `loss_history.csv`

```bash
python /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/scripts/train_cluster_vit.py \
  --npz_dir /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/diff3dformer_stageB/patients \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/train.csv \
  --val_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/val.csv \
  --out_dir /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/diff3dformer_stageC \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4
```
