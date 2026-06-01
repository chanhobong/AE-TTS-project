#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#
# MONAI AE — spatial tokens Stage B (schema v2).
# Canonical copy: AE_TTS/stageB_spatial_v2/monai_ae/

set -euo pipefail

export LD_LIBRARY_PATH="/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
module purge
module load libs/cuda

ROOT=/mnt/iusers01/eee01/g09698cb/scratch
AE_TTS="$ROOT/AE_TTS"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT="${CHECKPOINT:-$AE_TTS/MONAI/outputs/monai_ae_default/monai_ae_best.pth}"
OUT_DIR="${OUT_DIR:-$AE_TTS/outputs/diff3dformer_stageB_monai_ae_spatial_v2}"
VERIFY_NIFTI_SLICE_GEOMETRY="${VERIFY_NIFTI_SLICE_GEOMETRY:-0}"

PY_VERIFY=()
if [ "$VERIFY_NIFTI_SLICE_GEOMETRY" = "1" ]; then
  PY_VERIFY=(--verify_nifti_slice_geometry)
fi

/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/bin/python -u \
  "$SCRIPT_DIR/extract_embeddings_and_kmeans_spatial_tokens.py" \
  --normal_root /mnt/iusers01/eee01/g09698cb/Dataset/normal_RM_V2 \
  --tts_root /mnt/iusers01/eee01/g09698cb/Dataset/TTS_RM_V2 \
  --metadata_csv /mnt/iusers01/eee01/g09698cb/Dataset/Combined_Labels.csv \
  --train_csv "$AE_TTS/data/train.csv" \
  --val_csv "$AE_TTS/data/val.csv" \
  --assign_only_csvs "$AE_TTS/data/test.csv" \
  --expected_shape 128 128 64 \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT_DIR" \
  --batch_slices 32 \
  --k_clusters 64 \
  "${PY_VERIFY[@]}" \
  "$@"
