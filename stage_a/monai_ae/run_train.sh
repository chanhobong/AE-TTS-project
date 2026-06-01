#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 0-2:0
#
# MONAI AE Stage A (canonical: AE_TTS/stageA/monai_ae/)
#
#   sbatch AE_TTS/stageA/monai_ae/run_train.sh
#   OUT_DIR=... sbatch AE_TTS/stageA/monai_ae/run_train.sh

set -euo pipefail

ROOT="/mnt/iusers01/eee01/g09698cb/scratch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON="/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/bin/python"

if command -v module >/dev/null 2>&1; then
  module purge
  module load libs/cuda
fi

OUT_DIR="${OUT_DIR:-${ROOT}/AE_TTS/MONAI/outputs/monai_ae_default}"

exec "$PYTHON" -u "$SCRIPT_DIR/train_monai_ae.py" \
  --ct_root "/mnt/iusers01/eee01/g09698cb/Dataset/normal_RM_V2" \
  --metadata_csv "/mnt/iusers01/eee01/g09698cb/Dataset/Combined_Labels.csv" \
  --train_csv "${ROOT}/AE_TTS/data/train.csv" \
  --val_csv "${ROOT}/AE_TTS/data/val.csv" \
  --out_dir "$OUT_DIR" \
  --expected_shape 128 128 64 \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --loss l1 \
  --num_workers 2 \
  --patience 10 \
  "$@"
