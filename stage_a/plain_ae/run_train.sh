#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#
# Plain AE Stage A (canonical: AE_TTS/stageA/plain_ae/)
#
# Override before sbatch:
#   PLAIN_AE_LOSS        l1 | mse | mse_ssim  (default: l1)
#   PLAIN_AE_OUT_DIR     checkpoint dir     (default: .../outputs/train_plain_ae)
#   SSIM_WEIGHT / SSIM_DATA_RANGE  for mse_ssim

set -euo pipefail

export LD_LIBRARY_PATH="/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
module purge
module load libs/cuda

ROOT=/mnt/iusers01/eee01/g09698cb/scratch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/mnt/iusers01/eee01/g09698cb/miniconda3/envs/TTS_new_2/bin/python

PLAIN_AE_LOSS="${PLAIN_AE_LOSS:-l1}"
PLAIN_AE_OUT_DIR="${PLAIN_AE_OUT_DIR:-${ROOT}/AE_TTS/outputs/train_plain_ae}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
SSIM_DATA_RANGE="${SSIM_DATA_RANGE:-2.0}"

py_args=(
  -u "$SCRIPT_DIR/train_plain_ae.py"
  --ct_root /mnt/iusers01/eee01/g09698cb/Dataset/normal_RM_V2
  --metadata_csv /mnt/iusers01/eee01/g09698cb/Dataset/Combined_Labels.csv
  --train_csv "${ROOT}/AE_TTS/data/train.csv"
  --val_csv "${ROOT}/AE_TTS/data/val.csv"
  --out_dir "$PLAIN_AE_OUT_DIR"
  --expected_shape 128 128 64
  --epochs 100
  --batch_size 4
  --lr 1e-4
  --loss "$PLAIN_AE_LOSS"
)
if [[ "$PLAIN_AE_LOSS" == "mse_ssim" ]]; then
  py_args+=(--ssim_weight "$SSIM_WEIGHT" --ssim_data_range "$SSIM_DATA_RANGE")
fi

exec "$PYTHON" "${py_args[@]}"
