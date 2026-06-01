#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#
# Plain AE Stage A — MSE+SSIM variant (separate out_dir).

set -euo pipefail

ROOT=/mnt/iusers01/eee01/g09698cb/scratch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLAIN_AE_LOSS=mse_ssim
export PLAIN_AE_OUT_DIR="${PLAIN_AE_OUT_DIR:-${ROOT}/AE_TTS/outputs/train_plain_ae_mse_ssim}"
export SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
export SSIM_DATA_RANGE="${SSIM_DATA_RANGE:-2.0}"

exec bash "$SCRIPT_DIR/run_train.sh"
