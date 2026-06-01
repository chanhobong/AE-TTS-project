#!/usr/bin/env bash
# Run only pooling=p90_p10_regional_vol_mm (plain_ae + monai_ae — same defaults as
# run_repeated_head_sweep_p90_volmm.sh), then merge all run_* under OUT_PARENT.
#
# Use when that pooling failed / was interrupted; optionally delete the partial run dir first:
#   rm -rf "${LATENT_VOL}/repeated_head_sweep_p90_volmm/run_${RUN_TAG_BASE}_p90_p10_regional_vol_mm"
#
# Usage:
#   RUN_TAG_BASE=20260511_153350 LATENT_VOL=/path/to/latent_data \
#     bash utils/scripts/run_p90_p10_regional_vol_mm_and_merge.sh
#
# Same optional env as the main script: NPZ_MODE, OUT_PARENT, N_SPLITS, SWEEP_N_JOBS, INCLUDE_MONAI, …
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POOLINGS="p90_p10_regional_vol_mm"
export POOLING_PARALLEL="${POOLING_PARALLEL:-1}"
export RUN_COEF_STABILITY="${RUN_COEF_STABILITY:-0}"
export COEF_ONLY="${COEF_ONLY:-0}"

exec bash "${SCRIPT_DIR}/run_repeated_head_sweep_p90_volmm.sh"
