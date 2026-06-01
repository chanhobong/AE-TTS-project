#!/usr/bin/env bash
# MONAI AE: repeated StratifiedShuffleSplit on mean | std | cluster_hist (default 100 splits each).
# Calls repeated_stratified_shuffle_eval.py via run_monai_ae_pooling_logistic_repeat.sh.
#
# Default classifier: logistic only. Add RBF SVM with CLASSIFIERS (space-separated), e.g.:
#   CLASSIFIERS="logistic rbf_svc" RUN_TAG=v1 OUT_DIR=... bash utils/scripts/run_monai_ae_lr_mean_std_cluster_hist_100.sh
# RBF hyperparameters: SVC_C (default 1.0), SVC_GAMMA (default scale) — see run_monai_ae_pooling_logistic_repeat.sh
#
# Outputs (per pooling + classifier):
#   OUT_DIR/run_<RUN_TAG>/monai_ae/<mean|std|cluster_hist>/<logistic|rbf_svc|...>/
#
# Usage:
#   LATENT_VOL=/Volumes/.../latent_data bash utils/scripts/run_monai_ae_lr_mean_std_cluster_hist_100.sh
#   OUT_DIR=/path/to/Results/MONAI_lr_100 RUN_TAG=v1 bash utils/scripts/run_monai_ae_lr_mean_std_cluster_hist_100.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${LATENT_VOL:-}" ]]; then
  _L_REPO="${REPO_ROOT}/latent_data"
  _L_VOL="/Volumes/Chanho_PhD_Project/latent_data"
  if [[ -d "${_L_REPO}/diff3dformer_stageB_monai_ae/patients" ]]; then
    LATENT_VOL="${_L_REPO}"
  elif [[ -d "${_L_VOL}/diff3dformer_stageB_monai_ae/patients" ]]; then
    LATENT_VOL="${_L_VOL}"
  else
    LATENT_VOL="${_L_REPO}"
  fi
fi
export LATENT_VOL

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
N_SPLITS="${N_SPLITS:-100}"
CLASSIFIERS="${CLASSIFIERS:-logistic}"
export OUT_DIR="${OUT_DIR:-${LATENT_VOL}/out_monai_ae_lr_mean_std_cluster_hist}"

export RUN_TAG
export N_SPLITS

for POOLING in mean std cluster_hist; do
  for CLASSIFIER in ${CLASSIFIERS}; do
    echo "========== POOLING=${POOLING}  CLASSIFIER=${CLASSIFIER} =========="
    export CLASSIFIER
    POOLING="${POOLING}" bash "${SCRIPT_DIR}/run_monai_ae_pooling_logistic_repeat.sh"
  done
done

echo ""
echo "Summaries (run_${RUN_TAG}):"
for POOLING in mean std cluster_hist; do
  for CLASSIFIER in ${CLASSIFIERS}; do
    f="${OUT_DIR}/run_${RUN_TAG}/monai_ae/${POOLING}/${CLASSIFIER}/summary.csv"
    if [[ -f "${f}" ]]; then
      echo "--- ${f} ---"
      column -t -s, "${f}" 2>/dev/null || cat "${f}"
    fi
  done
done
