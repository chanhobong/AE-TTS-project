#!/usr/bin/env bash
# Run Plain AE and MONAI AE repeated_stratified_shuffle_eval with the **same RUN_TAG** and protocol,
# export test_oos_predictions.csv from both, then call ensemble_oos_same_split_eval.py.
#
# Prerequisites: data/train.csv val.csv test.csv under SPLIT_DIR; NPZ roots under LATENT_VOL.
#
# Required:
#   RUN_TAG  — shared tag so outputs land in run_${RUN_TAG}/ for both sides (set explicitly).
#
# Example (logistic on Plain, RBF on MONAI, std pooling):
#   LATENT_VOL=/Volumes/.../latent_data \
#   RUN_TAG=pair_std_001 POOLING=std \
#   CLASSIFIER_PLAIN=logistic CLASSIFIER_MONAI=rbf_svc \
#   bash utils/scripts/run_paired_plain_monai_oos_then_ensemble.sh
#
# If one side was trained with --classifier all, set e.g. FILTER_MODEL_PLAIN=logistic explicitly.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:?Set RUN_TAG (e.g. RUN_TAG=pair_20260209_001)}"
LATENT_VOL="${LATENT_VOL:-${REPO_ROOT}/latent_data}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
POOLING="${POOLING:-mean}"
N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
LOGISTIC_C="${LOGISTIC_C:-1.0}"
SVC_C="${SVC_C:-1.0}"
SVC_GAMMA="${SVC_GAMMA:-scale}"

CLASSIFIER_PLAIN="${CLASSIFIER_PLAIN:-logistic}"
CLASSIFIER_MONAI="${CLASSIFIER_MONAI:-rbf_svc}"
FILTER_MODEL_PLAIN="${FILTER_MODEL_PLAIN:-${CLASSIFIER_PLAIN}}"
FILTER_MODEL_MONAI="${FILTER_MODEL_MONAI:-${CLASSIFIER_MONAI}}"

OUT_PLAIN="${OUT_DIR_PLAIN:-${LATENT_VOL}/out_plain_ae_${POOLING}_logistic_repeat}"
OUT_MONAI="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_${POOLING}_logistic_repeat}"

CONCAT_AGE_SEX="${CONCAT_AGE_SEX:-0}"
FINAL_RESCALE_AFTER_CONCAT="${FINAL_RESCALE_AFTER_CONCAT:-0}"
CLUSTER_HIST_K="${CLUSTER_HIST_K:-}"

export LATENT_VOL
PY="${PYTHON:-python3}"
ENSEMBLE_OUT="${ENSEMBLE_OUT:-${LATENT_VOL}/ensemble_plain_monai_${POOLING}_${RUN_TAG}}"

echo "RUN_TAG=${RUN_TAG}  POOLING=${POOLING}  protocol: n_splits=${N_SPLITS} test_size=${TEST_SIZE} random_state=${RANDOM_STATE}"
echo "Plain:  classifier=${CLASSIFIER_PLAIN}  OUT_DIR=${OUT_PLAIN}"
echo "MONAI:  classifier=${CLASSIFIER_MONAI}  OUT_DIR=${OUT_MONAI}"

echo ""
echo "========== Plain AE (OOS export) =========="
WRITE_TEST_PRED=1 \
  CLASSIFIER="${CLASSIFIER_PLAIN}" \
  RUN_TAG="${RUN_TAG}" \
  POOLING="${POOLING}" \
  N_SPLITS="${N_SPLITS}" \
  TEST_SIZE="${TEST_SIZE}" \
  RANDOM_STATE="${RANDOM_STATE}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  OUT_DIR="${OUT_PLAIN}" \
  LOGISTIC_C="${LOGISTIC_C}" \
  SVC_C="${SVC_C}" \
  SVC_GAMMA="${SVC_GAMMA}" \
  CONCAT_AGE_SEX="${CONCAT_AGE_SEX}" \
  FINAL_RESCALE_AFTER_CONCAT="${FINAL_RESCALE_AFTER_CONCAT}" \
  CLUSTER_HIST_K="${CLUSTER_HIST_K}" \
  bash "${SCRIPT_DIR}/run_plain_ae_pooling_logistic_repeat.sh"

echo ""
echo "========== MONAI AE (OOS export) =========="
WRITE_TEST_PRED=1 \
  CLASSIFIER="${CLASSIFIER_MONAI}" \
  RUN_TAG="${RUN_TAG}" \
  POOLING="${POOLING}" \
  N_SPLITS="${N_SPLITS}" \
  TEST_SIZE="${TEST_SIZE}" \
  RANDOM_STATE="${RANDOM_STATE}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  OUT_DIR="${OUT_MONAI}" \
  LOGISTIC_C="${LOGISTIC_C}" \
  SVC_C="${SVC_C}" \
  SVC_GAMMA="${SVC_GAMMA}" \
  CONCAT_AGE_SEX="${CONCAT_AGE_SEX}" \
  FINAL_RESCALE_AFTER_CONCAT="${FINAL_RESCALE_AFTER_CONCAT}" \
  CLUSTER_HIST_K="${CLUSTER_HIST_K}" \
  bash "${SCRIPT_DIR}/run_monai_ae_pooling_logistic_repeat.sh"

PRED_PLAIN="${OUT_PLAIN}/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER_PLAIN}/test_oos_predictions.csv"
PRED_MONAI="${OUT_MONAI}/run_${RUN_TAG}/monai_ae/${POOLING}/${CLASSIFIER_MONAI}/test_oos_predictions.csv"

if [[ ! -f "${PRED_PLAIN}" ]]; then
  echo "Missing Plain OOS file: ${PRED_PLAIN}" >&2
  exit 1
fi
if [[ ! -f "${PRED_MONAI}" ]]; then
  echo "Missing MONAI OOS file: ${PRED_MONAI}" >&2
  exit 1
fi

echo ""
echo "========== Ensemble (same split join) =========="
FILTER_ARGS=()
if [[ -n "${FILTER_MODEL_PLAIN}" ]]; then
  FILTER_ARGS+=(--filter_model_plain "${FILTER_MODEL_PLAIN}")
fi
if [[ -n "${FILTER_MODEL_MONAI}" ]]; then
  FILTER_ARGS+=(--filter_model_monai "${FILTER_MODEL_MONAI}")
fi

"${PY}" "${SCRIPT_DIR}/ensemble_oos_same_split_eval.py" \
  --csv_plain "${PRED_PLAIN}" \
  --csv_monai "${PRED_MONAI}" \
  "${FILTER_ARGS[@]}" \
  --require_split_protocol_match \
  --out_dir "${ENSEMBLE_OUT}" \
  --run_tag "${RUN_TAG}"

echo ""
echo "OOS predictions:"
echo "  ${PRED_PLAIN}"
echo "  ${PRED_MONAI}"
echo "Ensemble summary: ${ENSEMBLE_OUT}/run_${RUN_TAG}/summary_methods.csv"
