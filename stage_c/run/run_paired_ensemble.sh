#!/usr/bin/env bash
# Plain + MONAI repeated eval (same RUN_TAG) → OOS CSVs → late ensemble + optional clinical stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:?Set RUN_TAG (e.g. RUN_TAG=thesis_v1)}"
LATENT_VOL="${LATENT_VOL:-/Volumes/Chanho_PhD_Project/latent_data}"
if [[ ! -d "${LATENT_VOL}" ]]; then
  LATENT_VOL="${REPO_ROOT}/latent_data"
fi

if [[ -f "${REPO_ROOT}/data/splits/train.csv" ]]; then
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits}"
else
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
fi

POOLING="${POOLING:-std}"
POOLING_MONAI="${POOLING_MONAI:-cluster_hist}"
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

OUT_PLAIN="${OUT_DIR_PLAIN:-${LATENT_VOL}/out_plain_ae_${POOLING}_repeat}"
OUT_MONAI="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_${POOLING_MONAI}_repeat}"
ENSEMBLE_OUT="${ENSEMBLE_OUT:-${LATENT_VOL}/ensemble_${RUN_TAG}}"
PY="${PYTHON:-python3}"
EVAL_DIR="${SCRIPT_DIR}/../eval"

export LATENT_VOL

echo "RUN_TAG=${RUN_TAG}  plain=${CLASSIFIER_PLAIN}/${POOLING}  monai=${CLASSIFIER_MONAI}/${POOLING_MONAI}"

echo ""
echo "========== Plain =========="
WRITE_TEST_PRED=1 \
  CLASSIFIER="${CLASSIFIER_PLAIN}" \
  POOLING="${POOLING}" \
  RUN_TAG="${RUN_TAG}" \
  N_SPLITS="${N_SPLITS}" TEST_SIZE="${TEST_SIZE}" RANDOM_STATE="${RANDOM_STATE}" \
  SPLIT_DIR="${SPLIT_DIR}" OUT_DIR="${OUT_PLAIN}" \
  LOGISTIC_C="${LOGISTIC_C}" SVC_C="${SVC_C}" SVC_GAMMA="${SVC_GAMMA}" \
  bash "${SCRIPT_DIR}/run_plain_repeat.sh"

echo ""
echo "========== MONAI =========="
WRITE_TEST_PRED=1 \
  CLASSIFIER="${CLASSIFIER_MONAI}" \
  POOLING="${POOLING_MONAI}" \
  RUN_TAG="${RUN_TAG}" \
  N_SPLITS="${N_SPLITS}" TEST_SIZE="${TEST_SIZE}" RANDOM_STATE="${RANDOM_STATE}" \
  SPLIT_DIR="${SPLIT_DIR}" OUT_DIR="${OUT_MONAI}" \
  LOGISTIC_C="${LOGISTIC_C}" SVC_C="${SVC_C}" SVC_GAMMA="${SVC_GAMMA}" \
  bash "${SCRIPT_DIR}/run_monai_repeat.sh"

PRED_PLAIN="${OUT_PLAIN}/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER_PLAIN}/test_oos_predictions.csv"
PRED_MONAI="${OUT_MONAI}/run_${RUN_TAG}/monai_ae/${POOLING_MONAI}/${CLASSIFIER_MONAI}/test_oos_predictions.csv"

for f in "${PRED_PLAIN}" "${PRED_MONAI}"; do
  [[ -f "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done

echo ""
echo "========== Ensemble =========="
FILTER_ARGS=()
[[ -n "${FILTER_MODEL_PLAIN}" ]] && FILTER_ARGS+=(--filter_model_plain "${FILTER_MODEL_PLAIN}")
[[ -n "${FILTER_MODEL_MONAI}" ]] && FILTER_ARGS+=(--filter_model_monai "${FILTER_MODEL_MONAI}")

CLIN_ARGS=()
if [[ -f "${SPLIT_DIR}/train.csv" ]]; then
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/train.csv")
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/val.csv")
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/test.csv")
fi

"${PY}" "${EVAL_DIR}/ensemble_oos_same_split_eval.py" \
  --csv_plain "${PRED_PLAIN}" \
  --csv_monai "${PRED_MONAI}" \
  "${FILTER_ARGS[@]}" \
  --require_split_protocol_match \
  "${CLIN_ARGS[@]}" \
  --out_dir "${ENSEMBLE_OUT}" \
  --run_tag "${RUN_TAG}"

echo ""
echo "Summary: ${ENSEMBLE_OUT}/run_${RUN_TAG}/summary_methods.csv"
