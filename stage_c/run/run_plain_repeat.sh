#!/usr/bin/env bash
# Plain AE NPZ → patient pooling → repeated StratifiedShuffleSplit eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${LATENT_VOL:-}" ]]; then
  _L_VOL="/Volumes/Chanho_PhD_Project/latent_data"
  LATENT_VOL="${LATENT_VOL:-$([[ -d "${_L_VOL}" ]] && echo "${_L_VOL}" || echo "${REPO_ROOT}/latent_data")}"
fi

# Prefer spatial v2 Stage B output names; fall back to legacy diff3dformer paths.
PLAIN_CANDIDATES=(
  "${NPZ_DIR:-}"
  "${LATENT_VOL}/stage_b/plain_ae/patients"
  "${LATENT_VOL}/diff3dformer_stageB_spatial_v2/plain_ae/patients"
  "${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients"
  "${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
)
NPZ_DIR=""
for c in "${PLAIN_CANDIDATES[@]}"; do
  [[ -n "${c}" && -d "${c}" ]] && NPZ_DIR="${c}" && break
done
if [[ -z "${NPZ_DIR}" ]]; then
  echo "Missing Plain NPZ dir. Set NPZ_DIR= or LATENT_VOL=." >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/data/splits/train.csv" ]]; then
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits}"
else
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
fi

POOLING="${POOLING:-std}"
N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
LOGISTIC_C="${LOGISTIC_C:-1.0}"
CLASSIFIER="${CLASSIFIER:-logistic}"
SVC_C="${SVC_C:-1.0}"
SVC_GAMMA="${SVC_GAMMA:-scale}"
OUT_DIR="${OUT_DIR:-${LATENT_VOL}/out_plain_ae_${POOLING}_repeat}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON:-python3}"
EVAL_PY="${SCRIPT_DIR}/../eval/repeated_stratified_shuffle_eval.py"

echo "NPZ_DIR=${NPZ_DIR}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "OUT_DIR=${OUT_DIR}/run_${RUN_TAG}"

EXTRA=()
[[ "${WRITE_TEST_PRED:-0}" == "1" ]] && EXTRA+=(--write_test_predictions)
[[ "${CONCAT_AGE_SEX:-0}" == "1" ]] && EXTRA+=(--concat_age_sex)
[[ "${FINAL_RESCALE_AFTER_CONCAT:-0}" == "1" ]] && EXTRA+=(--final_rescale_after_concat)
[[ -n "${CLUSTER_HIST_K:-}" ]] && EXTRA+=(--cluster_hist_k "${CLUSTER_HIST_K}")
[[ "${CLASSIFIER}" == "rbf_svc" || "${CLASSIFIER}" == "all" ]] && EXTRA+=(--svc_C "${SVC_C}" --svc_gamma "${SVC_GAMMA}")

"${PY}" "${EVAL_PY}" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --latent_source plain_ae "${NPZ_DIR}" \
  --pooling "${POOLING}" \
  --classifier "${CLASSIFIER}" \
  --logistic_C "${LOGISTIC_C}" \
  --n_splits "${N_SPLITS}" \
  --test_size "${TEST_SIZE}" \
  --random_state "${RANDOM_STATE}" \
  --out_dir "${OUT_DIR}" \
  --run_tag "${RUN_TAG}" \
  "${EXTRA[@]+"${EXTRA[@]}"}"

SUM="${OUT_DIR}/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER}/summary.csv"
echo "Wrote: ${SUM}"
[[ -f "${SUM}" ]] && column -t -s, "${SUM}" 2>/dev/null || cat "${SUM}"
