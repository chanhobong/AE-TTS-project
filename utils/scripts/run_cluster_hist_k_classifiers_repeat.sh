#!/usr/bin/env bash
# Cluster-histogram pooling: sweep K in 16 32 64 128; for each K run StratifiedShuffleSplit × N
# with three classifiers in one pass: logistic, linear_svc, rbf_svc (--classifier all).
# Same random splits for all three models per K (see repeated_stratified_shuffle_eval.py).
#
# Prerequisites: NPZ with cluster_ids (e.g. monai_ae/patients or plain_ae_slice_meta/patients).
#
# Usage:
#   LATENT_SOURCE=monai_ae bash utils/scripts/run_cluster_hist_k_classifiers_repeat.sh
#   NPZ_DIR=/path/to/patients K_LIST="16 32 64 128" N_SPLITS=100 \
#     OUT_DIR=/path/to/out RUN_TAG_BASE=myexp bash utils/scripts/run_cluster_hist_k_classifiers_repeat.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
LOGISTIC_C="${LOGISTIC_C:-1.0}"
SVC_C="${SVC_C:-1.0}"
SVC_GAMMA="${SVC_GAMMA:-scale}"
PY="${PYTHON:-python3}"
EVAL_PY="${SCRIPT_DIR}/repeated_stratified_shuffle_eval.py"

LATENT_SOURCE="${LATENT_SOURCE:-monai_ae}"
K_LIST="${K_LIST:-16 32 64 128}"
RUN_TAG_BASE="${RUN_TAG_BASE:-$(date +%Y%m%d_%H%M%S)}"

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

resolve_npz_dir() {
  local src="$1"
  case "${src}" in
    monai_ae)
      local p sm
      p="${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients"
      sm="${LATENT_VOL}/diff3dformer_stageB_monai_ae_slice_meta/patients"
      if [[ -d "${sm}" ]]; then echo "${sm}"; else echo "${p}"; fi
      ;;
    plain_ae)
      local p sm
      p="${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
      sm="${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients"
      if [[ -d "${sm}" ]]; then echo "${sm}"; else echo "${p}"; fi
      ;;
    *)
      echo "Unknown LATENT_SOURCE=${src} (use monai_ae or plain_ae, or set NPZ_DIR)" >&2
      exit 1
      ;;
  esac
}

if [[ -n "${NPZ_DIR:-}" ]]; then
  _NPZ="${NPZ_DIR}"
else
  _NPZ="$(resolve_npz_dir "${LATENT_SOURCE}")"
fi

if [[ ! -d "${_NPZ}" ]]; then
  echo "NPZ dir not found: ${_NPZ}" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-${LATENT_VOL}/out_cluster_hist_K_classifiers_repeat}"

echo "LATENT_SOURCE=${LATENT_SOURCE}  NPZ_DIR=${_NPZ}"
echo "OUT_DIR=${OUT_DIR}  K_LIST=${K_LIST}  n_splits=${N_SPLITS}  classifier=all"

for K in ${K_LIST}; do
  TAG="${RUN_TAG_BASE}_K${K}"
  echo ""
  echo "========== cluster_hist_k=${K}  run_${TAG} =========="
  EXTRA_EVAL_ARGS=()
  if [[ "${WRITE_TEST_PRED:-0}" == "1" || "${WRITE_TEST_PRED:-0}" == "true" ]]; then
    EXTRA_EVAL_ARGS+=(--write_test_predictions)
  fi
  if [[ "${CONCAT_AGE_SEX:-0}" == "1" || "${CONCAT_AGE_SEX:-0}" == "true" ]]; then
    EXTRA_EVAL_ARGS+=(--concat_age_sex)
  fi
  if [[ "${FINAL_RESCALE_AFTER_CONCAT:-0}" == "1" || "${FINAL_RESCALE_AFTER_CONCAT:-0}" == "true" ]]; then
    EXTRA_EVAL_ARGS+=(--final_rescale_after_concat)
  fi
  EXTRA_EVAL_ARGS+=(--svc_C "${SVC_C}" --svc_gamma "${SVC_GAMMA}")
  "${PY}" "${EVAL_PY}" \
    --labels_csv "${SPLIT_DIR}/train.csv" \
    --labels_csv "${SPLIT_DIR}/val.csv" \
    --labels_csv "${SPLIT_DIR}/test.csv" \
    --latent_source "${LATENT_SOURCE}" "${_NPZ}" \
    --pooling cluster_hist \
    --cluster_hist_k "${K}" \
    --classifier all \
    --logistic_C "${LOGISTIC_C}" \
    --logistic_penalty l2 \
    --logistic_solver lbfgs \
    --n_splits "${N_SPLITS}" \
    --test_size "${TEST_SIZE}" \
    --random_state "${RANDOM_STATE}" \
    --out_dir "${OUT_DIR}" \
    --run_tag "${TAG}" \
    "${EXTRA_EVAL_ARGS[@]}"
done

echo ""
echo "Done. Each K: ${OUT_DIR}/run_${RUN_TAG_BASE}_K<K>/${LATENT_SOURCE//\//_}/cluster_hist/all/summary.csv"
