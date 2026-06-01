#!/usr/bin/env bash
# Thesis canonical pipeline: clinical_2 / RUN_TAG=mytag
#
# Plain:  p90_p10_vol_mm + Elastic-Net LR (logistic_en_C0.1_r0.2) via repeated_head_sweep_std
# MONAI:  cluster_hist + RBF-SVC on repeat_2 NPZ layout
# Fusion: fixed w=0.5 late ensemble + optional age/sex LR stack
#
# Example (full rerun on local latent store):
#   export LATENT_VOL=/path/to/latent_data
#   export RUN_TAG=mytag
#   bash stage_c/run/run_paired_ensemble_mytag.sh
#
# Example (ensemble only — OOS CSVs already on disk):
#   SKIP_TOWER_RERUN=1 bash stage_c/run/run_paired_ensemble_mytag.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-mytag}"
LATENT_VOL="${LATENT_VOL:-/Volumes/Chanho_PhD_Project/latent_data}"
if [[ ! -d "${LATENT_VOL}" ]]; then
  LATENT_VOL="${REPO_ROOT}/latent_data"
fi

if [[ -f "${REPO_ROOT}/data/splits/train.csv" ]]; then
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits}"
else
  SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
fi

N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
PY="${PYTHON:-python3}"
EVAL_DIR="${SCRIPT_DIR}/../eval"
UTILS_DIR="${REPO_ROOT}/utils/scripts"

PLAIN_POOLING="${PLAIN_POOLING:-p90_p10_vol_mm}"
PLAIN_HEAD_ID="${PLAIN_HEAD_ID:-logistic_en_C0.1_r0.2}"
MONAI_POOLING="${MONAI_POOLING:-cluster_hist}"
CLASSIFIER_MONAI="${CLASSIFIER_MONAI:-rbf_svc}"
SVC_C="${SVC_C:-1.0}"
SVC_GAMMA="${SVC_GAMMA:-scale}"

OUT_PLAIN="${OUT_DIR_PLAIN:-${LATENT_VOL}/out_plain_p90_p10_vol_mm_head_repeat_combo_plain_slice_meta_ae2}"
OUT_MONAI="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_cluster_hist_rbf_repeat_2}"
ENSEMBLE_OUT="${ENSEMBLE_OUT:-${LATENT_VOL}/ensemble_fix_mytag_clinical_2}"

PLAIN_NPZ="${PLAIN_NPZ_DIR:-${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients}"
MONAI_NPZ="${MONAI_NPZ_DIR:-${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients}"

SKIP_TOWER_RERUN="${SKIP_TOWER_RERUN:-0}"

echo "Thesis canonical (clinical_2)"
echo "  RUN_TAG=${RUN_TAG}"
echo "  Plain:  ${PLAIN_POOLING} / ${PLAIN_HEAD_ID}"
echo "  MONAI:  ${MONAI_POOLING} / ${CLASSIFIER_MONAI}"
echo "  OUT:    ${ENSEMBLE_OUT}/run_${RUN_TAG}/summary_methods.csv"

if [[ "${SKIP_TOWER_RERUN}" != "1" ]]; then
  [[ -d "${PLAIN_NPZ}" ]] || { echo "Missing Plain NPZ: ${PLAIN_NPZ} (set PLAIN_NPZ_DIR=)" >&2; exit 1; }
  [[ -d "${MONAI_NPZ}" ]] || { echo "Missing MONAI NPZ: ${MONAI_NPZ} (set MONAI_NPZ_DIR=)" >&2; exit 1; }

  echo ""
  echo "========== Plain (trajectory head sweep) =========="
  PYTHONPATH="${EVAL_DIR}:${UTILS_DIR}" "${PY}" "${UTILS_DIR}/repeated_head_sweep_std.py" \
    --labels_csv "${SPLIT_DIR}/train.csv" \
    --labels_csv "${SPLIT_DIR}/val.csv" \
    --labels_csv "${SPLIT_DIR}/test.csv" \
    --latent_source plain_ae "${PLAIN_NPZ}" \
    --pooling "${PLAIN_POOLING}" \
    --only_head_id "${PLAIN_HEAD_ID}" \
    --write_test_predictions \
    --n_splits "${N_SPLITS}" \
    --test_size "${TEST_SIZE}" \
    --random_state "${RANDOM_STATE}" \
    --n_jobs "${N_JOBS:-1}" \
    --out_dir "${OUT_PLAIN}" \
    --run_tag "${RUN_TAG}"

  echo ""
  echo "========== MONAI (cluster_hist + RBF) =========="
  WRITE_TEST_PRED=1 \
    NPZ_DIR="${MONAI_NPZ}" \
    OUT_DIR="${OUT_MONAI}" \
    POOLING="${MONAI_POOLING}" \
    CLASSIFIER="${CLASSIFIER_MONAI}" \
    RUN_TAG="${RUN_TAG}" \
    N_SPLITS="${N_SPLITS}" TEST_SIZE="${TEST_SIZE}" RANDOM_STATE="${RANDOM_STATE}" \
    SPLIT_DIR="${SPLIT_DIR}" \
    SVC_C="${SVC_C}" SVC_GAMMA="${SVC_GAMMA}" \
    CLUSTER_HIST_K="${CLUSTER_HIST_K:-}" \
    bash "${SCRIPT_DIR}/run_monai_repeat.sh"
fi

PRED_PLAIN="${OUT_PLAIN}/run_${RUN_TAG}/plain_ae/${PLAIN_POOLING}/${PLAIN_HEAD_ID}/test_oos_predictions.csv"
PRED_MONAI="${OUT_MONAI}/run_${RUN_TAG}/monai_ae/${MONAI_POOLING}/${CLASSIFIER_MONAI}/test_oos_predictions.csv"

for f in "${PRED_PLAIN}" "${PRED_MONAI}"; do
  [[ -f "${f}" ]] || { echo "Missing OOS predictions: ${f}" >&2; exit 1; }
done

echo ""
echo "========== Ensemble (clinical_2) =========="
CLIN_ARGS=()
if [[ -f "${SPLIT_DIR}/train.csv" ]]; then
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/train.csv")
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/val.csv")
  CLIN_ARGS+=(--clinical_labels_csv "${SPLIT_DIR}/test.csv")
fi

"${PY}" "${EVAL_DIR}/ensemble_oos_same_split_eval.py" \
  --csv_plain "${PRED_PLAIN}" \
  --csv_monai "${PRED_MONAI}" \
  --filter_model_plain "${PLAIN_HEAD_ID}" \
  --filter_model_monai "${CLASSIFIER_MONAI}" \
  --require_split_protocol_match \
  --split_driver_protocol_only \
  "${CLIN_ARGS[@]}" \
  --out_dir "${ENSEMBLE_OUT}" \
  --run_tag "${RUN_TAG}"

echo ""
echo "Summary: ${ENSEMBLE_OUT}/run_${RUN_TAG}/summary_methods.csv"
echo "Frozen reference (no NPZ required): results/main_summary_methods.csv"
