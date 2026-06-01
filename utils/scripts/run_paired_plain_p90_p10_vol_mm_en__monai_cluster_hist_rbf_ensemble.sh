#!/usr/bin/env bash
# Plain AE: trajectory pooling ``p90_p10_vol_mm`` + Elastic-Net logistic (C=0.1, l1_ratio=0.2)
#   via repeated_head_sweep_std.py (--only_head_id logistic_en_C0.1_r0.2).
# MONAI AE: **cluster_hist** pooling + RBF SVC (C=1, gamma=scale by default) × N splits
#   via repeated_stratified_shuffle_eval.py.
# Then ensemble_oos_same_split_eval (same repeat_id + patient_id join).
#
# CLUSTER_HIST_K: set to pin histogram length (e.g. 64). If unset, omitted → infer from NPZ.
#
# Same RUN_TAG, --n_splits / --test_size / --random_state / labels CSVs keep StratifiedShuffleSplit aligned.
#
# IMPORTANT: Both feature pipelines must yield the **same patient count and row order** with the **same y**.
#
# Example:
#   LATENT_VOL=/Volumes/.../latent_data RUN_TAG=exp_p90_ch_001 CLUSTER_HIST_K=64 \\
#     bash utils/scripts/run_paired_plain_p90_p10_vol_mm_en__monai_cluster_hist_rbf_ensemble.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:?Set RUN_TAG (e.g. exp_p90_ch_001)}"
LATENT_VOL="${LATENT_VOL:-${REPO_ROOT}/latent_data}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
PY="${PYTHON:-python3}"

MONAI_POOLING="${MONAI_POOLING:-cluster_hist}"

PLAIN_HEAD_ID="${PLAIN_HEAD_ID:-logistic_en_C0.1_r0.2}"
PLAIN_POOLING="${PLAIN_POOLING:-p90_p10_vol_mm}"

HEAD_OUT="${OUT_DIR_PLAIN_HEAD:-${LATENT_VOL}/out_plain_${PLAIN_POOLING}_head_repeat}"
MONAI_OUT="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_cluster_hist_rbf_repeat}"

MONAI_EXTRA=()
if [[ -n "${CLUSTER_HIST_K:-}" ]]; then
  MONAI_EXTRA+=(--cluster_hist_k "${CLUSTER_HIST_K}")
fi

# Plain NPZ (trajectory features; prefers slice_meta patients like other scripts)
PLAIN_P="${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
PLAIN_SM="${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients"
PLAIN_NPZ="${PLAIN_NPZ_DIR:-}"
if [[ -z "${PLAIN_NPZ}" ]]; then
  if [[ -d "${PLAIN_SM}" ]]; then
    PLAIN_NPZ="${PLAIN_SM}"
    echo "[npz] Plain: slice_meta/patients"
  else
    PLAIN_NPZ="${PLAIN_P}"
    echo "[npz] Plain: plain_ae/patients"
  fi
fi

MONAI_P="${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients"
MONAI_SM="${LATENT_VOL}/diff3dformer_stageB_monai_ae_slice_meta/patients"
MONAI_NPZ="${MONAI_NPZ_DIR:-}"
if [[ -z "${MONAI_NPZ}" ]]; then
  if [[ -d "${MONAI_SM}" ]]; then
    MONAI_NPZ="${MONAI_SM}"
    echo "[npz] MONAI: slice_meta/patients"
  else
    MONAI_NPZ="${MONAI_P}"
    echo "[npz] MONAI: monai_ae/patients"
  fi
fi

[[ -d "${PLAIN_NPZ}" ]] || { echo "Missing Plain NPZ dir: ${PLAIN_NPZ}" >&2; exit 1; }
[[ -d "${MONAI_NPZ}" ]] || { echo "Missing MONAI NPZ dir: ${MONAI_NPZ}" >&2; exit 1; }

echo "RUN_TAG=${RUN_TAG}  n_splits=${N_SPLITS} test_size=${TEST_SIZE} random_state=${RANDOM_STATE}"
echo "Plain head: ${PLAIN_POOLING} / ${PLAIN_HEAD_ID}"
echo "MONAI: ${MONAI_POOLING} / rbf_svc  C=${SVC_C:-1.0}  gamma=${SVC_GAMMA:-scale}  CLUSTER_HIST_K=${CLUSTER_HIST_K:-<infer>}"

echo ""
echo "========== Plain AE (head sweep, single head + OOS export) =========="
"${PY}" "${SCRIPT_DIR}/repeated_head_sweep_std.py" \
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
  --out_dir "${HEAD_OUT}" \
  --run_tag "${RUN_TAG}"

echo ""
echo "========== MONAI AE (cluster_hist + RBF, OOS export) =========="
"${PY}" "${SCRIPT_DIR}/repeated_stratified_shuffle_eval.py" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --latent_source monai_ae "${MONAI_NPZ}" \
  --pooling "${MONAI_POOLING}" \
  "${MONAI_EXTRA[@]}" \
  --classifier rbf_svc \
  --svc_C "${SVC_C:-1.0}" \
  --svc_gamma "${SVC_GAMMA:-scale}" \
  --n_splits "${N_SPLITS}" \
  --test_size "${TEST_SIZE}" \
  --random_state "${RANDOM_STATE}" \
  --write_test_predictions \
  --out_dir "${MONAI_OUT}" \
  --run_tag "${RUN_TAG}"

PRED_PLAIN="${HEAD_OUT}/run_${RUN_TAG}/plain_ae/${PLAIN_POOLING}/${PLAIN_HEAD_ID}/test_oos_predictions.csv"
PRED_MONAI="${MONAI_OUT}/run_${RUN_TAG}/monai_ae/${MONAI_POOLING}/rbf_svc/test_oos_predictions.csv"

if [[ ! -f "${PRED_PLAIN}" ]]; then
  echo "Missing Plain OOS: ${PRED_PLAIN}" >&2
  exit 1
fi
if [[ ! -f "${PRED_MONAI}" ]]; then
  echo "Missing MONAI OOS: ${PRED_MONAI}" >&2
  exit 1
fi

ENSEMBLE_OUT="${ENSEMBLE_OUT:-${LATENT_VOL}/ensemble_plain_${PLAIN_POOLING}__monai_${MONAI_POOLING}_rbf_${RUN_TAG}}"

echo ""
echo "========== Ensemble =========="
"${PY}" "${SCRIPT_DIR}/ensemble_oos_same_split_eval.py" \
  --csv_plain "${PRED_PLAIN}" \
  --csv_monai "${PRED_MONAI}" \
  --filter_model_plain "${PLAIN_HEAD_ID}" \
  --filter_model_monai rbf_svc \
  --require_split_protocol_match \
  --split_driver_protocol_only \
  --out_dir "${ENSEMBLE_OUT}" \
  --run_tag "${RUN_TAG}"

echo ""
echo "Plain OOS:  ${PRED_PLAIN}"
echo "MONAI OOS:  ${PRED_MONAI}"
echo "Summary:    ${ENSEMBLE_OUT}/run_${RUN_TAG}/summary_methods.csv"
