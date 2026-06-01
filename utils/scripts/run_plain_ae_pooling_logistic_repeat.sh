#!/usr/bin/env bash
# Plain AE patient-level features + LogisticRegression, StratifiedShuffleSplit × N (default 100).
# Uses repeated_stratified_shuffle_eval.py (--pooling mean|std|mean_std|cluster_hist).
# cluster_hist: NPZ must include cluster_ids (mask-aware if mask present). Prefers
#   diff3dformer_stageB_plain_ae_slice_meta/patients when that directory exists.
#
# Outputs (under OUT_DIR/run_<RUN_TAG>/):
#   plain_ae/<POOLING>/<CLASSIFIER>/repeats.csv
#   plain_ae/<POOLING>/<CLASSIFIER>/summary.csv   <- roc_mean, roc_std, roc_p2p5, pr_*, ...
#   plain_ae/<POOLING>/<CLASSIFIER>/hist_roc_auc_<CLASSIFIER>.png
#   plain_ae/<POOLING>/<CLASSIFIER>/test_oos_predictions.csv   (if WRITE_TEST_PRED=1)
#
# Env highlights:
#   WRITE_TEST_PRED=1 — export long OOS table for ensemble_oos_same_split_eval.py
#   CLASSIFIER=logistic|linear_svc|rbf_svc|all   (default logistic)
#   CLUSTER_HIST_K=<int> — passed to --cluster_hist_k when set (cluster_hist pooling)
#   CONCAT_AGE_SEX=1, FINAL_RESCALE_AFTER_CONCAT=1 — match MONAI/Plain protocol
#   SVC_C, SVC_GAMMA — when using rbf_svc or all
#
# Usage:
#   LATENT_VOL=/Volumes/.../latent_data bash utils/scripts/run_plain_ae_pooling_logistic_repeat.sh
#   POOLING=std N_SPLITS=100 bash utils/scripts/run_plain_ae_pooling_logistic_repeat.sh
#   WRITE_TEST_PRED=1 RUN_TAG=my_pair_001 POOLING=std bash utils/scripts/run_plain_ae_pooling_logistic_repeat.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${LATENT_VOL:-}" ]]; then
  _L_REPO="${REPO_ROOT}/latent_data"
  _L_VOL="/Volumes/Chanho_PhD_Project/latent_data"
  if [[ -d "${_L_REPO}/diff3dformer_stageB_plain_ae/patients" ]]; then
    LATENT_VOL="${_L_REPO}"
  elif [[ -d "${_L_VOL}/diff3dformer_stageB_plain_ae/patients" ]]; then
    LATENT_VOL="${_L_VOL}"
  else
    LATENT_VOL="${_L_REPO}"
  fi
fi

PLAIN_P="${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
PLAIN_SM="${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients"
NPZ_DIR="${NPZ_DIR:-}"
if [[ -z "${NPZ_DIR}" ]]; then
  if [[ -d "${PLAIN_SM}" ]]; then
    NPZ_DIR="${PLAIN_SM}"
    echo "[npz] using plain_ae_slice_meta/patients"
  else
    NPZ_DIR="${PLAIN_P}"
    echo "[npz] using plain_ae/patients"
  fi
fi

if [[ ! -d "${NPZ_DIR}" ]]; then
  echo "Missing NPZ dir: ${NPZ_DIR}" >&2
  exit 1
fi

SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
POOLING="${POOLING:-mean}"
N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
LOGISTIC_C="${LOGISTIC_C:-1.0}"
CLASSIFIER="${CLASSIFIER:-logistic}"
SVC_C="${SVC_C:-1.0}"
SVC_GAMMA="${SVC_GAMMA:-scale}"
OUT_DIR="${OUT_DIR:-${LATENT_VOL}/out_plain_ae_${POOLING}_logistic_repeat}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON:-python3}"
EVAL_PY="${SCRIPT_DIR}/repeated_stratified_shuffle_eval.py"

echo "NPZ_DIR=${NPZ_DIR}"
echo "OUT_DIR=${OUT_DIR}  run_${RUN_TAG}"
echo "POOLING=${POOLING}  n_splits=${N_SPLITS}  classifier=${CLASSIFIER}  C=${LOGISTIC_C}"

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
if [[ -n "${CLUSTER_HIST_K:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cluster_hist_k "${CLUSTER_HIST_K}")
fi
if [[ "${CLASSIFIER}" == "rbf_svc" || "${CLASSIFIER}" == "all" ]]; then
  EXTRA_EVAL_ARGS+=(--svc_C "${SVC_C}" --svc_gamma "${SVC_GAMMA}")
fi

"${PY}" "${EVAL_PY}" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --latent_source plain_ae "${NPZ_DIR}" \
  --pooling "${POOLING}" \
  --classifier "${CLASSIFIER}" \
  --logistic_C "${LOGISTIC_C}" \
  --logistic_penalty l2 \
  --logistic_solver lbfgs \
  --n_splits "${N_SPLITS}" \
  --test_size "${TEST_SIZE}" \
  --random_state "${RANDOM_STATE}" \
  --out_dir "${OUT_DIR}" \
  --run_tag "${RUN_TAG}" \
  "${EXTRA_EVAL_ARGS[@]}"

SUM="${OUT_DIR}/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER}/summary.csv"
echo ""
echo "Wrote: ${SUM}"
if [[ -f "${SUM}" ]]; then
  column -t -s, "${SUM}" 2>/dev/null || cat "${SUM}"
fi
