#!/usr/bin/env bash
# Clinical-only (age+sex LR) vs clinical + fixed soft-ensemble score on the same splits
# (clinical_vs_clinical_plus_ensemble_repeat.py).
#
# Expects existing Plain + MONAI ``test_oos_predictions.csv`` from a paired run (same RUN_TAG,
# n_splits, test_size, random_state). Paths are inferred from PAIRING mode unless overridden.
#
# Pairing modes (set PAIRING):
#   p90_ch   — Plain ``p90_p10_vol_mm`` + ``logistic_en_C0.1_r0.2`` (repeated_head_sweep_std layout)
#              and MONAI ``cluster_hist`` + ``rbf_svc`` (default; matches
#              run_paired_plain_p90_p10_vol_mm_en__monai_cluster_hist_rbf_ensemble.sh)
#   pooling  — Plain/MONAI both ``repeated_stratified_shuffle_eval`` under plain_ae/monai_ae/
#              (default POOLING=std, CLASSIFIER_*; matches run_paired_plain_monai_oos_then_ensemble.sh)
#
# Or bypass inference:
#   PRED_PLAIN=/path/to/test_oos_predictions.csv PRED_MONAI=... bash this_script.sh
#
# If default paths are wrong but files exist under LATENT_VOL or repo ``latent_data``:
#   AUTO_DISCOVER_OOS=1 bash ...
#
# Example:
#   LATENT_VOL=/Volumes/Chanho_PhD_Project/latent_data RUN_TAG=exp_p90_ch_001 PAIRING=p90_ch \\
#     bash utils/scripts/run_clinical_vs_clinical_plus_ensemble_repeat.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:?Set RUN_TAG (must match the paired OOS export run)}"
LATENT_VOL="${LATENT_VOL:-${REPO_ROOT}/latent_data}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
PAIRING="${PAIRING:-p90_ch}"

N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
FIXED_W="${FIXED_W:-0.5}"

PY="${PYTHON:-python3}"

if [[ ! -f "${SPLIT_DIR}/train.csv" || ! -f "${SPLIT_DIR}/val.csv" || ! -f "${SPLIT_DIR}/test.csv" ]]; then
  echo "Expected ${SPLIT_DIR}/{train,val,test}.csv" >&2
  exit 1
fi

if [[ -z "${PRED_PLAIN:-}" || -z "${PRED_MONAI:-}" ]]; then
  case "${PAIRING}" in
    p90_ch)
      PLAIN_POOLING="${PLAIN_POOLING:-p90_p10_vol_mm}"
      PLAIN_HEAD_ID="${PLAIN_HEAD_ID:-logistic_en_C0.1_r0.2}"
      MONAI_POOLING="${MONAI_POOLING:-cluster_hist}"
      HEAD_OUT="${OUT_DIR_PLAIN_HEAD:-${LATENT_VOL}/out_plain_${PLAIN_POOLING}_head_repeat}"
      MONAI_OUT="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_cluster_hist_rbf_repeat}"
      PRED_PLAIN="${HEAD_OUT}/run_${RUN_TAG}/plain_ae/${PLAIN_POOLING}/${PLAIN_HEAD_ID}/test_oos_predictions.csv"
      PRED_MONAI="${MONAI_OUT}/run_${RUN_TAG}/monai_ae/${MONAI_POOLING}/rbf_svc/test_oos_predictions.csv"
      FILTER_PLAIN="${FILTER_MODEL_PLAIN:-${PLAIN_HEAD_ID}}"
      FILTER_MONAI="${FILTER_MODEL_MONAI:-rbf_svc}"
      ;;
    pooling)
      POOLING="${POOLING:-std}"
      CLASSIFIER_PLAIN="${CLASSIFIER_PLAIN:-logistic}"
      CLASSIFIER_MONAI="${CLASSIFIER_MONAI:-rbf_svc}"
      OUT_PLAIN="${OUT_DIR_PLAIN:-${LATENT_VOL}/out_plain_ae_${POOLING}_logistic_repeat}"
      OUT_MONAI="${OUT_DIR_MONAI:-${LATENT_VOL}/out_monai_ae_${POOLING}_logistic_repeat}"
      PRED_PLAIN="${OUT_PLAIN}/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER_PLAIN}/test_oos_predictions.csv"
      PRED_MONAI="${OUT_MONAI}/run_${RUN_TAG}/monai_ae/${POOLING}/${CLASSIFIER_MONAI}/test_oos_predictions.csv"
      FILTER_PLAIN="${FILTER_MODEL_PLAIN:-${CLASSIFIER_PLAIN}}"
      FILTER_MONAI="${FILTER_MODEL_MONAI:-${CLASSIFIER_MONAI}}"
      ;;
    *)
      echo "PAIRING must be p90_ch or pooling (got ${PAIRING})" >&2
      exit 1
      ;;
  esac
else
  FILTER_PLAIN="${FILTER_MODEL_PLAIN:-}"
  FILTER_MONAI="${FILTER_MODEL_MONAI:-}"
fi

_search_vol_roots() {
  if [[ -d "${LATENT_VOL}" ]]; then
    printf '%s\n' "$(cd "${LATENT_VOL}" && pwd)"
  fi
  if [[ -d "${REPO_ROOT}/latent_data" ]]; then
    local _repo
    _repo="$(cd "${REPO_ROOT}/latent_data" && pwd)"
    if [[ -d "${LATENT_VOL}" ]]; then
      local _lv
      _lv="$(cd "${LATENT_VOL}" && pwd)"
      if [[ "${_repo}" != "${_lv}" ]]; then
        printf '%s\n' "${_repo}"
      fi
    else
      printf '%s\n' "${_repo}"
    fi
  fi
}

_first_find_match() {
  # stdout: first path found, or empty
  local root="$1"
  local pattern="$2"
  find "${root}" -path "${pattern}" 2>/dev/null | head -n 1
}

_list_find_matches() {
  local root="$1"
  local pattern="$2"
  find "${root}" -path "${pattern}" 2>/dev/null | head -n 8
}

if [[ "${AUTO_DISCOVER_OOS:-0}" == "1" || "${AUTO_DISCOVER_OOS:-0}" == "true" ]]; then
  if [[ ! -f "${PRED_PLAIN}" ]]; then
    _pat=""
    if [[ "${PAIRING}" == "p90_ch" ]]; then
      _pat="*/run_${RUN_TAG}/plain_ae/${PLAIN_POOLING}/${PLAIN_HEAD_ID}/test_oos_predictions.csv"
    else
      _pat="*/run_${RUN_TAG}/plain_ae/${POOLING}/${CLASSIFIER_PLAIN}/test_oos_predictions.csv"
    fi
    while IFS= read -r _root; do
      [[ -z "${_root}" ]] && continue
      _hit="$(_first_find_match "${_root}" "${_pat}")"
      if [[ -n "${_hit}" && -f "${_hit}" ]]; then
        echo "[auto] Plain OOS: ${_hit}" >&2
        PRED_PLAIN="${_hit}"
        break
      fi
    done < <(_search_vol_roots)
  fi
  if [[ ! -f "${PRED_MONAI}" ]]; then
    if [[ "${PAIRING}" == "p90_ch" ]]; then
      _pat="*/run_${RUN_TAG}/monai_ae/${MONAI_POOLING}/rbf_svc/test_oos_predictions.csv"
    else
      _pat="*/run_${RUN_TAG}/monai_ae/${POOLING}/${CLASSIFIER_MONAI}/test_oos_predictions.csv"
    fi
    while IFS= read -r _root; do
      [[ -z "${_root}" ]] && continue
      _hit="$(_first_find_match "${_root}" "${_pat}")"
      if [[ -n "${_hit}" && -f "${_hit}" ]]; then
        echo "[auto] MONAI OOS: ${_hit}" >&2
        PRED_MONAI="${_hit}"
        break
      fi
    done < <(_search_vol_roots)
  fi
fi

if [[ ! -f "${PRED_PLAIN}" ]]; then
  echo "Missing Plain OOS: ${PRED_PLAIN}" >&2
  echo "  → Run the paired script with --write_test_predictions first, or set OUT_DIR_PLAIN_HEAD / PRED_PLAIN." >&2
  if [[ "${PAIRING}" == "p90_ch" ]]; then
    echo "  → Default parent dir is OUT_DIR_PLAIN_HEAD (same as HEAD_OUT in run_paired_plain_p90_p10_vol_mm_en__monai_cluster_hist_rbf_ensemble.sh)." >&2
    _pat="*/run_${RUN_TAG}/plain_ae/${PLAIN_POOLING}/${PLAIN_HEAD_ID}/test_oos_predictions.csv"
    echo "  Search pattern (first hits):" >&2
    while IFS= read -r _root; do
      [[ -z "${_root}" ]] && continue
      _list_find_matches "${_root}" "${_pat}" | while IFS= read -r _ln; do
        echo "    ${_ln}" >&2
      done
    done < <(_search_vol_roots)
    echo "  Retry with: AUTO_DISCOVER_OOS=1 ...  (pick correct RUN_TAG if several runs exist)" >&2
  fi
  exit 1
fi
if [[ ! -f "${PRED_MONAI}" ]]; then
  echo "Missing MONAI OOS: ${PRED_MONAI}" >&2
  echo "  → Set OUT_DIR_MONAI or PRED_MONAI; MONAI run must use --write_test_predictions." >&2
  if [[ "${PAIRING}" == "p90_ch" ]]; then
    _pat="*/run_${RUN_TAG}/monai_ae/${MONAI_POOLING}/rbf_svc/test_oos_predictions.csv"
    echo "  Search pattern (first hits):" >&2
    while IFS= read -r _root; do
      [[ -z "${_root}" ]] && continue
      _list_find_matches "${_root}" "${_pat}" | while IFS= read -r _ln; do
        echo "    ${_ln}" >&2
      done
    done < <(_search_vol_roots)
    echo "  Retry with: AUTO_DISCOVER_OOS=1 ..." >&2
  fi
  exit 1
fi

OUT_COMPARE="${CLINICAL_ENSEMBLE_OUT:-${LATENT_VOL}/clinical_vs_clinical_plus_ensemble_${PAIRING}_${RUN_TAG}}"
COMPARE_TAG="${CLINICAL_COMPARE_RUN_TAG:-${RUN_TAG}}"

FILTER_ARGS=()
if [[ -n "${FILTER_PLAIN}" ]]; then
  FILTER_ARGS+=(--filter_model_plain "${FILTER_PLAIN}")
fi
if [[ -n "${FILTER_MONAI}" ]]; then
  FILTER_ARGS+=(--filter_model_monai "${FILTER_MONAI}")
fi

PROB_ARGS=()
if [[ -n "${PROB_COL_PLAIN:-}" ]]; then
  PROB_ARGS+=(--prob_col_plain "${PROB_COL_PLAIN}")
fi
if [[ -n "${PROB_COL_MONAI:-}" ]]; then
  PROB_ARGS+=(--prob_col_monai "${PROB_COL_MONAI}")
fi

echo "PAIRING=${PAIRING}  RUN_TAG=${RUN_TAG}  fixed_w=${FIXED_W}"
echo "Plain:  ${PRED_PLAIN}"
echo "MONAI:  ${PRED_MONAI}"
echo "Out:    ${OUT_COMPARE}/run_${COMPARE_TAG}_clinical_vs_clinical_plus_ensemble/"

"${PY}" "${SCRIPT_DIR}/clinical_vs_clinical_plus_ensemble_repeat.py" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --csv_plain "${PRED_PLAIN}" \
  --csv_monai "${PRED_MONAI}" \
  ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} \
  ${PROB_ARGS[@]+"${PROB_ARGS[@]}"} \
  --fixed_w "${FIXED_W}" \
  --n_splits "${N_SPLITS}" \
  --test_size "${TEST_SIZE}" \
  --random_state "${RANDOM_STATE}" \
  --out_dir "${OUT_COMPARE}" \
  --run_tag "${COMPARE_TAG}"

echo ""
echo "Wrote repeats/summary under: ${OUT_COMPARE}/run_${COMPARE_TAG}_clinical_vs_clinical_plus_ensemble/"
