#!/usr/bin/env bash
# Repeated classifier head sweep — trajectory poolings (optional std).
#
# Default comparison: plain_ae vs monai_ae (slice_meta auto if present).
# Optional: plain_ae_VF (INCLUDE_PLAIN_VF=1).
#
# Baseline plain:     NPZ_MODE → .../diff3dformer_stageB_plain_ae_slice_meta/patients (or plain_ae/patients)
# MONAI:              same NPZ_MODE rule under .../diff3dformer_stageB_monai_ae(_slice_meta)/patients
# VF:                 PLAIN_VF_DIR (default .../plain_ae_VF_loss/patients)
#
# Default POOLINGS (override with POOLINGS="..."):
#   p90_p10, global_mean_vol_mm (= mean |dz|/dmm), p90_vol_mm, regional_vol_mm (basal|mid|apical concat),
#   p90_p10 + each of the above transition summaries.
#
# Usage:
#   bash utils/scripts/run_repeated_head_sweep_p90_volmm.sh
#   LATENT_VOL=/path/to/latent_data bash utils/scripts/run_repeated_head_sweep_p90_volmm.sh
#   INCLUDE_MONAI=0 bash utils/scripts/run_repeated_head_sweep_p90_volmm.sh
# Optional MONAI vs MONAI+VF only (no plain_ae):
#   MONAI_VS_MONAI_VF=1 LATENT_VOL=/path bash utils/scripts/run_repeated_head_sweep_p90_volmm.sh
#   Uses .../diff3dformer_stageB_monai_ae/patients and .../diff3dformer_stageB_monai_ae_VF_loss/patients
#   (override with MONAI_PATIENTS_DIR=... MONAI_VF_PATIENTS_DIR=...).
#   Default OUT_PARENT becomes .../repeated_head_sweep_monai unless OUT_PARENT is set.
#
# Resume only p90_p10_regional_vol_mm + merge:
#   RUN_TAG_BASE=... LATENT_VOL=... bash utils/scripts/run_p90_p10_regional_vol_mm_and_merge.sh
#
# COEF_ONLY=1 …   RUN_COEF_STABILITY=1 …   (see top of file history)
#
# Parallelism (CPU):
#   SWEEP_N_JOBS=-1     — repeated CV splits parallelized inside each Python sweep (joblib loky).
#   POOLING_PARALLEL=4  — run up to 4 different --pooling invocations at once (separate processes).
#   Avoid oversubscription: e.g. SWEEP_N_JOBS=-1 with POOLING_PARALLEL=1, or SWEEP_N_JOBS=1 with POOLING_PARALLEL=4.
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

SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
TRAIN_CSV="${SPLIT_DIR}/train.csv"
VAL_CSV="${SPLIT_DIR}/val.csv"
TEST_CSV="${SPLIT_DIR}/test.csv"
for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing split CSV: ${f}" >&2
    exit 1
  fi
done

PLAIN_P="${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
PLAIN_SM="${LATENT_VOL}/diff3dformer_stageB_plain_ae_slice_meta/patients"
MONAI_P="${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients"
MONAI_SM="${LATENT_VOL}/diff3dformer_stageB_monai_ae_slice_meta/patients"
PLAIN_VF_DIR="${PLAIN_VF_DIR:-${LATENT_VOL}/diff3dformer_stageB_plain_ae_VF_loss/patients}"
INCLUDE_PLAIN_VF="${INCLUDE_PLAIN_VF:-0}"
INCLUDE_MONAI="${INCLUDE_MONAI:-1}"
MONAI_VS_MONAI_VF="${MONAI_VS_MONAI_VF:-0}"
MONAI_PATIENTS_DIR="${MONAI_PATIENTS_DIR:-${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients}"
MONAI_VF_PATIENTS_DIR="${MONAI_VF_PATIENTS_DIR:-${LATENT_VOL}/diff3dformer_stageB_monai_ae_VF_loss/patients}"

NPZ_MODE="${NPZ_MODE:-auto}"
case "${NPZ_MODE}" in
  slice_meta)
    PLAIN_DIR="${PLAIN_SM}"
    MONAI_DIR="${MONAI_SM}"
    ;;
  patients)
    PLAIN_DIR="${PLAIN_P}"
    MONAI_DIR="${MONAI_P}"
    ;;
  auto)
    if [[ -d "${PLAIN_SM}" ]]; then
      PLAIN_DIR="${PLAIN_SM}"
      echo "[npz] NPZ_MODE=auto → plain_ae_slice_meta/patients"
    else
      PLAIN_DIR="${PLAIN_P}"
      echo "[npz] NPZ_MODE=auto → plain_ae/patients"
    fi
    if [[ -d "${MONAI_SM}" ]]; then
      MONAI_DIR="${MONAI_SM}"
      echo "[npz] NPZ_MODE=auto → monai_ae_slice_meta/patients"
    else
      MONAI_DIR="${MONAI_P}"
      echo "[npz] NPZ_MODE=auto → monai_ae/patients"
    fi
    ;;
  *)
    echo "NPZ_MODE must be auto | patients | slice_meta (got ${NPZ_MODE})" >&2
    exit 1
    ;;
esac

if [[ "${MONAI_VS_MONAI_VF}" == "1" ]]; then
  if [[ ! -d "${MONAI_PATIENTS_DIR}" ]]; then
    echo "MONAI_VS_MONAI_VF=1 but missing MONAI patients dir: ${MONAI_PATIENTS_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${MONAI_VF_PATIENTS_DIR}" ]]; then
    echo "MONAI_VS_MONAI_VF=1 but missing MONAI VF patients dir: ${MONAI_VF_PATIENTS_DIR}" >&2
    exit 1
  fi
else
  if [[ ! -d "${PLAIN_DIR}" ]]; then
    echo "Missing baseline plain_ae NPZ dir: ${PLAIN_DIR}" >&2
    exit 1
  fi
fi

if [[ -n "${OUT_PARENT:-}" ]]; then
  :
elif [[ "${MONAI_VS_MONAI_VF}" == "1" ]]; then
  OUT_PARENT="${LATENT_VOL}/repeated_head_sweep_monai"
else
  OUT_PARENT="${LATENT_VOL}/repeated_head_sweep_p90_volmm"
fi
mkdir -p "${OUT_PARENT}"

PY="${PYTHON:-python3}"
SWEEP_PY="${SCRIPT_DIR}/repeated_head_sweep_std.py"

POOLINGS="${POOLINGS:-p90_p10 global_mean_vol_mm p90_vol_mm regional_vol_mm p90_p10_global_mean_vol_mm p90_p10_p90_vol_mm p90_p10_regional_vol_mm}"
_POOLINGS=()
for _p in ${POOLINGS}; do
  if [[ "${_p}" == "p90_p10_volmm" ]]; then
    _POOLINGS+=("p90_p10_vol_mm")
  else
    _POOLINGS+=("${_p}")
  fi
done

N_SPLITS="${N_SPLITS:-100}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
EPS_MM="${EPS_MM:-1e-3}"
SWEEP_N_JOBS="${SWEEP_N_JOBS:-1}"
POOLING_PARALLEL="${POOLING_PARALLEL:-1}"
RUN_TAG_BASE="${RUN_TAG_BASE:-$(date +%Y%m%d_%H%M%S)}"
RUN_COEF_STABILITY="${RUN_COEF_STABILITY:-0}"
COEF_ONLY="${COEF_ONLY:-0}"
EN_C="${EN_C:-0.1}"
EN_L1_RATIO="${EN_L1_RATIO:-0.2}"
PLOT_TOP_K="${PLOT_TOP_K:-8}"

RECURSIVE_FLAG=()
if [[ "${RECURSIVE_NPZ:-0}" == "1" ]]; then
  RECURSIVE_FLAG=(--recursive_npz)
fi

_LABEL_ARGS=(--labels_csv "${TRAIN_CSV}" --labels_csv "${VAL_CSV}" --labels_csv "${TEST_CSV}")

if [[ "${MONAI_VS_MONAI_VF}" == "1" ]]; then
  _LATENT_ARGS=(
    --latent_source monai_ae "${MONAI_PATIENTS_DIR}"
    --latent_source monai_ae_VF "${MONAI_VF_PATIENTS_DIR}"
  )
  _SWP_SOURCES="monai_ae monai_ae_VF"
  echo "[npz] MONAI_VS_MONAI_VF: monai_ae → ${MONAI_PATIENTS_DIR}"
  echo "[npz] MONAI_VS_MONAI_VF: monai_ae_VF → ${MONAI_VF_PATIENTS_DIR}"
else
  _LATENT_ARGS=(--latent_source plain_ae "${PLAIN_DIR}")
  if [[ "${INCLUDE_MONAI}" == "1" ]]; then
    if [[ -d "${MONAI_DIR}" ]]; then
      _LATENT_ARGS+=(--latent_source monai_ae "${MONAI_DIR}")
      echo "[npz] monai_ae → ${MONAI_DIR}"
    else
      echo "[warn] INCLUDE_MONAI=1 but missing: ${MONAI_DIR}" >&2
    fi
  fi
  if [[ "${INCLUDE_PLAIN_VF}" == "1" ]]; then
    if [[ -d "${PLAIN_VF_DIR}" ]]; then
      _LATENT_ARGS+=(--latent_source plain_ae_VF "${PLAIN_VF_DIR}")
      echo "[npz] plain_ae_VF → ${PLAIN_VF_DIR}"
    else
      echo "[warn] INCLUDE_PLAIN_VF=1 but missing: ${PLAIN_VF_DIR}" >&2
    fi
  fi
  _SWP_SOURCES="plain_ae"
  if [[ "${INCLUDE_MONAI}" == "1" && -d "${MONAI_DIR}" ]]; then
    _SWP_SOURCES+=" monai_ae"
  fi
  if [[ "${INCLUDE_PLAIN_VF}" == "1" && -d "${PLAIN_VF_DIR}" ]]; then
    _SWP_SOURCES+=" plain_ae_VF"
  fi
fi

echo "LATENT_VOL=${LATENT_VOL}"
if [[ "${MONAI_VS_MONAI_VF}" == "1" ]]; then
  echo "MONAI_PATIENTS_DIR=${MONAI_PATIENTS_DIR}"
  echo "MONAI_VF_PATIENTS_DIR=${MONAI_VF_PATIENTS_DIR}"
else
  echo "PLAIN_DIR=${PLAIN_DIR}"
fi
echo "OUT_PARENT=${OUT_PARENT}"
echo "POOLINGS=${_POOLINGS[*]}"
echo "N_SPLITS=${N_SPLITS} test_size=${TEST_SIZE} random_state=${RANDOM_STATE}"
echo "SWEEP_N_JOBS=${SWEEP_N_JOBS} POOLING_PARALLEL=${POOLING_PARALLEL}"
echo "latent_sources: ${_SWP_SOURCES}"

_run_sweep_pooling() {
  local POOLING="$1"
  echo ""
  echo "=== pooling=${POOLING} (latent sources: ${_SWP_SOURCES}) ==="
  "${PY}" "${SWEEP_PY}" \
    "${_LABEL_ARGS[@]}" \
    "${_LATENT_ARGS[@]}" \
    --out_dir "${OUT_PARENT}" \
    --run_tag "${RUN_TAG_BASE}_${POOLING}" \
    --pooling "${POOLING}" \
    --eps_mm "${EPS_MM}" \
    --n_splits "${N_SPLITS}" \
    --test_size "${TEST_SIZE}" \
    --random_state "${RANDOM_STATE}" \
    --n_jobs "${SWEEP_N_JOBS}" \
    "${RECURSIVE_FLAG[@]:+${RECURSIVE_FLAG[@]}}"
}

if [[ "${COEF_ONLY}" != "1" ]]; then
if [[ "${POOLING_PARALLEL}" -le 1 ]]; then
  for POOLING in "${_POOLINGS[@]}"; do
    _run_sweep_pooling "${POOLING}"
  done
else
  declare -a _POOL_PIDS=()
  for POOLING in "${_POOLINGS[@]}"; do
    while [[ "${#_POOL_PIDS[@]}" -ge "${POOLING_PARALLEL}" ]]; do
      wait "${_POOL_PIDS[0]}"
      _POOL_PIDS=("${_POOL_PIDS[@]:1}")
    done
    _run_sweep_pooling "${POOLING}" &
    _POOL_PIDS+=($!)
  done
  for _pid in "${_POOL_PIDS[@]}"; do
    wait "${_pid}"
  done
fi
echo ""
echo "Merging run_*/summary.csv under ${OUT_PARENT} ..."
"${PY}" "${SCRIPT_DIR}/merge_repeated_runs_to_merged_csv.py" --out_parent "${OUT_PARENT}"
echo "Done. Merged: ${OUT_PARENT}/repeated_summary_merged_with_fixed.csv"
else
  echo "[info] COEF_ONLY=1 — skipping head sweep and merge."
fi

if [[ "${RUN_COEF_STABILITY}" == "1" || "${COEF_ONLY}" == "1" ]]; then
  COEF_ROOT="${OUT_PARENT}/coef_elasticnet_${RUN_TAG_BASE}"
  mkdir -p "${COEF_ROOT}"
  COEF_PY="${SCRIPT_DIR}/elasticnet_trajectory_coef_stability.py"
  echo ""
  echo "=== ElasticNet coef stability → ${COEF_ROOT}/<latent_source>/<pooling> ==="
  _coef_one_source() {
    local name="$1"
    local dir="$2"
    for POOLING in "${_POOLINGS[@]}"; do
      echo ""
      echo "--- coef stability ${name} pooling=${POOLING} ---"
      mkdir -p "${COEF_ROOT}/${name}"
      "${PY}" "${COEF_PY}" \
        "${_LABEL_ARGS[@]}" \
        --npz_dir "${dir}" \
        --pooling "${POOLING}" \
        --C "${EN_C}" \
        --l1_ratio "${EN_L1_RATIO}" \
        --eps_mm "${EPS_MM}" \
        --n_splits "${N_SPLITS}" \
        --test_size "${TEST_SIZE}" \
        --random_state "${RANDOM_STATE}" \
        --out_dir "${COEF_ROOT}/${name}/${POOLING}" \
        --plot_top_k "${PLOT_TOP_K}" \
        "${RECURSIVE_FLAG[@]:+${RECURSIVE_FLAG[@]}}"
    done
  }
  if [[ "${MONAI_VS_MONAI_VF}" == "1" ]]; then
    _coef_one_source "monai_ae" "${MONAI_PATIENTS_DIR}"
    _coef_one_source "monai_ae_VF" "${MONAI_VF_PATIENTS_DIR}"
  else
    _coef_one_source "plain_ae" "${PLAIN_DIR}"
    if [[ "${INCLUDE_MONAI}" == "1" && -d "${MONAI_DIR}" ]]; then
      _coef_one_source "monai_ae" "${MONAI_DIR}"
    fi
    if [[ "${INCLUDE_PLAIN_VF}" == "1" && -d "${PLAIN_VF_DIR}" ]]; then
      _coef_one_source "plain_ae_VF" "${PLAIN_VF_DIR}"
    fi
  fi
  echo "Coef stability: ${COEF_ROOT}/<latent_source>/<pooling>/"
fi
