#!/usr/bin/env bash
# 기본 동작: monai_ae만 재평가하고, 루트 CSV들은 기존 파일에 이어 붙임(다른 latent_source 행 유지).
#   → MODE=monai_only, MERGE_EXISTING_ROOT_METRICS=1 (기본값)
#
# 출력:
#   OUT_NEW/monai_ae/  (PCA·UMAP·diagnostics/)
#   OUT_NEW/classifier_metrics_mean_std_pooling.csv 등 루트 표: monai_ae 행만 교체·추가
#
# LATENT_VOL 미지정 시: repo/latent_data에 monai patients가 있으면 그걸 쓰고,
# 없으면 /Volumes/Chanho_PhD_Project/latent_data 가 있으면 자동 사용.
# 그래도 없으면 repo/latent_data로 두고 → 없으면 에러 + LATENT_VOL= 안내.
# 직접 지정: LATENT_VOL=/path/to/latent_data bash utils/scripts/run_out_new_latent_eval.sh
#
# plain/diff/mse_ssim까지 한 번에 다시 돌리려면:
#   MODE=full MERGE_EXISTING_ROOT_METRICS=0 bash utils/scripts/run_out_new_latent_eval.sh
#
# Usage:
#   bash utils/scripts/run_out_new_latent_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

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
OUT_NEW="${OUT_NEW:-${LATENT_VOL}/out_new}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
MODE="${MODE:-monai_only}"
MERGE_EXISTING_ROOT_METRICS="${MERGE_EXISTING_ROOT_METRICS:-1}"

PY="${PYTHON:-python3}"
EVAL_PY="${REPO_ROOT}/utils/scripts/latent_space_pca_umap.py"

TRAIN_CSV="${SPLIT_DIR}/train.csv"
VAL_CSV="${SPLIT_DIR}/val.csv"
TEST_CSV="${SPLIT_DIR}/test.csv"
for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing split CSV: ${f}" >&2
    exit 1
  fi
done

declare -a LATENT_ARGS=()

add_source() {
  local name="$1"
  local dir="$2"
  if [[ -d "${dir}" ]]; then
    LATENT_ARGS+=(--latent_source "${name}" "${dir}")
    echo "  + latent_source ${name} -> ${dir}"
  else
    echo "  (skip) ${name}: not a directory: ${dir}" >&2
  fi
}

if [[ "${MODE}" == "monai_only" ]]; then
  add_source "monai_ae" "${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients"
else
  add_source "plain_ae" "${LATENT_VOL}/diff3dformer_stageB_plain_ae/patients"
  add_source "diffae" "${LATENT_VOL}/diff3dformer_stageB_diffae/patients"
  add_source "plain_mse_ssim" "${LATENT_VOL}/diff3dformer_stageB_plain_ae_mse_ssim/patients"
  add_source "monai_ae" "${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients"
fi

if [[ "${#LATENT_ARGS[@]}" -eq 0 ]]; then
  echo "No patient NPZ directories found under LATENT_VOL=${LATENT_VOL}" >&2
  echo "  MODE=${MODE} — expected at least one of:" >&2
  if [[ "${MODE}" == "monai_only" ]]; then
    echo "    ${LATENT_VOL}/diff3dformer_stageB_monai_ae/patients" >&2
  else
    echo "    .../diff3dformer_stageB_{plain_ae,diffae,plain_ae_mse_ssim,monai_ae}/patients" >&2
  fi
  echo "  Fix: copy Stage B NPZ tree into ${REPO_ROOT}/latent_data/ or run:" >&2
  echo "    LATENT_VOL=/path/to/latent_data bash utils/scripts/run_out_new_latent_eval.sh" >&2
  exit 1
fi

mkdir -p "${OUT_NEW}"

echo "OUT_NEW=${OUT_NEW}"
echo "SPLIT_DIR=${SPLIT_DIR}"
echo "MODE=${MODE}"
echo "MERGE_EXISTING_ROOT_METRICS=${MERGE_EXISTING_ROOT_METRICS}"

declare -a MERGE_ARGS=()
if [[ "${MERGE_EXISTING_ROOT_METRICS}" == "1" ]]; then
  MERGE_ARGS+=(--merge_existing_root_metrics)
  echo "MERGE_EXISTING_ROOT_METRICS=1 (루트 표: 이번 실행 latent_source 행만 교체·추가)"
fi

exec "${PY}" "${EVAL_PY}" \
  --eval_classifiers \
  --eval_diagnostics \
  --split_dir "${SPLIT_DIR}" \
  --labels_csv "${TRAIN_CSV}" \
  --labels_csv "${VAL_CSV}" \
  --labels_csv "${TEST_CSV}" \
  "${LATENT_ARGS[@]}" \
  --out_dir "${OUT_NEW}" \
  --bootstrap_n "${BOOTSTRAP_N:-1000}" \
  --label_shuffle_n "${LABEL_SHUFFLE_N:-200}" \
  "${MERGE_ARGS[@]:+${MERGE_ARGS[@]}}"
