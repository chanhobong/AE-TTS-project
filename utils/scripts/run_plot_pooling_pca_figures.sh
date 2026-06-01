#!/usr/bin/env bash
# One-shot PCA triptych figures (Normal vs TTS), with explained variance in axis labels.
#
# Always runs:
#   1) plot_pooling_pca_triptych.py        → Plain AE: Mean | Std | P90–P10 (trajectory-ordered NPZ)
#   2) plot_pooling_pca_mean_std_cluster_hist.py → MONAI AE: Mean | Std | cluster histogram (infer K)
#   3) plot_plain_monai_fusion_pca.py      → Single PCA: Plain P90–P10 ⊕ MONAI cluster-hist (intersection, block-z)
#
# If CLUSTER_HIST_K is set (e.g. 64), also runs:
#   4) plot_monai_pooling_pca_triptych.py  → same MONAI triple with fixed K (file name includes K)
#
# Example (from repo root AE_TTS):
#   LATENT_VOL=/Volumes/Chanho_PhD_Project/latent_data CLUSTER_HIST_K=64 \\
#     bash utils/scripts/run_plot_pooling_pca_figures.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LATENT_VOL="${LATENT_VOL:-${REPO_ROOT}/latent_data}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data}"
PY="${PYTHON:-python3}"
FIG_DIR="${FIG_DIR:-${REPO_ROOT}/Report/Draft/figures}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/train.csv}"
EPS_MM="${EPS_MM:-1e-3}"
DPI="${DPI:-150}"
TITLE="${TITLE:-}"

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
[[ -f "${SPLIT_DIR}/train.csv" && -f "${SPLIT_DIR}/val.csv" && -f "${SPLIT_DIR}/test.csv" ]] || {
  echo "Need ${SPLIT_DIR}/{train,val,test}.csv" >&2
  exit 1
}
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing --train_csv target: ${TRAIN_CSV}" >&2; exit 1; }

RECURSIVE_PLAIN=()
if [[ "${PLAIN_RECURSIVE_NPZ:-0}" == "1" || "${PLAIN_RECURSIVE_NPZ:-0}" == "true" ]]; then
  RECURSIVE_PLAIN=(--recursive_npz)
fi
RECURSIVE_MONAI=()
if [[ "${MONAI_RECURSIVE_NPZ:-0}" == "1" || "${MONAI_RECURSIVE_NPZ:-0}" == "true" ]]; then
  RECURSIVE_MONAI=(--recursive_npz)
fi

TITLE_PLAIN=()
TITLE_MONAI=()
TITLE_MONAI_K=()
if [[ -n "${TITLE}" ]]; then
  TITLE_PLAIN=(--title "${TITLE} — Plain (trajectory pooling)")
  TITLE_MONAI=(--title "${TITLE} — MONAI (mean/std/cluster, inferred K)")
  TITLE_MONAI_K=(--title "${TITLE} — MONAI (mean/std/cluster, K=${CLUSTER_HIST_K})")
fi

echo "========== [1] Plain: Mean | Std | P90–P10 =========="
"${PY}" "${SCRIPT_DIR}/plot_pooling_pca_triptych.py" \
  --npz_dir "${PLAIN_NPZ}" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --train_csv "${TRAIN_CSV}" \
  --out_png "${FIG_DIR}/fig_pooling_pca_triptych.png" \
  --eps_mm "${EPS_MM}" \
  ${RECURSIVE_PLAIN[@]+"${RECURSIVE_PLAIN[@]}"} \
  --dpi "${DPI}" \
  ${TITLE_PLAIN[@]+"${TITLE_PLAIN[@]}"}

echo "========== [2] MONAI: Mean | Std | cluster-hist (inferred K) =========="
"${PY}" "${SCRIPT_DIR}/plot_pooling_pca_mean_std_cluster_hist.py" \
  --npz_dir "${MONAI_NPZ}" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --train_csv "${TRAIN_CSV}" \
  --out_png "${FIG_DIR}/fig_pooling_pca_mean_std_hist_monai.png" \
  ${RECURSIVE_MONAI[@]+"${RECURSIVE_MONAI[@]}"} \
  --dpi "${DPI}" \
  ${TITLE_MONAI[@]+"${TITLE_MONAI[@]}"}

echo "========== [3] Fusion PCA: Plain P90–P10 ⊕ MONAI cluster-hist =========="
FUSION_EXTRA=()
if [[ -n "${CLUSTER_HIST_K:-}" ]]; then
  FUSION_EXTRA+=(--cluster_hist_k "${CLUSTER_HIST_K}")
fi
FUSION_P_REC=()
if [[ "${PLAIN_RECURSIVE_NPZ:-0}" == "1" || "${PLAIN_RECURSIVE_NPZ:-0}" == "true" ]]; then
  FUSION_P_REC=(--plain_recursive_npz)
fi
FUSION_M_REC=()
if [[ "${MONAI_RECURSIVE_NPZ:-0}" == "1" || "${MONAI_RECURSIVE_NPZ:-0}" == "true" ]]; then
  FUSION_M_REC=(--monai_recursive_npz)
fi
"${PY}" "${SCRIPT_DIR}/plot_plain_monai_fusion_pca.py" \
  --plain_npz_dir "${PLAIN_NPZ}" \
  --monai_npz_dir "${MONAI_NPZ}" \
  --labels_csv "${SPLIT_DIR}/train.csv" \
  --labels_csv "${SPLIT_DIR}/val.csv" \
  --labels_csv "${SPLIT_DIR}/test.csv" \
  --train_csv "${TRAIN_CSV}" \
  --out_png "${FIG_DIR}/fig_plain_monai_fusion_pca.png" \
  --eps_mm "${EPS_MM}" \
  ${FUSION_P_REC[@]+"${FUSION_P_REC[@]}"} \
  ${FUSION_M_REC[@]+"${FUSION_M_REC[@]}"} \
  ${FUSION_EXTRA[@]+"${FUSION_EXTRA[@]}"} \
  --dpi "${DPI}"

if [[ -n "${CLUSTER_HIST_K:-}" ]]; then
  echo "========== [4] MONAI triptych (fixed K=${CLUSTER_HIST_K}) =========="
  "${PY}" "${SCRIPT_DIR}/plot_monai_pooling_pca_triptych.py" \
    --npz_dir "${MONAI_NPZ}" \
    --labels_csv "${SPLIT_DIR}/train.csv" \
    --labels_csv "${SPLIT_DIR}/val.csv" \
    --labels_csv "${SPLIT_DIR}/test.csv" \
    --train_csv "${TRAIN_CSV}" \
    --cluster_hist_k "${CLUSTER_HIST_K}" \
    --out_png "${FIG_DIR}/fig_monai_pooling_pca_triptych_k${CLUSTER_HIST_K}.png" \
    --dpi "${DPI}" \
    ${TITLE_MONAI_K[@]+"${TITLE_MONAI_K[@]}"}
fi

echo "Done. Figures under: ${FIG_DIR}"
