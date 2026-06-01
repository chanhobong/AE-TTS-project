#!/usr/bin/env bash
# Renamed: MONAI is cluster_hist + rbf_svc. This wrapper forwards to the current script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[note] use ${HERE}/run_paired_plain_p90_p10_vol_mm_en__monai_cluster_hist_rbf_ensemble.sh" >&2
exec bash "${HERE}/run_paired_plain_p90_p10_vol_mm_en__monai_cluster_hist_rbf_ensemble.sh"
