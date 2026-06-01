# Copy or symlink to spatial_viz_paths.local.sh and edit if your disks differ.
# Usage (from repo root):
#   source stage_c/viz/spatial_viz_paths.example.sh
#   python3 utils/scripts/make_spatial_coeff_roi_gif.py \
#     --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \
#     --nii_roi "${ROI_TTS}/${PATIENT_ID}/${PATIENT_ID}_roi.nii.gz" \
#     --coef_npz "${COEF_HIST_TRAINVAL}" \
#     --out_gif "${FIG_OUT_DIR}/${PATIENT_ID}_axial.gif"

export VOL_PHD_PROJECT="${VOL_PHD_PROJECT:-/Volumes/Chanho_PhD_Project}"

export NPZ_STAGE_B_SPATIAL="${NPZ_STAGE_B_SPATIAL:-${VOL_PHD_PROJECT}/latent_data/diff3dformer_stageB_monai_ae_spatial_v2/patients}"
export ROI_TTS="${ROI_TTS:-${VOL_PHD_PROJECT}/DatasetRaw/TTS_RM_V2}"
export ROI_NORMAL="${ROI_NORMAL:-${VOL_PHD_PROJECT}/DatasetRaw/normal_RM_V2}"

# Paths below are relative to AE_TTS repo root; run `cd` there first.
export COEF_HIST_TRAINVAL="${COEF_HIST_TRAINVAL:-Report/Draft/figures/_ag_overlay/cluster_hist_lr_coef_trainval.npy}"
export FIG_OUT_DIR="${FIG_OUT_DIR:-figures}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.mplconfig}"
export PATIENT_ID="${PATIENT_ID:-DCA_09939935}"
