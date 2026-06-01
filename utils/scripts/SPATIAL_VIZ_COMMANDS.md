# Spatial coef overlay — 경로 맞춤

항상 **`AE_TTS` 저장소 루트**에서 명령을 실행합니다 (`utils/scripts/...` 기준 경로 고정).

## 1. 공통 패스 불러오기

```bash
cd /Users/ch.b/Desktop/24-25/TSS/Code/TTS_Project/LDAE_TTS/AE_TTS
source utils/scripts/spatial_viz_paths.example.sh
mkdir -p .mplconfig
```

다른 디스크 이름이면 `spatial_viz_paths.example.sh`를 복사해 `VOL_PHD_PROJECT` 등만 고치세요.

## 2. 축별 GIF (`col_major` / `heatmap_flip ud`)

```bash
python3 utils/scripts/make_spatial_coeff_roi_gif.py \
  --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \
  --nii_roi "${ROI_TTS}/${PATIENT_ID}/${PATIENT_ID}_roi.nii.gz" \
  --coef_npz "${COEF_HIST_TRAINVAL}" \
  --out_gif "${FIG_OUT_DIR}/${PATIENT_ID}_roi_axial_colmaj_heat_ud.gif" \
  --spatial_decode col_major \
  --heatmap_flip ud \
  --underlying_gamma 0.62 \
  --underlying_brightness_mult 0.9 \
  --overlay_alpha 0.38 \
  --frame_duration_ms 100
```

## 3. 메타 `slice_position_norm` + `spatial_y/x` 단일 오버레이 PNG

stderr에 **`display≈slice_index`** 가 나오면, 그 axial 슬라이스 ROI를 저장한 `--underlying_npy`와 짝지으면 됩니다.

```bash
python3 utils/scripts/plot_spatial_coeff_grid_overlay.py \
  --npz "${NPZ_STAGE_B_SPATIAL}/${PATIENT_ID}.npz" \
  --coef_npz "${COEF_HIST_TRAINVAL}" \
  --slice_selector position_norm \
  --slice_position_center 0.8 \
  --slice_position_halfwidth 0.12 \
  --contrib_top_positive_k 8 \
  --contrib_emb_alignment cosine_relu \
  --emb_raw_norm_dampen \
  --underlying_npy "${FIG_OUT_DIR}/_tmp_slice_aligned.npy" \
  --out_png "${FIG_OUT_DIR}/${PATIENT_ID}_position_norm_overlay.png"
```

정상(normal) 코호트 ROI는 `--nii_roi`가 `ROI_NORMAL` 계열입니다.
