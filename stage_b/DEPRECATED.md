# Deprecated / do not use for new work

Retire or archive these when cleaning the repo. They are **not** part of `stageB_spatial_v2`.

## Stage B (legacy vector / leaky split)

| Path | Issue |
|------|--------|
| `plain_AE_stageA/StageB/run_stageB_plain_ae.sh` | `--extra_csvs` includes **test** in k-means fit |
| `sh/run_stageB_monai_ae.sh` | same `extra_csvs` pattern |
| `plain_AE_stageA/StageB/extract_embeddings_and_kmeans.py` | GAP/global 1D vector; use spatial v2 instead |
| `MONAI/extract_embeddings_and_kmeans_monai.py` | GAP bottleneck reduce |
| `Loss_experiment/run_stageB_monai_latent_var.sh` | `extra_csvs` legacy |

Outputs under `outputs/diff3dformer_stageB_plain_ae/` (non-spatial), `*_slice_meta`, `*_VF_loss`, etc. — keep only if you need diff; not for new papers.

## Evaluation

| Path | Issue |
|------|--------|
| `Loss_experiment/eval.py` | 100× `StratifiedShuffleSplit` on fixed NPZ; not independent test |

## Safe to keep (reference only)

- `stageB_spatial_v2/` — **use this**
- Stage A scripts under `plain_AE_stageA/`, `MONAI/` — still needed for encoder training

## Suggested cleanup order

1. Confirm no paper figure uses legacy OUT_DIR or `eval.py` repeats.
2. Move legacy OUT_DIR to `outputs/_archive/` (optional).
3. Delete or stub deprecated `run_stageB_*.sh` (or add `# DEPRECATED` header + exit 1).
4. Single symlink or README pointer → `stageB_spatial_v2/`.
