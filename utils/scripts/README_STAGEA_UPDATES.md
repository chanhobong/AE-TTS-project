## Stage A (DAE2D) Updates Summary

This document summarizes the modifications made to Stage‑A training to prevent encoder collapse and enforce conditioning usage (Diff3Dformer‑style slice DAE).

### Goals
- Make the 512‑dim `y_sem` embedding actually influence the denoiser.
- Keep the design diffusion‑autoencoder‑consistent (no VAE, no latent diffusion).
- Provide diagnostics to confirm conditioning is used.

### Key Changes
1) **CFG‑style conditioning dropout**
- During training, with probability `p_uncond`, set `y_sem_used[i] = 0`.
- Helps the denoiser learn to rely on conditioning (classifier‑free guidance idea).
- Validation uses **no dropout** (deterministic).

2) **Two‑term training loss**
- Diffusion loss: `L_diff = MSE(eps_pred, noise)`
- Reconstruction loss: `L_x0 = L1(x0_hat, x0)`
- Total: `L = L_diff + lambda_x0 * L_x0`
- `x0_hat` is clamped to `[-1, 1]` before computing `L_x0`.

3) **Warm‑up schedules**
- `p_uncond` increases linearly from 0 → `--p_uncond` over `--p_uncond_warmup_steps`.
- `lambda_x0` increases linearly from 0 → `--lambda_x0` over `--lambda_x0_warmup_steps`.

4) **Diagnostics on first batch**
Printed once per epoch:
- `y_sem_var` (should be > 0 to avoid collapse)
- `loss_real`, `loss_used`, `loss_rand`, `loss_zero`
  - Expect: `loss_real < loss_rand` and `loss_real < loss_zero`
- AdaGN scale/shift magnitude averaged across all AdaGN blocks

### Files Changed
- `AE_TTS/models/dae2d.py`
  - Added `get_adagn_stats()` to average AdaGN stats across all blocks.
- `AE_TTS/scripts/train_dae2d.py`
  - Added CFG‑style conditioning dropout.
  - Added x0 reconstruction loss and warm‑up schedules.
  - Added diagnostics.

### New/Updated CLI Flags
- `--p_uncond` (default 0.1)
- `--p_uncond_warmup_steps` (default 2000)
- `--lambda_x0` (default 0.1)
- `--lambda_x0_warmup_steps` (default 2000)

### Example Command
```bash
python /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/scripts/train_dae2d.py \
  --normal_root /path/to/normal_RM_V2 \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/train.csv \
  --val_csv /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/data/val.csv \
  --out_dir /mnt/iusers01/eee01/g09698cb/scratch/AE_TTS/outputs/train_dae2d \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4 \
  --p_uncond 0.1 \
  --p_uncond_warmup_steps 2000 \
  --lambda_x0 0.1 \
  --lambda_x0_warmup_steps 2000
```
