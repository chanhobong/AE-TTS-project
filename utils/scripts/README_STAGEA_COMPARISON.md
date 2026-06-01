# Stage A Study Notes: DAE2D vs DiffAE (CT Slices)

This document summarizes the two Stage A approaches in this repo:
- **DAE2D**: our custom 2D diffusion autoencoder for axial CT slices.
- **DiffAE**: the `diffae-master` implementation adapted for CT slices.

The goal of both is the same: learn a slice-level representation and a
conditional denoiser that uses the representation meaningfully.

## 0) Visual overview (ASCII)
DAE2D (custom):
```
CT volume (B,1,D,H,W)
        |
        v
Slice flatten -> (B*D,1,H,W) = x0
        |
        +--> Encoder E_phi -> y_sem (B*D,512)
        |
        +--> q_sample(x0,t,noise) -> x_t
                          |
                          v
          UNet2DConditional(x_t,t_norm,y_sem) -> eps_pred
                          |
                          v
      Loss: MSE(eps_pred, noise) + lambda_x0 * L1(x0_hat, x0)
```

DiffAE (diffae-master):
```
CT slices -> CTSliceDataset -> {"img": x0}
        |
        v
BeatGANs Encoder -> style vector
        |
        v
Conditional UNet denoiser -> eps_pred
        |
        v
DiffAE training loop (template-driven)
```


## 1) DAE2D (Custom)

### 1.1 High-level idea
We model each axial slice as a 2D image. The encoder maps a slice to a
semantic vector `y_sem` (default 512-d). A conditional UNet predicts the
noise at a random diffusion timestep, using `y_sem` via AdaGN.

### 1.2 Data flow (per training step)
1) **Load volume batch**: `x` is `(B, 1, D, H, W)` from `CTROIVolumeDataset`.
2) **Flatten to slices**: `x0` becomes `(B*D, 1, H, W)`.
3) **Sample t and noise**:
   - `t ~ Uniform{0..T-1}` per slice
   - `noise ~ N(0, I)`
4) **Forward diffusion**: `x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise`
5) **Encode condition**: `y_sem = Encoder(x0)`
6) **CFG-style drop** (train only): with prob `p_uncond`, set `y_sem=0`
7) **Denoise**: `eps_pred = UNet(x_t, t_norm, y_sem_used)`
8) **Losses**:
   - `L_diff = MSE(eps_pred, noise)`
   - `L_x0 = L1(clamp(x0_hat), x0)`
   - `x0_hat` from DDPM inverse: `(x_t - sqrt(1-a_bar)*eps_pred)/sqrt(a_bar)`
   - Total `L = L_diff + lambda_x0 * L_x0`

### 1.2.1 Core diffusion equations (DAE2D)
We use the standard forward diffusion:
```
q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
eps ~ N(0, I)
```
Noise prediction objective:
```
L_diff = || eps_pred(x_t, t, y_sem) - eps ||_2^2
```
Reconstruction from predicted noise:
```
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / (sqrt(alpha_bar_t) + 1e-8)
```
Auxiliary reconstruction loss:
```
L_x0 = || clamp(x0_hat, -1, 1) - x0 ||_1
```
Total objective:
```
L = L_diff + lambda_x0 * L_x0
```

### 1.3 Architecture details
**Encoder** (`SliceEncoder2D`):
- 4 conv stages, stride-2 downsampling
- Global avg pool to 1x1
- Linear layer to 512-d vector

**Denoiser** (`UNet2DConditional`):
- 3 down blocks + 2 mid blocks + 3 up blocks
- Each block is a `ResBlock` with:
  - time embedding (sinusoidal + MLP)
  - AdaGN conditioning on `y_sem`
- Output is predicted noise `eps_pred`

**Conditioning (AdaGN)**:
- Normalizes feature map with GroupNorm
- Generates scale/shift from `y_sem`
- Applies: `h * (1 + scale) + shift`

### 1.3.1 DAE2D tensor shapes (default settings)
Assume:
- Input slice size `H = W = 128`
- `in_ch = 1`
- `base_ch = 64`
- `y_sem_dim = 512`

**Encoder (SliceEncoder2D)**:
```
x0:            (B*D, 1, 128, 128)
conv1 3x3:     (B*D, 64, 128, 128)
conv2 4x4 s2:  (B*D, 128, 64, 64)
conv3 4x4 s2:  (B*D, 256, 32, 32)
conv4 4x4 s2:  (B*D, 512, 16, 16)
avg pool:      (B*D, 512, 1, 1)
fc:            (B*D, 512) = y_sem
```

**UNet2DConditional**:
```
x_t:           (B*D, 1, 128, 128)
down1:         (B*D, 64, 128, 128)
downsample1:   (B*D, 64, 64, 64)
down2:         (B*D, 128, 64, 64)
downsample2:   (B*D, 128, 32, 32)
down3:         (B*D, 256, 32, 32)
downsample3:   (B*D, 256, 16, 16)
mid1/mid2:     (B*D, 256, 16, 16)
upsample3:     (B*D, 256, 32, 32)
concat d3:     (B*D, 512, 32, 32)
up3:           (B*D, 128, 32, 32)
upsample2:     (B*D, 128, 64, 64)
concat d2:     (B*D, 256, 64, 64)
up2:           (B*D, 64, 64, 64)
upsample1:     (B*D, 64, 128, 128)
concat d1:     (B*D, 128, 128, 128)
up1:           (B*D, 64, 128, 128)
out 3x3:       (B*D, 1, 128, 128) = eps_pred
```

**Time embedding shapes**:
```
t_norm: (B*D,) -> time_emb MLP -> (B*D, time_dim)
```

**AdaGN shapes**:
```
cond:  (B*D, 512)
scale: (B*D, C, 1, 1)
shift: (B*D, C, 1, 1)
```

**Notes**:
- With 3 downsamplings, `H` and `W` should be divisible by 8.
- For a different `base_ch`, scale channels by that factor.

### 1.4 Conditioning safeguards (to prevent collapse)
These changes were explicitly added to make the encoder useful:
- **CFG-style dropout**: randomly set `y_sem=0` during training.
- **Auxiliary x0 loss**: add `L_x0` to force recon signal.
- **Clamp x0_hat**: stabilize the L1 loss by clamping to `[-1,1]`.
- **Warm-up schedules**:
  - `p_uncond` linearly ramps from 0 → target
  - `lambda_x0` linearly ramps from 0 → target

### 1.5 Diagnostics (first batch)
We log signals to check if the conditioning is actually used:
- `y_sem_var`: if ~0, encoder collapsed.
- `loss_real`: use real `y_sem`
- `loss_rand`: permute `y_sem` (should be higher than real)
- `loss_zero`: set `y_sem=0` (should be higher than real)
- `adagn_scale/shift`: magnitude of conditioning effect

**Interpretation rule of thumb**:
- If `loss_real ≈ loss_rand ≈ loss_zero`: conditioning not used.
- If `y_sem_var ≈ 0`: encoder collapse.
- If AdaGN magnitudes ~0: conditioning not injected.

### 1.5.1 Log interpretation guide (DAE2D)
Common patterns and what they mean:
- `loss_real < loss_rand` and `loss_real < loss_zero`: conditioning is used.
- `loss_real ≈ loss_rand`: denoiser ignores `y_sem`.
- `y_sem_var` trending upward: encoder becoming diverse.
- `adagn_scale/shift` tiny: AdaGN may be ineffective; check encoder output scale.
- `loss_x0` exploding: clamp may be off, or learning rate too high.

Practical thresholds (very rough):
- `y_sem_var` should move away from ~0 within a few epochs.
- `loss_rand - loss_real` should be > 0 and consistent.

### 1.6 Key files
- `AE_TTS/models/dae2d.py`: model (encoder, UNet, AdaGN, diffusion schedule)
- `AE_TTS/scripts/train_dae2d.py`: training loop and diagnostics
- `AE_TTS/data/dataset.py`: `CTROIVolumeDataset`

### 1.7 Typical CLI usage
```
python AE_TTS/scripts/train_dae2d.py \
  --normal_root /path/to/CT \
  --metadata_csv /path/to/meta.csv \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --epochs 100 \
  --batch_size 2 \
  --p_uncond 0.1 \
  --p_uncond_warmup_steps 2000 \
  --lambda_x0 0.1 \
  --lambda_x0_warmup_steps 2000
```


## 2) DiffAE (diffae-master) Stage A for CT

### 2.1 High-level idea
DiffAE uses a **BeatGANs autoencoder** where:
- Encoder produces a latent style vector (`style_ch`)
- A conditional denoising UNet learns to reverse diffusion

We adapt its dataset interface to CT slices and force 1-channel input/output.

### 2.2 Data flow (per training step)
1) **Dataset**: `CTSliceDataset` yields dict: `{ "img": tensor, "index": idx }`
2) **Model**:
   - Encoder produces style vector
   - UNet denoises with conditioning
3) **Training** is controlled by `diffae-master` templates and config

### 2.2.1 DiffAE objective (conceptual)
DiffAE follows the standard noise-prediction objective, but the exact
loss setup is defined by the template config. Conceptually:
```
L_diffae = E_{x0,t,eps} [ || eps_pred(x_t, t, style) - eps ||_2^2 ]
```
Where `style` is the encoder output (style vector).

### 2.2.2 DiffAE tensor shapes (template default)
The CT script uses `ffhq128_autoenc_base()` as the template:
- `img_size = 128`
- `net_ch = 128`
- `net_ch_mult = (1, 1, 2, 3, 4)` for the UNet
- `net_enc_channel_mult = (1, 1, 2, 3, 4, 4)` for the encoder
 - `net_num_res_blocks = 2` (UNet)
 - `net_enc_num_res_blocks = 2` (encoder)
 - `net_resblock_updown = True` (down/up via ResBlock)

From the template comments:
```
UNet resolutions (img_size=128):
128 -> 128 -> 64 -> 32 -> 16 -> 8
```
Approximate channel progression (UNet):
```
C = net_ch * mult
128 * 1, 128 * 1, 128 * 2, 128 * 3, 128 * 4
```
Encoder resolution ends at `4x4` (per template comment) with
`net_enc_channel_mult = (1, 1, 2, 3, 4, 4)`.

**Practical interpretation**:
- Input slice: `(B, 1, 128, 128)`
- Encoder: multiple downsampling blocks to `4x4` with higher channels
- Style vector size: `style_ch` (default 512)
- UNet: downsamples to `8x8`, then upsamples back to `128x128`

Exact internal block shapes are defined by BeatGANs modules in
`diffae-master`, but the dimensions below follow those configs.

### 2.2.3 DiffAE UNet shape table (ffhq128_autoenc_base)
Notation:
- `C0 = net_ch = 128`
- `mult = (1, 1, 2, 3, 4)`
- `C_l = C0 * mult[l]`
- Each level has `num_res_blocks = 2`
- Down/Up are `ResBlock(..., down=True/ up=True)` because
  `net_resblock_updown=True`

**UNet Down path (encoder side)**:
```
Input x_t:             (B, 1, 128, 128)
initial 3x3 conv:      (B, 128, 128, 128)

Level 0 (res=128, C=128):
  ResBlock x2 ->       (B, 128, 128, 128)
  Downsample ->        (B, 128, 64, 64)

Level 1 (res=64,  C=128):
  ResBlock x2 ->       (B, 128, 64, 64)
  Downsample ->        (B, 128, 32, 32)

Level 2 (res=32,  C=256):
  ResBlock x2 ->       (B, 256, 32, 32)
  Downsample ->        (B, 256, 16, 16)

Level 3 (res=16,  C=384):
  ResBlock x2 ->       (B, 384, 16, 16)
  Downsample ->        (B, 384, 8, 8)

Level 4 (res=8,   C=512):
  ResBlock x2 ->       (B, 512, 8, 8)

Middle block:
  ResBlock + Attn + ResBlock
  Output:              (B, 512, 8, 8)
```

**UNet Up path (decoder side)**:
Each level uses `num_res_blocks + 1 = 3` blocks. The final block at a level
upsamples (except at the highest resolution).
```
Level 4 (res=8,   C=512):
  ResBlock x3 ->       (B, 512, 8, 8)
  Upsample ->          (B, 512, 16, 16)

Level 3 (res=16,  C=384):
  ResBlock x3 ->       (B, 384, 16, 16)
  Upsample ->          (B, 384, 32, 32)

Level 2 (res=32,  C=256):
  ResBlock x3 ->       (B, 256, 32, 32)
  Upsample ->          (B, 256, 64, 64)

Level 1 (res=64,  C=128):
  ResBlock x3 ->       (B, 128, 64, 64)
  Upsample ->          (B, 128, 128, 128)

Level 0 (res=128, C=128):
  ResBlock x3 ->       (B, 128, 128, 128)
  Output 3x3 conv ->   (B, out_channels, 128, 128)  # out_channels=1 (CT)
```

**Skip connections**:
Skip features are collected from the down path and concatenated into the
up path. The channel counts above already reflect the *post-resblock* output
channels at each level; internally, the input to each up ResBlock is
`ch + skip_ch`.

### 2.2.4 DiffAE Encoder shape table (ffhq128_autoenc_base)
Encoder uses `net_enc_channel_mult = (1, 1, 2, 3, 4, 4)` with
`net_enc_num_res_blocks = 2`, and `enc_pool = adaptivenonzero`.

```
Input x0:              (B, 1, 128, 128)
initial 3x3 conv:      (B, 128, 128, 128)

Level 0 (res=128, C=128):
  ResBlock x2 ->       (B, 128, 128, 128)
  Downsample ->        (B, 128, 64, 64)

Level 1 (res=64,  C=128):
  ResBlock x2 ->       (B, 128, 64, 64)
  Downsample ->        (B, 128, 32, 32)

Level 2 (res=32,  C=256):
  ResBlock x2 ->       (B, 256, 32, 32)
  Downsample ->        (B, 256, 16, 16)

Level 3 (res=16,  C=384):
  ResBlock x2 ->       (B, 384, 16, 16)
  Downsample ->        (B, 384, 8, 8)

Level 4 (res=8,   C=512):
  ResBlock x2 ->       (B, 512, 8, 8)
  Downsample ->        (B, 512, 4, 4)

Level 5 (res=4,   C=512):
  ResBlock x2 ->       (B, 512, 4, 4)

Middle block:
  ResBlock + Attn + ResBlock
  Output:              (B, 512, 4, 4)

Pool + 1x1 conv + flatten:
  AdaptiveAvgPool ->   (B, 512, 1, 1)
  1x1 conv ->          (B, style_ch, 1, 1)  # style_ch=512
  Flatten ->           (B, 512)
```

**Notes**:
- Encoder and UNet use attention at resolutions in `net_attn` (default `(16,)`).
- For CT, `in_channels = out_channels = 1`.

### 2.3 CT adaptation changes
We added fields to `TrainConfig` and modified the dataset loader:
- `ct_root`, `ct_metadata_csv`, `ct_split_csv`
- `ct_expected_shape_xyz`, `ct_num_channels`
- `ct_ct_suffix`, `ct_mask_filename`
- `ct_apply_lv_mask`
- `in_channels`, `out_channels` explicitly set to 1

We also ensured:
- `model_out_channels` uses `conf.out_channels`
- `sample()` and `infer()` use `conf.in_channels`
- `log_sample()` aligns `x_T` batch size

### 2.4 Training entry point
The CT adapter script:
- `AE_TTS/scripts/train_diffae_stageA_ct.py`
  - Loads a template config from `diffae-master`
  - Overrides dataset and channel settings
  - Calls `experiment.train(conf, gpus=..., nodes=...)`

Relevant config in `diffae-master/config.py`:
- `TrainConfig.make_dataset()` has a `ct_slices` branch
- `TrainConfig.make_model_conf()` uses `in_channels/out_channels`

### 2.5 Typical CLI usage
```
python AE_TTS/scripts/train_diffae_stageA_ct.py \
  --ct_root /path/to/CT \
  --metadata_csv /path/to/meta.csv \
  --train_csv /path/to/train.csv \
  --expected_shape 128 128 64 \
  --batch_size 4 \
  --lr 1e-4 \
  --total_samples 2000000 \
  --in_channels 1 \
  --out_channels 1 \
  --style_ch 512 \
  --gpus 0
```

### 2.6 Key files
- `AE_TTS/scripts/train_diffae_stageA_ct.py`: CT wrapper + config override
- `AE_TTS/diffae-master/config.py`: config and dataset integration
- `AE_TTS/diffae-master/dataset.py`: `CTSliceDataset` for DiffAE
- `AE_TTS/diffae-master/experiment.py`: training loop and sampling


## 3) Side-by-side comparison

| Aspect | DAE2D (Custom) | DiffAE (diffae-master) |
|---|---|---|
| Input unit | 2D slice (from CT volume) | 2D slice (from CT volume) |
| Encoder output | `y_sem` (default 512-d) | `style_ch` (default 512) |
| Denoiser | Custom UNet2D with AdaGN | BeatGANs UNet |
| Conditioning | AdaGN from `y_sem` | BeatGANs-style conditioning |
| Loss | `MSE(eps_pred, noise)` + `lambda_x0 * L1(x0_hat, x0)` | DiffAE default objectives |
| CFG dropout | Yes (`p_uncond`) | No (not by default) |
| Warm-up | `p_uncond` and `lambda_x0` warm-up | Not in base config |
| Diagnostics | `loss_real/rand/zero`, `y_sem_var`, AdaGN stats | Default DiffAE logs |
| Code location | `AE_TTS/models/dae2d.py` | `AE_TTS/diffae-master/*` |
| Channel fix | Manual (always 1) | Config override for CT |


## 4) Troubleshooting guide

### 4.1 Conditioning collapse (DAE2D)
Symptoms:
- `y_sem_var ~ 0`
- `loss_real ≈ loss_rand ≈ loss_zero`
Likely causes:
- Conditioning not injected (AdaGN magnitudes ~0)
- Encoder too weak or saturated
What to check:
- `p_uncond` schedule ramping
- `lambda_x0` warm-up active
- AdaGN stats are non-zero

### 4.2 Channel mismatch (DiffAE)
Symptoms:
- Runtime errors expecting 1 channel but got 3
Fixes already applied:
- `conf.in_channels = 1`, `conf.out_channels = 1`
- `model_out_channels` uses `conf.out_channels`
- Noise creation uses `conf.in_channels`
If it reappears:
- Verify `data_name == "ct_slices"`
- Check `conf.make_model_conf()` was called after overrides

### 4.3 Loss is noisy
This is normal in diffusion. Look for trend over epochs:
- If loss decreases on average, training is fine.
- If it diverges, reduce LR or increase batch size.

### 4.4 Debug checklist for logs
If training looks unstable:
- Check `p_uncond` warm-up is not too aggressive.
- Verify `lambda_x0` is not too large early.
- Inspect `loss_real` vs `loss_rand` on first batch.
- Confirm input normalization range (expected `[-1, 1]`).


## 5) Practical study checklist
- Confirm dataset path and CSVs used by the script.
- Confirm `in_channels=1` end-to-end.
- Confirm `y_sem`/`style_ch` dimension (usually 512).
- Watch `loss_real` vs `loss_rand/zero`.
- Track moving average of train loss, not per-step noise.
- Read the config-driven DiffAE defaults before interpreting logs.


## 6) When to use which
Use **DAE2D** if:
- You want explicit diagnostics and control
- You want CFG-style dropout and x0 loss

Use **DiffAE** if:
- You want a stronger, proven baseline quickly
- You want to compare against a published implementation


## 7) Output artifacts
DAE2D:
- `encoder_checkpoint.pth` (encoder + denoiser)
- `loss_history.csv`

DiffAE:
- Saved checkpoints under the experiment name directory
- Logs controlled by `diffae-master` trainer


## 8) Glossary (quick study)
- `x0`: clean slice image (input to diffusion).
- `x_t`: noisy slice at timestep `t`.
- `eps`: Gaussian noise sampled for forward diffusion.
- `eps_pred`: denoiser’s prediction of `eps`.
- `alpha_bar_t`: cumulative product of `(1 - beta_t)` up to `t`.
- `y_sem`: DAE2D semantic vector (slice embedding).
- `style_ch`: DiffAE encoder output channels (style vector dim).
- `AdaGN`: GroupNorm + affine modulation from condition vector.
- `CFG drop`: conditioning dropout (use zeros sometimes).


## 9) Training checklist (before you run)
- **Data**: CT slices normalized to `[-1, 1]`.
- **Channels**: `in_channels=1`, `out_channels=1` end-to-end.
- **Sizes**: `img_size=128` (or divisible by 8 for DAE2D).
- **Batch**: watch GPU memory; DiffAE uses slices as batch items.
- **AMP**: use fp16/amp only if gradients remain stable.
- **Seeds**: fix `seed` for reproducible diagnostics.


## 10) Log interpretation table (symptom → cause → fix)
| Symptom | Likely cause | Fix |
|---|---|---|
| `y_sem_var ~ 0` | encoder collapse | raise `lambda_x0`, check LR, verify CFG drop |
| `loss_real ≈ loss_rand` | condition ignored | increase `p_uncond`, check AdaGN stats |
| AdaGN stats ~0 | conditioning too weak | check encoder output scale, try higher `base_ch` |
| loss oscillates wildly | LR too high | lower LR or increase batch |
| NaNs | overflow/instability | disable AMP, reduce LR, grad clip |


## 11) Ablation guide (study experiments)
- `lambda_x0 = 0`: remove x0 guidance, compare `y_sem_var`.
- `p_uncond = 0`: no CFG drop, check if conditioning still used.
- freeze encoder for N epochs, then unfreeze (to see denoiser behavior).
- reduce `y_sem_dim` (512 → 256) and compare clustering quality.


## 12) Stage B bridge (what to save)
For Stage B (k-means on embeddings):
- Extract `y_sem` / `style` per slice.
- L2 normalize embeddings before spherical k-means.
- Save per patient:
  - `embeddings`: `(num_slices, dim)`
  - `cluster_id`: `(num_slices,)`
  - `mask`: `(num_slices,)` if padding used


## 13) Practical tips (stability & speed)
- **EMA** (DiffAE): usually improves sample stability.
- **Grad clip**: keep `grad_clip=1` (DiffAE default).
- **Warm-up**: use linear warm-up for sensitive hyperparams.
- **Monitor**: log moving average loss per epoch.
