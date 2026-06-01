# GitHub publish checklist

Use this after copying `stageA` / `stageB_spatial_v2` from HPC into `stage_a` / `stage_b`.

## 1. One-time layout

```bash
export AE_TTS_ROOT="$(pwd)"
# On HPC (example):
# cp -a stageA/plain_ae stage_a/plain_ae
# cp -a stageA/monai_ae stage_a/monai_ae
# cp -a stageB_spatial_v2/plain_ae stage_b/plain_ae
# cp -a stageB_spatial_v2/monai_ae stage_b/monai_ae

mkdir -p data/splits outputs
# Copy CSVs locally only — never git add:
# cp /secure/path/train.csv data/splits/
```

## 2. Commit (code only)

```bash
git add README.md PIPELINE.md .gitignore GITHUB_SETUP.md
git add stage_a/ stage_b/ stage_c/
git add data/README.md data/splits/.gitkeep
git add utils/scripts/README.md
# Optional legacy: git add utils/scripts/   (skip for a clean public repo)
# Do NOT: git add data/splits/*.csv outputs/ latent_data/ *.pth *.npz
git status
git commit -m "Add stage_a/b/c pipeline and downstream eval"
git push -u origin main
```

Suggested tag after Stage C freeze:

```bash
git tag -a v0.3-stage-c -m "Stage C repeated eval and ensemble"
git push origin v0.3-stage-c
```

## 3. Tags

```bash
git tag -a v0.1-stage-a -m "Stage A training scripts"
git tag -a v0.2-stage-b-spatial-v2 -m "Stage B spatial v2 NPZ pipeline"
```

## 4. Release assets (not in git)

Upload to GitHub Release or Zenodo:

- `plain_ae checkpoint` + SHA256
- `monai_ae checkpoint` + SHA256
- Sample NPZ schema v2 (one de-identified patient optional)
- `stage_b/logs/*.out` showing `--train_csv` / `--val_csv` only

## 5. Repo name

Suggested: `ae-tts-pipeline` or `cardiac-ae-spatial-v2`.

Topics: `medical-imaging`, `autoencoder`, `ct`, `representation-learning`

## 6. What stays outside repo

- `_archive/` — `plain_AE_stageA/`, old Stage B, `diffae-master/`
- Full `AE_TTS/` monolith (Report, Dissertation, latent_data, figures) — separate private repo or omit
