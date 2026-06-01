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

## 2. Commit (code + frozen results)

```bash
git add README.md PIPELINE.md .gitignore GITHUB_SETUP.md
git add stage_a/ stage_b/ stage_c/
git add results/
git add data/README.md data/splits/.gitkeep
git add utils/scripts/README.md
# Do NOT: git add data/splits/*.csv outputs/ latent_data/ *.pth *.npz
git status
git commit -m "Add canonical Stage C results (clinical_2) and thesis reproduce script"
git push -u origin main
```

Suggested tag after Stage C freeze:

```bash
git tag -a v0.3-stage-c -m "Stage C eval + clinical_2 main results"
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
- Stage B NPZ directory layout doc (or one de-identified sample NPZ)
- `stage_b/logs/*.out` showing `--train_csv` / `--val_csv` only

Link the release from README **Reproducibility assets**.

## 5. Repo name

Suggested: `ae-tts-pipeline` or `cardiac-ae-spatial-v2`.

Topics: `medical-imaging`, `autoencoder`, `ct`, `representation-learning`

## 6. What stays outside repo

- `_archive/` — `plain_AE_stageA/`, old Stage B, `diffae-master/`
- Full local monolith (`Report/`, `Dissertation/`, `Presentation/`, `figures/`, patient CSVs) — private only
- Exploratory `utils/scripts/` sweeps — optional; canonical path is `stage_c/` + `results/`

## 7. Verify frozen results table

```bash
python3 stage_c/analysis/build_benchmark_comparison_table.py \
  --manifest results/benchmark_manifest.json \
  --out_csv results/benchmark_comparison_table.csv
```

Expected main rows: Plain 0.874, MONAI 0.796, ensemble 0.891, +clinical 0.915 (ROC mean).
