# Data layout (not in git)

Patient imaging and split tables **must be provided locally**. Nothing under `data/splits/` is committed.

## ROI NIfTI (external)

Typical layout on a mounted volume:

```text
/Volumes/<project>/DatasetRaw/TTS_RM_V2/<patient_id>/<patient_id>_roi.nii.gz
/Volumes/<project>/DatasetRaw/normal_RM_V2/<patient_id>/<patient_id>_roi.nii.gz
```

Metadata CSV with age, sex, label (e.g. `Combined_Labels.csv`) — also external.

## Split CSVs (local only)

Create:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

**Minimum columns**

| Column | Description |
|--------|-------------|
| `patient_id` | Folder / NPZ stem ID |
| `case` or `label` | 0 = control/normal, 1 = TTS |
| `age` | Years |
| `sex` | `F` / `M` |

Example row (do not commit real IDs to a public repo if policy requires de-identification):

```csv
patient_id,case,age,sex
EXAMPLE_001,1,72,F
```

## Point scripts at splits

```bash
export AE_TTS_ROOT="$(git rev-parse --show-toplevel)"
export SPLIT_DIR="${AE_TTS_ROOT}/data/splits"

# Stage A
--train_csv "${SPLIT_DIR}/train.csv" \
--val_csv   "${SPLIT_DIR}/val.csv"

# Stage B / C
--labels_csv "${SPLIT_DIR}/train.csv" \
--labels_csv "${SPLIT_DIR}/val.csv" \
--labels_csv "${SPLIT_DIR}/test.csv"
```

## Stage A fit policy

- **train + val** → AE optimization  
- **test** → excluded from AE and from k-means centroid fit (assign-only in Stage B)

## Template

Copy from your institution’s split generator; do not add real files to git.  
Optional: keep a private fork or `.git/info/exclude` for local CSV copies.
