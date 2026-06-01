# `repeated_summary_merged_with_fixed.csv` — 반복 분할 평가 결과 통합표

이 문서는 **고정 train/val/test 분할에서 얻은 AUC**와 **반복적인 stratified hold-out 분포**를 한 표에서 비교하기 위한 산출물인 `repeated_summary_merged_with_fixed.csv`의 의미, 생성 절차, 해석 방법을 정리합니다. 이후 실험을 반복할 때 동일한 프로토콜과 산출물 이름을 재사용하면 보고·비교가 일관됩니다.

---

## 1. 왜 이 파일이 필요한가

- 고정 테스트 셋 한 번의 ROC-AUC / PR-AUC는 표본이 작을 때 **특정 분할에 운 좋게 맞은 값**일 수 있습니다.
- `repeated_stratified_shuffle_eval.py`는 **전체 라벨이 있는 환자 풀**에서 stratified로 train/test를 반복 샘플링하고, 매번 train에만 `StandardScaler`를 맞춘 뒤 test에서만 평가합니다. 이렇게 얻은 **100회(기본값) AUC 분포**는 “같은 모델·같은 특징이라도 분할에 따라 성능이 얼마나 흔들리는지”를 보여 줍니다.
- `repeated_summary_merged_with_fixed.csv`는  
  - 반복 분할에 대한 **요약 통계(평균, 표준편차, 분위수 등)** 와  
  - k-fold 단계에서 고른 **안정 설정**에 대한 **고정 테스트 AUC**가 그 분포 안에서 **어느 정도 위치(경험적 CDF)**에 있는지  
  를 한 행에 묶어 줍니다.

---

## 2. 실험 파이프라인 (참고)

대략적인 순서는 다음과 같습니다.

1. **특징**: Stage B `patients/*.npz`에서 `mean` / `std` / `mean_std` / `cluster_hist` 등 pooling 방식으로 환자 단위 벡터를 만듭니다.
2. **고정 분할에서의 안정 설정**: `latent_kfold_cv.py`의 `--eval_stable_min_std` 등으로 `test_metrics_stable_minstd.csv`에 (latent_source, pooling, model) 조합과 고정 test ROC/PR이 기록됩니다.
3. **반복 분할 평가**: `run_repeated_from_stable_settings.py`가 위 CSV의 각 행에 대해 `repeated_stratified_shuffle_eval.py`를 호출합니다.
4. **통합표**: 각 `run_<타임스탬프>_...` 폴더 아래에 생성된 `summary.csv`와 `compare_fixed_test_positions.csv`를 병합해 `repeated_summary_merged_with_fixed.csv`를 만듭니다 (아래 §5).

프로토콜 상세는 `repeated_stratified_shuffle_eval.py` 상단 docstring과 `run_repeated_from_stable_settings.py` 주석을 따릅니다.

**기본 설정(요약)**

| 항목 | 기본값 |
|------|--------|
| 분할기 | `StratifiedShuffleSplit` |
| `n_splits` | 100 |
| `test_size` | 0.2 |
| `random_state` | 42 |
| 스케일러 | `StandardScaler` (train에만 fit) |
| 분류기 | `LogisticRegression`, `LinearSVC`, `RBF SVC(probability=True)` — `--classifier`로 하나만 쓸 수 있음 |

---

## 3. 한 번의 배치 실행에서 생기는 디렉터리 구조

`run_repeated_from_stable_settings.py`는 설정 CSV의 **각 행마다** `repeated_stratified_shuffle_eval.py`를 한 번씩 호출합니다. 배치 시작 시 찍힌 타임스탬프 `ts`(예: `20260506_171300`)가 모든 행에 공통이고, `run_tag`는 `"{ts}_{latent_source}_{pooling}_{model}"` 형태입니다.

그래서 `out_repeated_from_stable` 아래에는 **같은 `ts`로 시작하는 `run_*` 폴더가 여러 개** 생깁니다(설정 행 수만큼). 각 폴더 안 구조는 대략 다음과 같습니다.

```text
out_repeated_from_stable/
  run_20260506_171300_diffae_mean_std_linear_svc/
    compare_fixed_test_positions.csv    # 이 설정 한 줄에 대한 고정 테스트 vs 분포 (옵션)
    diffae/
      mean_std/
        linear_svc/
          repeats.csv
          summary.csv
          hist_roc_auc_linear_svc.png
          hist_pr_auc_linear_svc.png
  run_20260506_171300_plain_ae_std_logistic/
    ...
```

통합표 `repeated_summary_merged_with_fixed.csv`를 만들 때는 보통 **한 번의 배치 = 같은 `ts` 접두사를 가진 모든 `run_*` 디렉터리**에서 `summary.csv`와 `compare_fixed_test_positions.csv`를 모아 합칩니다.

---

## 4. 열 정의 — `repeated_summary_merged_with_fixed.csv`

한 행은 **(latent_source, pooling, model)** 조합 하나에 대응합니다. 열은 대개 다음과 같이 해석합니다.

### 4.1 식별 열

| 열 | 의미 |
|----|------|
| `latent_source` | 잠재 벡터 출처(예: `plain_ae`, `diffae`, `monai_ae`). |
| `pooling` | 환자 수준 특징 종류: `mean`, `std`, `mean_std`, `cluster_hist`. |
| `model` | `logistic`, `linear_svc`, `rbf_svc` 중 실제로 평가된 분류기. |

### 4.2 고정 테스트 분할 대비 (“fixed”) 열

`--fixed_test_csv`(예: `test_metrics_stable_minstd.csv`)를 넘겼을 때만 의미가 있습니다.

| 열 | 의미 |
|----|------|
| `fixed_test_roc_auc` | 고정 hold-out 테스트에서 측정한 ROC-AUC (참조값). |
| `fixed_test_pr_auc` | 고정 hold-out 테스트에서 측정한 PR-AUC (Average Precision). |
| `cdf_position` | 100회 반복에서 나온 **test ROC-AUC**들에 대해, `mean(roc_auc <= fixed_test_roc_auc)`. 즉 **경험적 누적분포에서 고정 값이 놓인 위치**(0~1). 값이 크면 고정 테스트 ROC가 반복 분할에서 나오는 값들보다 **대체로 크다**(상대적으로 낙관적)는 뜻에 가깝습니다. |
| `pr_cdf_position` | PR-AUC에 대해 동일하게 `mean(pr_auc <= fixed_test_pr_auc)`. |

### 4.3 반복 분할 분포 요약 (같은 조합·같은 분류기에 대해 `repeats.csv`의 test 지표로 계산)

| 열 | 의미 |
|----|------|
| `roc_mean`, `roc_std` | 반복 test ROC-AUC의 평균, 표준편차. |
| `roc_median`, `roc_iqr` | 중앙값, IQR(Q3−Q1). |
| `roc_min`, `roc_max` | 최소·최대. |
| `roc_p2p5`, `roc_p97p5` | 2.5%, 97.5% 분위수. |
| `pr_mean`, `pr_std`, … | PR-AUC에 대한 동일 통계. |

**주의**: `roc_std`는 **반복 분할 간 변동**을 말하며, k-fold CV fold 간 표준편차와는 다른 실험입니다.

---

## 5. 통합 CSV 만드는 방법

저장소에는 “모든 `run_*`를 자동 순회하는” 단일 진입 스크립트가 없을 수 있으므로, **한 번의 배치(`run_repeated_from_stable_settings.py`)가 끝난 뒤** 같은 타임스탬프 접두사의 폴더들을 한꺼번에 모아 병합하는 방식을 권장합니다.

### 5.1 `compare_fixed_test_positions.csv` 정리

스크립트는 ROC용 행과 PR용 행을 따로 쌓을 수 있어, 동일 키 `(latent_source, pooling, model)`에 대해 한 행으로 합치는 것이 안전합니다.

### 5.2 예시: 같은 배치 타임스탬프의 모든 `run_*` 폴더에서 병합

`BATCH_TS`만 이번에 터미널에 찍힌 값(또는 폴더 이름에서 읽은 `20260506_171300` 형태)으로 바꿉니다.

```python
import pandas as pd
from pathlib import Path

out_parent = Path("/Volumes/Chanho_PhD_Project/latent_data/out_repeated_from_stable")
BATCH_TS = "20260506_171300"  # 배치 공통 타임스탬프

run_dirs = sorted(out_parent.glob(f"run_{BATCH_TS}_*"))
if not run_dirs:
    raise SystemExit(f"No dirs matching run_{BATCH_TS}_* under {out_parent}")

summaries, fixed_parts = [], []
for run_dir in run_dirs:
    for p in run_dir.rglob("summary.csv"):
        summaries.append(pd.read_csv(p))
    fx = run_dir / "compare_fixed_test_positions.csv"
    if fx.exists():
        fixed_parts.append(pd.read_csv(fx))

if not summaries:
    raise SystemExit("No summary.csv found under batch run dirs.")
df_sum = pd.concat(summaries, ignore_index=True)

if not fixed_parts:
    merged = df_sum.copy()
    print("Warning: no compare_fixed_test_positions.csv; merged file will lack fixed_* and cdf columns.")
else:
    df_fx = pd.concat(fixed_parts, ignore_index=True)
    keys = ["latent_source", "pooling", "model"]
    roc_part = df_fx.dropna(subset=["fixed_test_roc_auc"])[keys + ["fixed_test_roc_auc", "cdf_position"]].drop_duplicates(keys)
    pr_part = df_fx.dropna(subset=["fixed_test_pr_auc"])[keys + ["fixed_test_pr_auc", "pr_cdf_position"]].drop_duplicates(keys)
    df_fixed = roc_part.merge(pr_part, on=keys, how="outer")
    merged = df_sum.merge(df_fixed, on=keys, how="left")

out = out_parent / f"repeated_summary_merged_with_fixed_{BATCH_TS}.csv"
merged.to_csv(out, index=False)
print("Wrote:", out)
```

고정 파일명 `repeated_summary_merged_with_fixed.csv`로 덮어쓰려면 `out`을 `out_parent / "repeated_summary_merged_with_fixed.csv"`로 두면 됩니다. 여러 배치를 한 파일에 쌓을 때는 **배치 열을 추가**하거나 파일명에 타임스탬프를 붙여 충돌을 피하는 것이 좋습니다.

### 5.3 품질 점검

- 병합 후 `latent_source`가 비어 있거나 숫자 열에 문자열이 섞인 행이 있으면 CSV 편집 오류일 수 있으니 제거합니다.
- `cdf_position`이 0 또는 1에 매우 가깝다면, 고정 테스트 AUC가 반복 분포의 **극단**에 해당한다는 뜻이므로 해석 시 주의합니다.

---

## 6. 결과 읽는 법 (짧게)

- **반복 평균 vs 고정 테스트**: `roc_mean`과 `fixed_test_roc_auc`를 비교하면, 고정 분할이 “전형적인” 무작위 80/20 stratified 분할보다 좋은지 나쁜지 감을 잡을 수 있습니다.
- **CDF 위치**: `cdf_position` ≈ 0.9이면, 반복에서 나온 ROC-AUC의 약 90%가 고정 테스트 ROC보다 작거나 같다는 뜻입니다(고정 값이 상대적으로 높음).
- **분산**: `roc_std`, `roc_iqr`이 크면 표본 크기·클래스 불균형·특징 안정성 문제로 분할에 민감할 수 있습니다.

---

## 7. 관련 파일 (코드)

| 파일 | 역할 |
|------|------|
| `utils/scripts/repeated_stratified_shuffle_eval.py` | 반복 StratifiedShuffleSplit 평가, `summary.csv`, `compare_fixed_test_positions.csv` 생성. |
| `utils/scripts/run_repeated_from_stable_settings.py` | `test_metrics_stable_minstd.csv`의 각 행에 대해 위 스크립트 배치 실행. |
| `utils/scripts/latent_kfold_cv.py` | 고정 분할·k-fold 및 `test_metrics_stable_minstd.csv` 등 생성. |

---

## 8. 경로 관습

외장 볼륨 예시는 `/Volumes/Chanho_PhD_Project/latent_data/out_repeated_from_stable/`처럼 쓰였습니다. 로컬만 쓸 경우에도 **동일한 파일 이름**(`repeated_summary_merged_with_fixed.csv`)을 유지하면 논문·보조자료에서 표를 갱신하기 쉽습니다.
