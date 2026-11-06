# 전이학습 프로젝트·재현 확인 실행 결과 보고서

> **프로젝트:** Day 14 전이학습 응용 실습 07(+08 재현) — Head 기준선·Partial·Head+weight 세 후보 비교  
> **문서 유형:** 기술 검증 결과 요약 / 교육용 실습 평가  
> **작성일:** 2026-11-04  
> **판정:** **기능 요구사항 충족(PASS) / README 정답 실행 예 4줄 일치, 재실행 시 metrics·predictions·history·confusion 동일. test 성능은 미평가(설계상 미사용)**  
> **평가 대상:** `07_project.py`(TODO A07·A08), `outputs/metrics.csv`, `predictions.csv`, `history.csv`, `confusion.csv`, `project_cam.png`

---

## 1. Executive Summary

본 실습의 목적은 실습 03·04에서 익힌 학습 범위와 불균형 대응을 **하나의 프로젝트로 묶어** 세 후보를 같은 조건에서 비교하고, 예측·혼동행렬·CAM을 저장한 뒤 **다시 실행해 같은 결과가 나오는지(실습 08)** 확인하는 것이다. 이를 위해 TODO A07에서 후보를 `[("head","none"), ("partial","none"), ("head","weight")]`로 고정하고, TODO A08에서 마지막 후보의 valid 예측을 `valid.samples` 순서대로 `file·actual·predicted·confidence` 표로 만들었다.

**실제 실행 결과 기준**, head/none **0.8111 · 0.8038 · 0.5000**, partial/none **0.9556 · 0.9562 · 0.9333**, head/weight **1.0000 · 1.0000 · 1.0000**(accuracy · macro F1 · triangle recall)이었다. `predictions.csv`는 **90행, 오답 0**이고 test는 사용하지 않았다. 오답이 없어 `project_error_*.png`는 만들어지지 않았고, CAM은 valid 0번(circle_000.png)의 예측 클래스 circle에 대해 저장했다. 같은 명령으로 다시 실행하자 seconds를 뺀 모든 CSV가 동일했다.

**README 질문 요약 (2.3절 상세):**
- **재실행 비교를 위해 고정할 것:** 입력 data, source 배열, 변환, 클래스 매핑, seed·스레드, 후보·학습 설정, BN 정책, valid 순서. 이번 재실행에서 이 조건이 같아 결과가 같았다.
- **test를 남겨 두는 이유:** valid는 epoch·후보 선택에 이미 쓰여 낙관적이므로, 선택이 끝난 뒤 한 번만 쓰는 test가 있어야 일반화 성능을 정직하게 보고할 수 있다.

**최종 판단:** 프로젝트 통합과 재현 확인의 실습 완료 조건을 충족했다(PASS). 단, 세 후보 중 head/weight가 valid에서 가장 좋았다는 것은 단일 seed·valid 90장 기준이며 `predictions.csv`는 "마지막 후보"의 예측이지 자동 선택한 최고 모델이 아니다. 시간은 단일 실행 참고값이다.

## 2. 목적 및 검증 범위

### 2.1 검증 목적

- 기준선에서 한 조건씩 바꾼 두 후보를 기준선과 같은 조건에서 비교한다.
- 마지막 후보의 valid 예측을 파일 이름과 함께 순서대로 저장한다.
- 혼동행렬·오답 이미지·CAM을 저장하고, 재실행 시 같은지 확인한다(실습 08).

### 2.2 검증 범위

| 구분 | 포함 | 제외 |
|:--|:--|:--|
| 데이터 | train 180 학습, valid 90 선택·평가 | test(매핑·중복 감사 외 사용 금지) |
| 후보 | head/none, partial/none, head/weight | partial+weight, BN affine 해제 |
| 산출 | metrics·history·predictions·confusion CSV, project_cam.png, 오답 시 project_error_*.png | 모델 파일 저장(응용 요구 없음) |
| 재현 | 같은 명령 2회 실행 후 CSV 비교 | 다른 기기·OS |
| 기준값 | README 실습 07·08 정답 실행 예(참고값) | `expected/`(없음) |

**TODO 정답:**
- A07 `experiments=[("head","none"),("partial","none"),("head","weight")]` — 원본은 기준선 하나뿐이다.
- A08 `result=pd.DataFrame({"file":[Path(path).name for path,_ in valid.samples], "actual":y_valid.tolist(), "predicted":pred.tolist(), "confidence":prob.max(1).values.tolist()})` — 원본은 열 이름만 있는 빈 표라 `예측 행: 0`이 된다. 반복문이 끝난 뒤의 `pred`·`prob`은 마지막 후보(head/weight)의 값이다.

TODO 외 코드는 원본을 유지하고 세미콜론 문장·긴 dict·한 줄 for만 주석을 위해 줄을 나눴다(AST 비교로 로직 동일 확인).

### 2.3 README 질문 답변

#### Q1. 어떤 입력·설정·seed·매핑을 고정해야 재실행 결과를 비교할 수 있는가?

| 고정 대상 | 이 프로젝트의 값 | 바뀌면 생기는 일 |
|:--|:--|:--|
| 입력 데이터 | `data/` train 180·valid 90, 원본 덮어쓰기 없음 | 다른 데이터로 학습·평가 |
| source 배열 | `source_backbone.npz` (sha256 `fdedd0b6…29f4`, provenance와 일치) | 출발점이 다른 전이학습 |
| 입력 변환 | Resize(32,32) + ToTensor, Normalize 없음 | 입력 분포 변화 |
| 클래스 매핑 | `{circle:0, square:1, triangle:2}` | 번호·triangle_recall 의미 변화 |
| seed·스레드 | `torch.manual_seed(14)`(후보마다 재고정), `set_num_threads(1)` | Head 초기값·연산 순서 변화 |
| 후보·학습 설정 | 3후보, AdamW lr 0.003·wd 0.0001, 8 epoch, 전체 1배치 | 비교 기준 변화 |
| 학습 범위·BN | requires_grad(Head / +model[4]), `eval()`로 BN 통계 고정 | 학습 파라미터·BN 통계 변화 |
| 선택 규칙 | valid 손실(가중치 없음) 최소 epoch | 다른 epoch 선택 |
| 출력 순서 | `valid.samples` 순서(파일 이름순) | 행 단위 비교 불가 |

이번 실습 08 재실행에서 위 조건이 같았고, seconds를 뺀 `metrics.csv`와 `predictions.csv`·`history.csv`·`confusion.csv`가 모두 같았다(실측). 시간(seconds)은 고정할 수 없는 값이라 비교에서 뺀다. 라이브러리 버전·장치(README는 torch 2.10, 이번은 2.15)가 바뀌면 마지막 자릿수가 달라질 수 있으므로 버전도 함께 기록한다.

#### Q2. test를 왜 남겨 두는가?

valid는 이미 **epoch 선택**(손실 최소 epoch)과 **후보 비교**(세 후보 중 어느 것이 나은지)에 쓰였다. 선택에 쓴 데이터의 성능은 그 선택에 맞춰진 값이라 실제보다 좋게 보일 수 있다. 이번에도 head/weight가 valid 1.0000이었지만, 이것이 처음 보는 데이터에서도 1.0이라는 근거는 아니다. test를 학습·추론·임계값 선택 어디에도 쓰지 않고 남겨 두어야, 모든 결정을 끝낸 뒤 **한 번만** 평가해 일반화 성능을 정직하게 보고할 수 있다. 이 프로젝트는 test를 열지 않았다(`test 사용: False`).

### 2.4 실습 08 기록표

| 기록 | 작성 |
|:--|:--|
| 사용 모델·source 출처 | TinyShapeCNN(3-8-16, GAP 4×4) + 새 Linear(256,3). source는 합성 도형 240장으로 학습한 Backbone 수치 배열(seed 14, 50 epoch 중 48 선택, ImageNet 아님, Target 미사용) |
| train/valid 클래스 개수 | train 120·40·20, valid 30·30·30 (circle·square·triangle) |
| 학습 범위·BN 정책 | head: Linear 771개 / partial: 마지막 Conv 1,152 + Head = 1,923개. BN은 `eval()`로 running 통계 고정, affine 미해제 |
| 주 지표와 이유 | macro F1(+triangle recall). train이 6:2:1 불균형이라 소수 클래스 실패를 클래스 동일 비중으로 드러내기 위해 |
| 관찰한 오류·CAM target | 최종(head/weight) 모델 valid 오답 0건. CAM은 valid 0번 circle_000.png, target = 예측 클래스 circle. 초기 비교에서 head/none은 triangle 15장→circle 오답(03·04 실측)으로, 최종 모델 결과와 구분 |
| 재실행 확인·한계 1개 | 2회 실행 시 seconds 외 CSV 4종 동일. 한계: 단일 seed·valid 90장 결과이며 test 미평가 |

## 3. 처리 결과 상세

### 3.1 처리 흐름

| 순서 | 처리 | 적용 설정 | 기대 기능 / 효과 |
|:--:|:--|:--|:--|
| 1 | 데이터 Tensor 준비 | (180,3,32,32) / (90,3,32,32) | 같은 입력 |
| 2 | 후보 고정 | 3후보 (TODO A07) | 기준선 + 한 조건씩 변경 |
| 3 | 후보별 학습·선택 | seed 14, 8 epoch, valid 손실 최소 epoch | rows·all_history |
| 4 | 비교 저장 | `metrics.csv`(3행), `history.csv`(24행) | 후보 비교·곡선 |
| 5 | 예측표 | 마지막 후보 pred·prob → `predictions.csv` (TODO A08) | 90행 |
| 6 | 혼동행렬 | `confusion.csv` 3×3 | 오류 유형 |
| 7 | 오답 이미지 | `project_error_*.png` (오답 있을 때) | 이번 0장 |
| 8 | Grad-CAM | 첫 오답 또는 valid 0번, target = 예측 클래스 | `project_cam.png` |

### 3.2 후보별 결과 (`outputs/metrics.csv`)

| strategy | balance | params | best_epoch | accuracy | macro_f1 | triangle_recall | seconds 1회차 / 2회차(참고) |
|:--|:--|--:|--:|--:|--:|--:|:--|
| head | none | 771 | 8 | 0.8111 | 0.8038 | 0.5000 | 0.1924 / 0.2277 |
| partial | none | 1923 | 8 | 0.9556 | 0.9562 | 0.9333 | 0.1892 / 0.2420 |
| head | weight | 771 | 8 | 1.0000 | 1.0000 | 1.0000 | 0.2111 / 0.1995 |

- 기준선 대비 Partial은 학습 범위를, head/weight는 손실 가중치를 바꿨다. 이번 데이터에서는 **파라미터를 늘리지 않은 head/weight**가 triangle을 모두 맞혔다.
- 세 후보 모두 best_epoch=8(마지막)로, valid 손실이 아직 감소 중이었다(`history.csv`: head/none 0.4635, partial 0.3217, head/weight 0.2378).
- seconds는 실행마다 순위가 바뀌었다(1회차 partial이 가장 짧고, 2회차는 head/weight가 가장 짧음). 시간으로 후보 순위를 정하지 않는다.

### 3.3 마지막 후보(head/weight) 예측 (`outputs/predictions.csv`, `confusion.csv`)

| 항목 | 값 |
|:--|:--|
| 행 수 / 순서 | 90행, `valid.samples` 순서(circle_000 → triangle_029) |
| 혼동행렬 | circle 30·0·0 / square 0·30·0 / triangle 0·0·30 |
| 오답 | 0건 → `project_error_*.png` 생성 안 함 |
| confidence 최솟값 | 0.4057 (square_024.png, 정답) |
| 클래스별 confidence 최솟값 / 평균 | circle 0.5321 / 0.8038 · square 0.4057 / 0.7966 · triangle 0.4971 / 0.7962 |

정답이어도 확률 0.5 미만인 사례(square_024 0.4057, square_008 0.4801, triangle_013 0.4971)가 있다. 05 초기 모델에서도 square_024·triangle_013이 가장 낮은 확률이었다(05 보고서). 같은 이미지가 두 모델에서 애매하게 판단된 것은 **관찰**이고, 그 원인(크기·위치 등)은 **확인 전 가설**이다.

### 3.4 Grad-CAM (`outputs/project_cam.png`)

![project CAM](outputs/project_cam.png)

- 오답이 없어 valid 0번(circle_000.png)을 사용했고, target은 예측 클래스 circle이다. mean CAM은 0.1241(노트북 같은 로직 측정)이다.
- 반응은 원의 왼쪽 아래·오른쪽 아래 가장자리에 모였다. 06의 초기 모델 circle 지도와 비슷한 위치다. 이 지도는 circle logit에 대한 반응 지도이며 원인·정답 위치가 아니다.

### 3.5 표준 출력 (`outputs/logs/outputs_07_project.log`, 재실행 `outputs_07_project_rerun.log`)

```text
strategy balance  params  best_epoch  accuracy  macro_f1  triangle_recall
    head    none     771           8    0.8111    0.8038           0.5000
 partial    none    1923           8    0.9556    0.9562           0.9333
    head  weight     771           8    1.0000    1.0000           1.0000
예측 행: 90 오답: 0 test 사용: False
```

README 실습 07 정답 실행 예 4줄과 같고, 재실행 로그도 같다.

## 4. 품질 검증 및 합격 기준

### 4.1 합격 기준 대비 결과

| 항목 | 합격 기준(README) | 실제 실행 결과 | 판정 |
|:--|:--|:--|:--:|
| 비교표 | 3후보 지표 | **3행 모두 정답 예와 동일** | PASS |
| 예측 행 / 오답 / test | 90 / 0 / False | **90 / 0 / False** | PASS |
| 생성 파일 | metrics·predictions·history·confusion CSV, project_cam.png | **모두 생성** | PASS |
| 오답 이미지 | 실제 오답이 있을 때만 | **0장(오답 0건)** | PASS |
| 재현(실습 08) | 행 순서·값 동일 | **seconds 외 CSV 4종 동일** | PASS |
| 원본 data 보존 | 변경 없음 | **원본 300장 sha256 manifest와 일치** | PASS |
| 시간 | 고정하지 않음 | 0.19 ~ 0.24 s | 측정값 |
| test 성능 | 남겨 둠 | 미사용 | 미평가 |

> **주의:** 공유(네트워크) 폴더에서 `project_cam.png`를 두 번 저장하자 파일 바이트(md5)가 달랐다. 같은 코드를 로컬 디스크에서 2회 실행하면 PNG 바이트가 같았으므로(보조 측정) 공유 폴더 저장 과정의 차이로 본다. 재현 비교는 CSV 값과 로컬 실행 기준으로 판단했다.

### 4.2 기능 검증 (Acceptance Test)

| TC ID | 검증 내용 | 관찰 근거 | 결과 |
|:--|:--|:--|:--:|
| TC-01 | 표준 출력 4줄 | 로그 = README 정답 예 | PASS |
| TC-02 | 예측표 구조 | 90행, 열 file·actual·predicted·confidence, valid.samples 순서 | PASS |
| TC-03 | 예측표 = 혼동행렬 | predictions의 actual·predicted 집계 = confusion.csv | PASS |
| TC-04 | 재실행 값 동일 | metrics(seconds 제외)·predictions·history·confusion 동일 | PASS |
| TC-05 | 로컬 재실행 바이트 | seconds가 든 metrics.csv 외 모든 출력 md5 동일 | PASS |
| TC-06 | test 미사용 | 코드가 train·valid만 읽음 | PASS |
| TC-07 | 오답 이미지 경로 | 오답 0건 | 미평가 |

## 5. 리스크 및 개선 과제

| 우선순위 | 이슈 / 잠재 영향 | 개선 과제 | 검증 산출물 |
|:--:|:--|:--|:--|
| P0 | 수업 venv(torch 2.10)에서 결과 확인 안 함 | `python 07_project.py` 2회 실행 | 출력 4줄·CSV 동일 |
| P1 | `predictions.csv`가 "마지막 후보" 고정 → 후보 순서를 바꾸면 다른 모델 예측이 저장됨 | 저장 대상 후보 이름을 파일·열에 기록 | 후보명 포함 CSV |
| P1 | best_epoch=8(마지막) → epoch 부족 가능성 | epoch 늘려 손실 곡선 확인 | history 비교 |
| P1 | 단일 seed·valid 90장으로 후보 결정 | seed 반복, 결정 후 test 1회 평가 | seed별 표, test 보고 |
| P2 | 확률 0.5 미만 정답 사례 존재 | 저확신 사례 목록과 CAM 확인 | 저확신 목록 |
| P2 | 공유 폴더 PNG 바이트 변화 | 바이트 비교는 로컬 디스크에서 수행 | 로컬 md5 로그 |

## 6. 다음 단계: 현업 적용 승인 전 검증 계획

| 단계 | 검증 항목 | 방법 | 통과 조건 |
|:--:|:--|:--|:--|
| 1 | 환경 재현 | 수업 venv·다른 PC에서 2회 실행 | CSV 4종 동일, 버전 기록 |
| 2 | 후보 결정 근거 | seed 5개 × 3후보, epoch 확장 | macro F1 평균·범위로 후보 선택 |
| 3 | 모델 고정 | 선택 후보를 `torch.save`로 저장, 복원은 `weights_only=True` | 복원 후 valid 예측 동일 |
| 4 | 최종 평가 | 고정한 모델로 test 1회 평가 | test 지표·오류 사례 보고 |
| 5 | 저확신·CAM 점검 | 확률 0.6 미만 사례의 지도 검토 | 관찰·가설 분리 기록 |
| 6 | 실제 데이터 | 실사 이미지·공식 ResNet18 경로와 비교 | 같은 기준표로 보고 |

## 7. 최종 판단

- **기능 검증:** TODO A07·A08을 완성해 세 후보 비교·예측표·혼동행렬·CAM을 저장했고, 출력이 README와 같았다.
- **재현성:** 실습 08 재실행에서 seconds 외 결과가 모두 같았고, 원본 data는 바뀌지 않았다.
- **관찰:** valid 기준으로 head/weight가 가장 좋았고 최종 예측의 오답은 0건이었다. 05·06 초기 모델 결과와 구분해 보고했다.
- **판단:** 본 실습은 **완료(PASS)**로 분류한다. 후보를 확정하기 전에 seed 반복과 test 1회 평가를 먼저 수행한다.

---

## 부록 A. 근거 파일 및 검증 이력

### A.1 근거 파일

| 파일 | 역할 |
|:--|:--|
| `07_project.py` | TODO A07·A08 완성 + 전체 주석 |
| `outputs/metrics.csv` | **실제 실행 비교표**(재실행 값, seconds 포함) |
| `outputs/history.csv` | **실제 실행 epoch별 valid 손실** 24행 |
| `outputs/predictions.csv` | **실제 실행 head/weight 예측** 90행 |
| `outputs/confusion.csv` | **실제 실행 혼동행렬** |
| `outputs/project_cam.png` | **실제 실행 Grad-CAM 합성 이미지** |
| `outputs/logs/outputs_07_project.log`, `outputs_07_project_rerun.log` | **1·2회차 표준 출력** |
| `../07_project.ipynb` | 단계별 노트북(실습 08 재현 셀 포함, 결과: `../outputs_notebook/`) |
| `../README.md` | 실습 07·08 정답 실행 예와 설명 질문(참고값) |

### A.2 검증 이력

| 일자 | 내용 | 결과 |
|:--|:--|:--|
| 2026-11-04 | `07_project.py` 1회차 실행, README 정답 예와 대조 | 4줄 일치 |
| 2026-11-04 | 2회차 실행(실습 08), CSV 4종 비교 | seconds 외 동일 |
| 2026-11-04 | 로컬 디스크 복사본으로 2회 실행, 출력 md5 비교 | metrics.csv(seconds)만 다름 |
| 2026-11-04 | 원본 `02_응용/data/` sha256을 manifest와 대조 | 300/300 일치 |
| 2026-11-04 | 노트북 실행: 재현 셀 | metrics·history·predictions·확률 동일 True |


- **재현 명령:** `python 07_project.py` (실습 08: 같은 명령을 한 번 더)
- **보고 원칙:** **측정된 사실**, **그 사실에서 계산한 값**, **아직 검증되지 않은 가정**을 구분하여 작성했다.

*교육용 실습 결과 보고서 — 2026-11-04 실제 실행 산출물(`outputs/`)과 README 실습 07·08을 근거로 작성했으며 운영 승인 문서가 아니다.*
