# 전이학습 분류 후보 비교·선택·복원 실행 결과 보고서

> **프로젝트:** Day 14 심화 day14_vision (실습07~08) — 4개 후보 비교, 규칙 기반 선택, 모델 저장과 새 프로세스 복원  
> **문서 유형:** 기술 검증 결과 요약 / 교육용 실습 평가  
> **작성일:** 2026-11-04  
> **판정:** **기능 요구사항 충족(PASS) / 선택 head·sampler, 산출물 expected와 일치, 복원 3항목 True. 후보 간 성능 우열은 valid 포화로 미확정, ImageNet 경로 미검증**  
> **평가 대상:** `day14_vision_주석/main.py`(TODO D07~D09), `restore.py`(TODO D10), `outputs/` 전체, `reports/model_card.md`

---

## 1. Executive Summary

본 프로젝트의 목적은 Day 14 전이학습 실습을 하나로 묶어, **같은 8 epoch 예산의 후보들을 valid만으로 비교**하고, **미리 정한 규칙으로 선택**한 모델을 **클래스 매핑·입력 계약과 함께 저장**한 뒤 **새 프로세스에서 같은 예측을 재현**하는 절차를 익히는 것이다. 이를 위해 TODO D07에서 후보 4개(head·none, partial·none, head·weight, head·sampler)를 정의하고, D08에서 선택 키 `(macro_f1, -loss)`, D09에서 `torch.save`, D10에서 `load_state_dict`를 완성하였다.

**실제 실행 결과 기준**, 네 후보 모두 valid Accuracy·Macro F1·클래스별 Recall이 **1.0**이었고, valid loss는 0.012641 / 0.008148 / 0.006550 / **0.006475**였다. 규칙에 따라 loss가 가장 작은 **head·sampler**가 선택되었다(test 사용 False). 혼동행렬은 대각선 30/30/30으로 **오답 0건**이며, `restore.py`의 **ID 순서·확률·예측 일치가 모두 True**였다. metrics·history·predictions·confusion·cam_index·cam_correct.png가 expected와 일치했다.

**README 요구사항 요약 (2.3절 상세):**
- **선택 규칙:** Macro F1 동률(1.0)이라 valid loss 최소인 head·sampler를 골랐다. 2위와 차이는 0.000075다.
- **weight·sampler 분리:** 별개 후보로 실행했고, sampler의 클래스별 추출 수를 history에 기록했다.
- **저장·복원:** state_dict·class_to_idx·입력 계약을 함께 저장했고 새 프로세스 복원 3항목이 True였다.
- **오답 0건 보고:** 오류 이미지를 만들지 않았고, 오류 분석은 실습05 초기 모델과 구분한다.
- **모델 출처:** 소형 합성 source이며 ImageNet ResNet18 경로는 미실행·미검증이다.

**최종 판단:** 후보 비교·선택·저장·복원의 실습 완료 조건을 충족했다(PASS). 다만 모든 후보가 valid 1.0으로 포화되어 **선택은 loss의 작은 차이에 의존**했고, 같은 valid로 반복 선택했으므로 성능은 낙관적 추정이다. test 평가와 ImageNet 경로는 수행하지 않았다.

## 2. 목적 및 검증 범위

### 2.1 검증 목적

- 같은 seed·분할·8 epoch 예산으로 네 후보를 공정하게 비교한다.
- test를 보지 않고 Macro F1 → loss → 후보 순서 규칙으로 선택한다.
- 선택 모델을 클래스 매핑·입력 계약과 함께 저장하고, 새 프로세스에서 같은 확률·예측을 재현한다.
- 실제 오답이 있을 때만 오류 이미지를 만들고, CAM의 출처·target을 기록한다.

### 2.2 검증 범위

| 구분 | 포함 | 제외 |
|:--|:--|:--|
| 입력 | 자체 포함 `data/` train 180장(120/40/20), valid 90장(30/30/30) | test 30장 (클래스 매핑·중복 감사 외 사용 안 함) |
| 후보 | head·none, partial·none, head·weight, head·sampler (8 epoch, seed 14, BN 통계 고정) | scratch 보조 후보, full 전략 |
| 산출 | metrics.csv/json, history.csv, predictions.csv, confusion.csv, cam_index.json, cam_correct.png, best_model.pt | error_*.png (오답 0건) |
| 검증 | expected 대조, restore 3항목 | `resnet18_official.py` (공식 가중치 필요, 미실행) |

### 2.3 README 요구사항 답변

#### Q1. 선택 규칙(Macro F1 최대 → valid loss 최소 → 후보 순서)을 어떻게 구현했고 무엇이 선택되었는가?

선택 키를 `key=(summary["macro_f1"], -summary["loss"])`로 두고 `key > best_key`일 때만 교체했다. 튜플은 앞에서부터 비교하므로 Macro F1이 먼저, 동률이면 −loss가 큰(=loss가 작은) 후보가 이긴다. 엄격한 `>` 비교라 완전 동률이면 먼저 나온 후보가 유지되어 세 번째 규칙(후보 순서)도 충족된다. 네 후보 모두 Macro F1 1.0이었고 loss 최소인 **head·sampler(0.006475)**가 선택되었다. 2위 head·weight(0.006550)와의 차이는 0.000075로, 단일 seed에서 두 후보의 우열이 확정적이라고 보지 않는다. 원본 자리표시(`(accuracy,)`)를 쓰면 모두 1.0 동률이라 첫 후보 head·none이 선택된다.

#### Q2. Sampler와 Class weight를 별개 후보로 두었는가? 두 손실을 같은 척도로 비교하지 않았는가?

그렇다. 두 방법은 서로 다른 후보 행이며 동시 적용하지 않았다. `history.csv`의 head_sampler 행에는 epoch별 추출 수(예: 1 epoch 57/60/63)가, 나머지 후보에는 120/40/20이 기록되었다. weight 후보의 `train_objective`는 가중 손실이므로 비가중 `valid_loss`와 같은 척도로 직접 비교하지 않았다. 후보 간 비교와 선택에는 모두 같은 비가중 valid loss를 사용했다.

#### Q3. 저장 모델에 무엇을 넣었고, 새 프로세스 복원은 성공했는가?

`best_model.pt`(14,201 bytes)에 다음을 저장했다.

| 키 | 내용 |
|:--|:--|
| model | 선택 후보 state_dict (CPU 텐서) |
| class_to_idx | `{'circle': 0, 'square': 1, 'triangle': 2}` |
| input | RGB, size [32,32], Resize((32,32)) → ToTensor(), float [0,1], 정규화 없음 |
| architecture | TinyShapeCNN_3-8-16_GAP4 (models.build_model) |
| selected | 선택 후보 요약 dict |
| test_used | False |

`restore.py`는 `weights_only=True`로 읽고 `model.load_state_dict(saved["model"])`로 복원했다. 실행 결과 **ID 순서 True, 확률 일치 True(atol 1e-6), 예측 일치 True**였다.

#### Q4. 오답이 0건이면 어떻게 보고하는가?

혼동행렬이 대각선 30/30/30으로 **오답 0건**이다. `save_explanations()`는 `error_*.png`와 `cam_wrong.png`를 만들지 않았고 `cam_index.json`에는 correct 1건(circle_000, target 0)만 기록되었다. 오류 분석은 실습05의 **초기 1 epoch 모델(오답 4건)** 결과로 따로 제출하며, 최종 모델 결과와 섞지 않는다.

#### Q5. 이 모델은 ImageNet ResNet18 전이학습 결과인가?

아니다. Backbone은 별도 합성 이미지 240장으로 학습한 TinyShapeCNN이며(`source_provenance.json`: imagenet false, target_used false), 무작위 가중치도 아니다. `resnet18_official.py`는 공식 가중치 다운로드가 필요하고 GPU 사용 코드가 포함된 선택 경로라 이 환경에서 실행하지 않았다. ImageNet 전이학습 성능은 **미검증**이며, 두 경로의 수치를 한 비교표에 섞지 않는다.

## 3. 처리 결과 상세

### 3.1 처리 흐름

| 순서 | 처리 | 적용 설정 | 기대 기능 |
|:--:|:--|:--|:--|
| 1 | 후보 정의 (D07) | 4개 (strategy, balance) | 동일 예산 비교 |
| 2 | 후보 학습 | `experiment(..., epochs=8)` | 후보별 best epoch 모델 |
| 3 | 선택 (D08) | `(macro_f1, -loss)` 최대 | 규칙 기반 선택 |
| 4 | 결과 저장 | metrics·history·predictions·confusion·metrics.json | 근거 기록 |
| 5 | 설명 | `save_explanations()`, cam_index.json | 실제 오답·CAM |
| 6 | 모델 저장 (D09) | `torch.save({...})` | 가중치+계약 |
| 7 | 복원 (D10) | 새 프로세스 `load_state_dict` | 재현 확인 |

**TODO 정답:**
- D07 `experiments=[("head","none"),("partial","none"),("head","weight"),("head","sampler")]`
- D08 `key=(summary["macro_f1"],-summary["loss"])`
- D09 `torch.save({"model":chosen_state,"class_to_idx":..., "input":{...}, "architecture":..., "selected":chosen, "test_used":False}, OUT/"best_model.pt")` (class_to_idx는 `engine.datasets(DATA)[0].class_to_idx`)
- D10 `model.load_state_dict(saved["model"])`

TODO 외 코드는 원본을 유지했다. 제공 모듈(engine·models·explain·resnet18_official)은 주석만 추가했다.

### 3.2 후보 비교 (`outputs/metrics.csv`)

| 후보 | accuracy | macro_f1 | triangle_recall | valid loss | best_epoch | trainable | seconds(참고) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| head·none | 1.0 | 1.0 | 1.0 | 0.012641 | 8 | 771 | 38.48 |
| partial·none | 1.0 | 1.0 | 1.0 | 0.008148 | 8 | 1,955 | 2.52 |
| head·weight | 1.0 | 1.0 | 1.0 | 0.006550 | 8 | 771 | 2.67 |
| **head·sampler (선택)** | 1.0 | 1.0 | 1.0 | **0.006475** | 8 | 771 | 2.80 |

**주의:** head·none의 38.48초는 첫 실행에서 네트워크 마운트의 이미지를 처음 읽는 비용이 섞인 값으로 보인다. 시간은 단일 실행 참고값이며 순위에 쓰지 않는다. 후보별 수치는 실습03·04의 같은 조건 결과와 일치한다.

### 3.3 선택 모델 결과

- **혼동행렬 (`confusion.csv`, 행=실제, 열=예측):** [[30,0,0],[0,30,0],[0,0,30]]
- **클래스별 리포트 (`metrics.json`):** circle·square·triangle 모두 precision·recall·F1 1.0, support 30
- **CAM (`cam_index.json`, `cam_correct.png`):** valid/circle/circle_000.png, target_class 0. 원 아래쪽 경계에 반응이 크다(관찰만 기록).
- **Model Card:** `day14_vision_주석/reports/model_card.md`에 데이터·조건·선택 근거·오류·한계·재현을 실측으로 작성했다.

### 3.4 표준 출력 (`outputs/stdout.txt`, `outputs/restore_stdout.txt`)

```text
 accuracy  macro_f1  circle_recall  square_recall  triangle_recall   loss strategy balance  best_epoch  trainable  epochs           source
      1.0       1.0            1.0            1.0              1.0 0.0126     head    none           8        771       8 synthetic_source
      1.0       1.0            1.0            1.0              1.0 0.0081  partial    none           8       1955       8 synthetic_source
      1.0       1.0            1.0            1.0              1.0 0.0066     head  weight           8        771       8 synthetic_source
      1.0       1.0            1.0            1.0              1.0 0.0065     head sampler           8        771       8 synthetic_source
선택: head sampler test 사용: False
ID 순서: True
확률 일치: True
예측 일치: True
```

## 4. 품질 검증 및 합격 기준

### 4.1 합격 기준 대비 결과

| 기준 | 목표 | 실제 결과 | 판정 |
|:--|:--|:--|:--:|
| 출력표·선택 | README 예상 출력 | 동일 (head sampler, test 사용 False) | PASS |
| metrics.csv | expected와 일치 | 4행, 수치 최대 차 1.1e-08 (seconds 제외) | PASS |
| history.csv | expected와 일치 | 32행, 최대 차 3.4e-08 | PASS |
| predictions.csv | expected와 일치 | 90행, 확률 최대 차 1.5e-07 | PASS |
| confusion·cam_index·metrics.json report | expected와 일치 | 동일 | PASS |
| cam_correct.png | expected와 일치 | 픽셀 최대 차 0 | PASS |
| 복원 | 3항목 True | True / True / True | PASS |
| test 미사용 | False | test_used False | PASS |
| 후보 우열 | — | valid 1.0 포화 | 미평가 |
| ImageNet 경로 | — | 미실행 | 미검증 |

### 4.2 기능 검증 (Acceptance Test)

| TC ID | 검증 내용 | 관찰 근거 | 결과 |
|:--|:--|:--|:--:|
| TC-01 | 후보 4개 동일 예산 | metrics epochs 열 모두 8 | PASS |
| TC-02 | 선택 규칙 | F1 동률 → loss 최소 후보 선택 | PASS |
| TC-03 | 저장 계약 | best_model.pt 키 6개, weights_only 로드 성공 | PASS |
| TC-04 | 새 프로세스 재현 | restore 3항목 True | PASS |
| TC-05 | 오답 0건 처리 | error_*.png·cam_wrong.png 없음 | PASS |
| TC-06 | sampler 기록 | history head_sampler draw 열 | PASS |

## 5. 리스크 및 개선 과제

| 우선순위 | 이슈 / 잠재 영향 | 권고 조치 | 검증 산출물 |
|:--:|:--|:--|:--|
| P0 | 같은 valid로 4후보 반복 선택 → valid 1.0은 낙관적 | 선택 후 test 1회 평가, 또는 별도 holdout | test 지표 |
| P1 | 모든 후보 valid 포화 → loss 0.000075 차이로 선택 | seed 반복·더 어려운 valid로 재비교 | 후보별 평균·분산 |
| P1 | 저장 계약이 문자열 기록 → 복원 시 자동 검증 없음 | restore에서 class_to_idx·입력 크기 일치 확인 | 계약 검증 로그 |
| P1 | ImageNet ResNet18 경로 미검증 | 공식 가중치 준비 후 head·partial 실행 | outputs/imagenet 결과 |
| P2 | scratch 보조 후보 미실행 | pretrained=False 후보를 별도 표로 추가 | 사전학습 효과 비교 |
| P2 | 첫 실행 시간이 비교 불가 | 워밍업 후 반복 측정 | 시간 중앙값 |

## 6. 다음 단계: 현업 적용 승인 전 검증 계획

| 단계 | 검증 항목 | 방법 | 통과 기준 |
|:--:|:--|:--|:--|
| 1 | 최종 성능 확인 | 선택 모델로 test 30장 1회 평가 | 사전 정의 Macro F1 기준 충족 |
| 2 | 안정성 | seed 5개 × 후보 4개 반복 | 선택 결과의 일관성 보고 |
| 3 | 실제 source 비교 | ImageNet ResNet18 head·partial 실행 | 같은 조건 표에서 분리 보고 |
| 4 | 배포 재현 | 다른 장치·OS에서 restore | 3항목 True |
| 5 | 운영 데이터 | 실제 촬영 이미지로 오류 분석 | 오류 유형별 대응 계획 |

## 7. 최종 판단

- **기능 검증:** TODO D07~D10을 완성해 네 후보 비교, 규칙 기반 선택(head·sampler), 계약 포함 저장, 새 프로세스 복원(3항목 True)을 수행했고 산출물이 expected와 일치했다.
- **핵심 관찰:** 모든 후보가 valid 1.0이라 선택은 valid loss의 작은 차이로 결정되었다. 최종 모델 오답은 0건이다.
- **한계:** valid 반복 선택에 따른 낙관적 추정, 단일 seed, test 미평가, ImageNet 경로 미검증이다.
- **권고:** 본 프로젝트는 **완료(PASS) 및 후속 검증 진행**으로 분류한다. 운영 판단 전에 test 1회 평가와 seed 반복 비교를 먼저 수행한다.

---

## 부록 A. 근거 파일 및 검증 이력

### A.1 근거 파일

| 파일 | 역할 |
|:--|:--|
| `../day14_vision_주석/main.py` | TODO D07·D08·D09 완성 + 전체 주석 |
| `../day14_vision_주석/restore.py` | TODO D10 완성 + 전체 주석 |
| `../day14_vision_주석/engine.py`, `models.py`, `explain.py`, `resnet18_official.py` | 제공 모듈 + 주석 (코드 변경 없음) |
| `../day14_vision_주석/outputs/metrics.csv`, `metrics.json`, `history.csv`, `predictions.csv`, `confusion.csv`, `cam_index.json`, `cam_correct.png`, `best_model.pt` | **실제 실행 산출물** |
| `../day14_vision_주석/outputs/stdout.txt`, `restore_stdout.txt` | **실제 실행 표준 출력** |
| `../day14_vision_주석/reports/model_card.md` | 실측 기반 Model Card |
| `expected/` | 과제 제공 참고 결과 |
| `day14_vision.ipynb` | 단계별 학습 노트북 (결과: `outputs_notebook/`) |
| `data/source_provenance.json` | source Backbone 출처 기록 |

### A.2 검증 이력

- **검증 방식:** 주석본 `main.py`·`restore.py`를 CPU 1-thread로 실제 실행하고 CSV는 열별(seconds 제외), JSON은 내용, PNG는 픽셀 단위로 expected와 비교했다. 주석본 `data/source_backbone.npz`는 원본과 바이트가 같다.
- **미실행:** `resnet18_official.py` (공식 ImageNet 가중치 다운로드 필요, CUDA 사용 코드 포함 — 프로젝트 방침상 실행하지 않고 논리만 주석).
- **재현 명령:** `cd day14_vision_주석 && python main.py && python restore.py`
- **보고 원칙:** **측정된 사실**, **그 사실에서 계산한 값**, **아직 검증되지 않은 가정**을 구분하여 작성했다.

*교육용 실습 결과 보고서 — 2026-11-04 실제 실행 산출물과 `expected/`, `README.md`를 근거로 작성했으며 운영 승인 문서가 아니다.*
