# Day15 Detection | 학습 기록에서 best epoch 선정·mAP 해석 결과요약보고서

> **문서 유형:** 기본필수 실습 05 · results.csv 형식 기록의 선정 기준 적용과 지표 해석 보고서  
> **보고서 ID:** D15-B05-MET  
> **작성 기준일:** 2026-11-06
> **검증 범위:** `05_metrics.py`(TODO B06), `data/metrics_example.csv`, `logs/05_metrics.log`, 보조 `_demo_model/results.csv`  
> **판정:** **선정·해석 검증 통과(PASS) / best epoch 4 · mAP50 0.55 · mAP50-95 0.27 (README와 일치)**  
> **모델 출처 표기:** 해당 없음 — `metrics_example.csv`는 읽기 연습용 **가상 기록**, 실제 YOLO 성능 아님

## 1. Executive Summary

**검증 목적.** 학습 기록 표(results.csv 형식)에 **선정 기준**을 적용해 best epoch를 찾고, 그 epoch의 mAP50·mAP50-95를 **한 문장으로 해석**한다. 이 실습의 선정 기준은 "valid mAP50-95가 가장 큰 행, 동점이면 먼저 나온 행"이다. TODO B06 `best = history.loc[history["map50_95"].idxmax()]`를 완성했다.

**핵심 결과.**

| 항목 | 값 |
|:--|:--|
| 선정 기준 | valid mAP50-95 최대(`idxmax`: 첫 최대값) |
| **best epoch** | **4** (5개 epoch 중) |
| **mAP50** | **0.55** |
| **mAP50-95** | **0.27** |
| 마지막 epoch | 5 (mAP50 0.53 · mAP50-95 0.25) — best가 아님 |

**한 문장 해석.**

> **"valid mAP50-95가 가장 큰 epoch 4가 best이며, IoU 0.5 기준으로는 평균 AP 0.55로 물체를 절반 남짓 맞게 찾지만 IoU 0.50~0.95 평균은 0.27로 절반 이하로 떨어져 박스 위치가 정밀하지 않다는 뜻이고, 이 값은 가상 기록이라 실제 모델 성능이 아니다."**

**README 질문 요약 (2.3절):**
- **mAP50-95 < mAP50인 이유:** IoU 기준을 0.95까지 엄격하게 올린 10개 AP의 평균이라, 위치가 조금만 어긋나도 점수가 깎인다.
- **실제 모델 결과인가:** 아니다. 가상 기록이다.

**최종 판단:** PASS. 기준을 명시하고 best epoch·두 지표를 찾아 한 문장으로 해석했다. 숫자를 성능으로 발표하지 않는다.

## 2. 목적 및 검증 범위

### 2.1 실험 범위와 비교 조건

| 항목 | 값 |
|:--|:--|
| 입력 | `data/metrics_example.csv` (5행 × 5열: epoch·map50·map50_95·precision·recall, 가상) |
| 선정 기준 (과제) | `map50_95` 최대 행, 동점이면 먼저 나온 행 |
| 처리 | `history["map50_95"]` → `.idxmax()` → `history.loc[index]` |
| 비교용 기준 (보고서) | mAP50 최대 · 마지막 epoch · Ultralytics fitness — 같은 표에 적용해 결과가 같은지 확인 |
| 비교 기준값 | README 확인 출력(참고값) |
| 제외 | 실제 학습·val 재계산 |

### 2.2 입력 기록과 선정 기준 적용

| epoch | map50 | map50_95 | precision | recall | map50 − map50_95 | map50_95 / map50 | 비고 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 0.20 | 0.08 | 0.45 | 0.30 | 0.12 | 0.40 | |
| 2 | 0.38 | 0.15 | 0.58 | 0.45 | 0.23 | 0.39 | |
| 3 | 0.51 | 0.24 | 0.69 | 0.57 | 0.27 | 0.47 | |
| **4** | **0.55** | **0.27** | **0.73** | **0.62** | 0.28 | **0.49** | **best (선정)** |
| 5 | 0.53 | 0.25 | 0.72 | 0.60 | 0.28 | 0.47 | 마지막 epoch |

(오른쪽 두 열은 보고서에서 계산한 값이다.)

### 2.3 README 질문 답변

#### Q1. mAP50-95가 mAP50보다 낮은 이유는?

mAP50은 예측 박스가 정답과 IoU ≥ 0.5만 겹치면 TP로 센다. mAP50-95는 IoU 기준을 0.50, 0.55, …, 0.95로 **10번 바꿔 각각 AP를 구한 평균**이다. 기준이 0.9·0.95로 올라가면 거의 완벽히 겹친 박스만 TP가 되므로 AP가 크게 떨어진다. 그래서 같은 예측에서 항상 mAP50-95 ≤ mAP50이다.

이번 표에서 best epoch 4의 mAP50-95는 mAP50의 **49%**(0.27 / 0.55)다. 이것은 "대략 찾기는 하지만 박스 테두리가 정답과 꼭 맞지는 않는다"는 뜻이다. 비율은 epoch 1~2의 0.39~0.40에서 epoch 4의 0.49로 올라, 학습이 진행되며 **위치 정밀도가 함께 좋아진** 기록으로 읽힌다.

#### Q2. 이 예제 표가 실제 모델 결과인가?

**아니다.** 스크립트 첫 줄과 출력이 "읽기 연습용 가상 기록 — 실제 YOLO 성능 아님"이라고 밝히고, DATA_NOTE도 "실제 성능으로 발표하지 마세요"라고 적는다. 실제 Ultralytics `results.csv`는 열 이름이 `metrics/mAP50(B)`, `metrics/mAP50-95(B)`처럼 다르고, 손실·학습률 열도 함께 들어 있다(4.2절).

#### 주제. 선정 기준을 바꾸면 best epoch가 달라지는가?

| 선정 기준 | 계산 | 선택 epoch | mAP50 | mAP50-95 |
|:--|:--|--:|--:|--:|
| **mAP50-95 최대 (과제 기준)** | `map50_95.idxmax()` | **4** | 0.55 | 0.27 |
| mAP50 최대 | `map50.idxmax()` | 4 | 0.55 | 0.27 |
| 마지막 epoch | 마지막 행 | 5 | 0.53 | 0.25 |
| Ultralytics 8.4.21 fitness | mAP50-95 단독(가중치 [P 0, R 0, mAP50 0, mAP50-95 1]) | 4 | 0.55 | 0.27 |
| 예전 Ultralytics fitness | 0.1·mAP50 + 0.9·mAP50-95 (epoch 4: 0.298 최대) | 4 | 0.55 | 0.27 |

이 표에서는 **마지막 epoch를 고르는 경우만 결과가 다르다.** epoch 5는 epoch 4보다 두 지표가 모두 낮다. 그래서 "가장 오래 학습한 것 = 가장 좋은 것"이 아니며, 기준을 명시하고 골라야 한다. 기준이 같은 답을 낸 것은 이 표에서 두 지표의 최고 epoch가 우연히 같기 때문이다. 일반적으로는 다를 수 있다(4.2절 실제 예).

주의: 설치된 Ultralytics 8.4.21의 `fitness`는 mAP50-95만 쓴다(소스 확인). 예전 버전의 0.1·mAP50 + 0.9·mAP50-95 정의와 다르므로, 보고서에는 사용한 버전의 정의를 함께 적는다.

## 3. 처리 결과 상세

### 3.1 메서드 체인 단계별 값

| 단계 | 코드 | 값 |
|:--|:--|:--|
| 1 | `history["map50_95"]` | [0.08, 0.15, 0.24, 0.27, 0.25] |
| 2 | `.idxmax()` | 3 (0부터 센 행 번호, 첫 최대값) |
| 3 | `history.loc[3]` | epoch 4 행 전체 |
| 4 | `int(best["epoch"])`, `best["map50"]`, `best["map50_95"]` | 4, 0.55, 0.27 |

### 3.2 표준 출력 (`logs/05_metrics.log`)

```text
자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님
선택 epoch: 4
mAP50: 0.55 mAP50-95: 0.27
mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.
```

### 3.3 한 문장 해석을 만드는 순서

| 순서 | 넣을 내용 | 이번 값 |
|:--:|:--|:--|
| 1 | 선정 기준과 best epoch | valid mAP50-95 최대 → epoch 4 |
| 2 | mAP50의 뜻 | IoU 0.5 기준 평균 AP 0.55 |
| 3 | mAP50-95의 뜻과 차이 | IoU 0.50~0.95 평균 0.27, mAP50의 49% → 위치 정밀도가 낮음 |
| 4 | 자료의 성격 | 가상 기록, valid 평가 지표(accuracy 아님) |

## 4. 보조 확인

### 4.1 같은 방법을 실제 results.csv에 적용 (`_demo_model/results.csv`)

실습 06·07에서 쓴 demo.pt 준비 학습(120 epoch, Ultralytics 8.4.21)의 실제 results.csv에 같은 기준을 적용했다. 이 기록은 가상 표가 아니라 이 환경에서 실제로 학습한 결과이며, 같은 val로 epoch를 골랐으므로 일반화 성능이 아니다.

| 선정 기준 | 선택 epoch | mAP50 | mAP50-95 |
|:--|--:|--:|--:|
| mAP50-95 최대 (`idxmax`, 첫 최대값) | 119 | 0.97292 | 0.87596 |
| mAP50 최대 | 89 | 0.98894 | 0.78702 |
| 마지막 epoch | 120 | 0.97292 | 0.87596 |

| 관찰 | 해석 |
|:--|:--|
| mAP50 최대(89)와 mAP50-95 최대(119)가 다름 | 실제 기록에서는 기준에 따라 best epoch가 달라진다. epoch 89는 대략 찾기(mAP50)는 가장 좋지만 위치 정밀도(mAP50-95 0.787)가 낮다 |
| epoch 119와 120의 mAP50-95가 소수 5자리까지 같음 | `idxmax`는 먼저 나온 119를 고른다. Ultralytics는 `best_fitness == fitness`일 때 best.pt를 다시 저장하므로 진짜 동점이면 나중 epoch가 best.pt가 된다(소스 확인). results.csv는 반올림 값이라 실제로 동점인지는 이 파일로 알 수 없다 |

### 4.2 실제 results.csv와 연습 표의 열 차이

| 연습 표 열 | 실제 Ultralytics results.csv 열 |
|:--|:--|
| epoch | epoch |
| map50 | metrics/mAP50(B) |
| map50_95 | metrics/mAP50-95(B) |
| precision · recall | metrics/precision(B) · metrics/recall(B) |
| (없음) | time, train/box_loss·cls_loss·dfl_loss, val/…_loss, lr/pg0~2 |

## 5. 품질 검증 및 합격 기준

| TC ID | 검증 내용 | 합격 기준 | 관찰 근거 | 결과 |
|:--|:--|:--|:--|:--:|
| TC-01 | 선정 기준 적용 | mAP50-95 최대 행 | B06 코드 | PASS |
| TC-02 | best epoch | 4 | stdout | PASS |
| TC-03 | mAP50 · mAP50-95 | 0.55 · 0.27 | stdout | PASS |
| TC-04 | 한 문장 해석 | 기준·epoch·두 지표·자료 성격 포함 | 1절 인용 | PASS |
| TC-05 | 가상 표시 | "실제 YOLO 성능 아님" 출력 | stdout 1행 | PASS |
| TC-06 | README 일치 | 4줄 동일 | 로그 대조 | PASS |
| TC-07 | 기준 민감도 | 마지막 epoch와 구별 | 2.3절 표 | PASS |

## 6. 리스크 및 개선 과제

| 우선순위 | 이슈 | 권고 조치 |
|:--:|:--|:--|
| P0 | 가상 표를 실제 성능으로 인용 | 결과와 함께 자료 출처(source) 기록 |
| P1 | 기준을 적지 않으면 best가 무엇인지 모호 | "valid mAP50-95 최대, 동점은 먼저" 처럼 기준을 한 줄로 명시 |
| P1 | 라이브러리 fitness 정의가 버전마다 다름 | 사용한 Ultralytics 버전과 fitness 정의를 함께 기록 |
| P1 | best를 고른 valid로 다시 평가하면 낙관적 | 최종 성능은 test에서 한 번 확인 |
| P2 | mAP를 accuracy로 오해 | mAP는 위치+분류의 valid 평가임을 명시 |

## 7. 다음 단계

| 단계 | 내용 |
|:--:|:--|
| 1 | 학습된 모델로 val 추론과 오류 확인(실습 06) |
| 2 | 실제 val 지표 읽기(응용 05: `metrics.box.map50`, `.map`) |
| 3 | best·last checkpoint 비교(심화 05) |

## 8. 최종 판단

- **선정:** 기준(valid mAP50-95 최대)으로 best epoch 4, mAP50 0.55, mAP50-95 0.27을 찾았고 README와 같다.
- **해석:** IoU 0.5에서는 절반 남짓 맞지만 엄격한 IoU 평균은 그 절반 이하라 위치 정밀도가 낮다. 마지막 epoch 5는 best가 아니다.
- **판정:** **PASS.** 숫자는 가상 기록이다.

---

## 부록 A. 근거 파일 및 검증 이력

| 파일 | 역할 |
|:--|:--|
| `05_metrics.py` | TODO B06 + 전체 주석 |
| `data/metrics_example.csv` | 가상 학습 기록(입력) |
| `logs/05_metrics.log` | 실제 실행 출력 |
| `../../_demo_model/results.csv` | 4.1절 실제 학습 기록 |
| `../05_metrics.ipynb`, `../outputs_notebook/p05_history.png` | 추이 그림 노트북 |

## 재현

| 항목 | 값 |
|:--|:--|
| 실행 명령 | `python 05_metrics.py` |
| 환경 | Windows, Intel Core Ultra 9 275HX, CPU 실행 |
| Python | 3.13.16 |
| 주요 패키지 | Ultralytics 8.4.203, torch 2.14.0+cu130 (CUDA 빌드지만 device=cpu로 실행) |
| seed·스레드 | 난수 미사용(결정적 계산) |
| 재실행 일치 | 노트북에서 같은 epoch 4 선택 |
| 원본 보존 | 원본 미수정 |
