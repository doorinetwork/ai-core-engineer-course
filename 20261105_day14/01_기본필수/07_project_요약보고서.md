# 전이학습 프로젝트 실행 결과 보고서

> **프로젝트:** Day 14 전이학습·오류 분석 기본필수 실습 07(+08 재현) — 전이학습 프로젝트 (PPT 78·83쪽)  
> **문서 유형:** 기술 검증 결과 요약 / 교육용 실습 평가  
> **작성일:** 2026-11-04  
> **판정:** **기능 요구사항 충족(PASS) / README 정답 출력 2줄과 일치, 재실행 시 CSV·JSON 5개 바이트 동일. test 미사용, 소표본이라 일반화 성능은 미평가**  
> **평가 대상:** `07_project.py`(TODO B07·B08·B09), `metrics.csv`, `metrics.json`, `predictions.csv`, `history.csv`, `confusion.csv`, `project_cam.png`, `logs/07_project.log`, `logs/07_project_rerun.log`

---

## 1. Executive Summary

본 실습의 목적은 실습 01~06의 요소(source Backbone 고정, Head 학습, valid 기준 epoch 선택, 지표·예측 기록, 오류·CAM)를 **하나의 재현 가능한 전이학습 프로젝트**로 묶고, 실습 08에서 **같은 설정으로 다시 실행해 결과가 같은지** 확인하는 것이다. 이를 위해 TODO B07(`loss.backward()`·`optimizer.step()`), B08(`predictions = logits.argmax(dim=1)`), B09(`metrics.json`에 UTF-8 JSON 저장)를 완성하였다.

**실제 실행 결과 기준**, Head-only 8 epoch 학습에서 valid 손실은 0.8207 → 0.0905로 매 epoch 감소했고 **best_epoch 8**이 선택되었다. 최종 모델은 valid 9장에서 **Accuracy 1.0, Macro F1 1.0, 오답 0건**이었다. **test는 사용하지 않았다(test_used: false).** 실습 08 재실행에서 metrics.csv·metrics.json·predictions.csv·history.csv·confusion.csv가 **바이트 단위로 같았고**, project_cam.png는 픽셀이 같았다.

**README 질문 요약 (2.3절 상세):**
- **재실행 비교를 위해 고정할 것:** 입력 데이터(원본 36장 SHA256), 분할·클래스 매핑, source 배열(SHA256), seed 14·스레드 수 1, 모델 구조·학습 범위·BN 정책, Optimizer·lr·epoch, 지표 정책(labels·zero_division)을 고정해야 한다.
- **test를 남겨 두는 이유:** valid는 epoch 선택에 이미 쓰여 낙관적으로 치우친다. 모든 선택이 끝난 뒤 한 번만 쓰는 test가 있어야 처음 보는 데이터의 성능을 정직하게 추정할 수 있다.

**최종 판단:** 전이학습 프로젝트 구성과 재현성의 실습 완료 조건을 충족했다(PASS). 단, valid 9장에서 1.0은 소표본 결과이며 **같은 valid로 epoch을 고르고 평가했으므로 일반화 성능이 아니다**. 05·06의 초기 모델 오류(4건)와 07 최종 모델 결과(0건)는 서로 다른 모델의 결과다.

## 2. 목적 및 검증 범위

### 2.1 검증 목적

- Head-only 전이학습을 epoch 단위로 수행하고 valid 손실로 best 상태를 선택한다.
- 최종 예측·지표·혼동행렬·학습 기록을 파일로 남긴다.
- 오답이 있으면 저장하고, 없으면 억지로 만들지 않는다.
- 같은 설정으로 재실행해 결과 파일이 같은지 확인한다(실습 08).

### 2.2 검증 범위

| 구분 | 포함 | 제외 |
|:--|:--|:--|
| 데이터 | train 21장(학습), valid 9장(epoch 선택·평가) | test(최종 평가용으로 보존) |
| 모델 | source TinyShapeCNN Backbone 고정 + Head(771) | partial, 공식 ResNet18 |
| 학습 | AdamW lr 0.01, wd 0, 8 epoch, 전체 배치, BN eval 고정 | 불균형 대응(weight·sampler) |
| 산출물 | metrics.csv·json, predictions.csv, history.csv, confusion.csv, project_cam.png | 모델 파일 저장(기본필수 요구 없음) |
| 기준값 | README 실습 07 "확인 출력"(참고값) | — |

**TODO 정답:**
- B07 `loss.backward()` → `optimizer.step()`. 원본 `pass`는 갱신이 없어 Head가 초기값 그대로였다.
- B08 `predictions = logits.argmax(dim=1)`. 원본 zeros는 모두 circle(0)로 예측한 가짜 결과였다.
- B09 `metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)` → `(OUT / "metrics.json").write_text(metrics_json, encoding="utf-8")`. 경로는 README 생성 파일 목록의 `metrics.json`(OUT 폴더)을 사용했다.

TODO 외 코드는 원본 그대로이며 주석만 추가했다.

### 2.3 README 질문 답변

#### Q1. 어떤 입력·설정·seed·매핑을 고정해야 재실행 결과를 비교할 수 있는가?

| 분류 | 고정 항목 | 이번 실행 값 | 확인 방법 |
|:--|:--|:--|:--|
| 입력 | 원본 이미지 36장 | manifest SHA256 36/36 일치 | `data/manifest.csv` 대조 |
| 분할 | train/valid/test 폴더 | 21 / 9 / 6장, group 누수 0 | 실습 01 감사 |
| 매핑 | `class_to_idx` | circle 0, square 1, triangle 2 | 실습 01 출력 |
| source | `source_backbone.npz` | SHA256 `fdedd0b6…dd29f4` | provenance 대조 |
| 전처리 | Resize(32,32) + ToTensor | RGB float [0,1] | 코드 |
| 난수·연산 | `manual_seed(14)`, `set_num_threads(1)` | Head 초기값·연산 순서 동일 | 코드 |
| 모델·범위 | 구조, Head-only, BN eval 고정 | 학습 파라미터 771 | 실습 02·03 |
| 학습 | AdamW lr 0.01, wd 0, 8 epoch, 전체 배치 | history 8행 | history.csv |
| 선택 규칙 | valid 손실 최소 epoch | best_epoch 8 | metrics |
| 지표 정책 | labels [0,1,2], macro, zero_division 0 | Macro F1 1.0 | metrics |
| 환경 | 라이브러리 버전·장치 | CPU, torch 2.10.0 등(부록 A) | 기록 |

이 항목을 고정한 상태로 재실행한 결과, CSV·JSON 5개가 바이트 단위로 같았다(3.5절). 장치·라이브러리 버전이 바뀌면 마지막 자릿수가 달라질 수 있으므로 환경도 함께 기록한다.

#### Q2. test를 왜 남겨 두는가?

이 프로젝트는 valid를 **epoch 선택**(best_epoch 8)과 **최종 지표 계산**에 모두 사용했다. 선택에 쓴 데이터로 평가하면 그 데이터에 맞춰 고른 결과라 성능이 낙관적으로 나온다. test 6장은 클래스 매핑·개수 감사(실습 01) 외에 학습·추론·설명·임계값 선택에 한 번도 쓰지 않았으므로, **모든 설계를 확정한 뒤 단 한 번 평가**하면 처음 보는 데이터에 대한 정직한 추정치가 된다. test를 보고 설정을 다시 바꾸면 test도 valid처럼 오염되므로, 이 단계에서는 `test_used: False`를 결과에 기록하고 남겨 둔다.

## 3. 처리 결과 상세

### 3.1 처리 흐름

| 순서 | 처리 | 적용 설정 | 산출물 |
|:--:|:--|:--|:--|
| 1 | 데이터·모델 | train/valid Tensor, source Backbone | — |
| 2 | Head-only | 전체 잠금 → Head 열기, `eval()` | 학습 771개 |
| 3 | 학습 (TODO B07) | zero_grad → loss → backward → step, 8 epoch | history |
| 4 | best 선택 | valid 손실 최소 상태 복사 → 복원 | best_epoch |
| 5 | 예측 (TODO B08) | softmax 확률, argmax 예측 | predictions |
| 6 | 기록 (TODO B09 포함) | metrics.csv·json, history.csv, predictions.csv, confusion.csv | 파일 5개 |
| 7 | 오답 저장 | 이전 project_error 삭제 → 오답만 저장 | 0건 → 파일 없음 |
| 8 | CAM | 오답 없으면 valid 0번(circle_000), target=예측 | project_cam.png |

### 3.2 학습 기록 (`history.csv`)

| epoch | valid_loss |
|--:|--:|
| 1 | 0.8207 |
| 2 | 0.6393 |
| 3 | 0.4311 |
| 4 | 0.2686 |
| 5 | 0.1748 |
| 6 | 0.1318 |
| 7 | 0.1067 |
| 8 | **0.0905** (best) |

valid 손실이 8 epoch 내내 감소해 마지막 epoch가 best로 선택되었다. 아직 감소 중이므로 epoch를 늘리면 더 낮아질 수 있으나, 이번 실행에서는 확인하지 않았다.

### 3.3 지표·예측 (`metrics.json`, `predictions.csv`, `confusion.csv`)

```json
{"accuracy": 1.0, "best_epoch": 8, "macro_f1": 1.0, "source": "synthetic_source_pretraining", "test_used": false, "valid_n": 9}
```

| file | actual | predicted | confidence |
|:--|--:|--:|--:|
| circle_000.png | 0 | 0 | 0.934 |
| circle_001.png | 0 | 0 | 0.950 |
| circle_002.png | 0 | 0 | 0.964 |
| square_000.png | 1 | 1 | 0.986 |
| square_001.png | 1 | 1 | 0.986 |
| square_002.png | 1 | 1 | **0.763** (최저) |
| triangle_000.png | 2 | 2 | 0.884 |
| triangle_001.png | 2 | 2 | 0.976 |
| triangle_002.png | 2 | 2 | 0.808 |

혼동행렬은 대각선 [3, 3, 3], 나머지 0이다. 05 초기 모델에서 틀렸던 square_002는 맞혔지만 확신도가 가장 낮다.

### 3.4 오답·CAM

- **오답 0건**: `project_error_*.png`는 생성되지 않았다(이전 파일 삭제 규칙도 작동).
- `project_cam.png`: 오답이 없어 valid 0번 circle_000.png, target circle의 CAM이다. 원 테두리 왼쪽 아래에 반응이 모였다(그림 관찰). 반응 지도이며 원인 증명이 아니다.

### 3.5 실습 08 재현 확인

| 파일 | 1차 실행 vs 2차 실행 |
|:--|:--|
| metrics.csv | 바이트 동일 |
| metrics.json | 바이트 동일 |
| predictions.csv | 바이트 동일(행 순서·값 동일) |
| history.csv | 바이트 동일 |
| confusion.csv | 바이트 동일 |
| project_cam.png | 픽셀 동일(196,608값 차이 0). 파일 바이트는 프로젝트 저장소의 PNG 재저장 때문에 다름 |
| stdout | 2줄 동일 |
| 원본 data | 원본 36장 SHA256이 manifest와 일치(변경 없음) |

### 3.6 실습 08 기록표

| 기록 | 작성 |
|:--|:--|
| 사용 모델·source 출처 | TinyShapeCNN(Conv 3→8→16, GAP 4×4, Linear 256→3). Backbone은 별도 합성 도형 240장으로 사전학습한 `source_backbone.npz`(ImageNet 아님, Target 미사용, SHA256 일치) |
| train/valid 클래스 개수 | train [12, 6, 3], valid [3, 3, 3] (circle·square·triangle) |
| 학습 범위·BN 정책 | Head(9번) 771개만 학습, Backbone 잠금, `eval()`로 BN running 통계 고정 |
| 주 지표와 이유 | Macro F1. train이 12:6:3 불균형이라 Accuracy만으로는 소수 클래스 실패가 숨음(실습 04). 이번 최종 결과 1.0 |
| 관찰한 오류·CAM target | 최종 모델 오답 0건. 05·06 초기 모델은 오답 4건(모두 circle로 예측), CAM target은 예측 클래스 circle |
| 재실행 확인·한계 1개 | 재실행 CSV·JSON 5개 바이트 동일. 한계: valid 9장으로 epoch 선택과 평가를 함께 해 1.0은 일반화 성능이 아님 |

### 3.7 표준 출력 (`logs/07_project.log`, 재실행 `logs/07_project_rerun.log` 동일)

```text
{"accuracy": 1.0, "best_epoch": 8, "macro_f1": 1.0, "source": "synthetic_source_pretraining", "test_used": false, "valid_n": 9}
오답 수: 0 예측 행: 9 test 사용: False
```

## 4. 품질 검증 및 합격 기준

| TC ID | 검증 내용 | 합격 기준 | 관찰 근거 | 결과 |
|:--|:--|:--|:--|:--:|
| TC-01 | metrics JSON 출력 | README 1행과 동일 | stdout | PASS |
| TC-02 | 오답·행 수·test | 오답 0, 예측 9행, test False | stdout | PASS |
| TC-03 | 학습 진행 | valid 손실 감소 | history 0.8207 → 0.0905 | PASS |
| TC-04 | metrics.json | UTF-8 JSON, 키 6개 | 파일 확인 | PASS |
| TC-05 | 생성 파일 | README 목록 6종 생성, 오답 이미지 없음 | 파일 확인 | PASS |
| TC-06 | 재현성(08) | 결과 파일 동일 | 3.5절 | PASS |
| TC-07 | 원본 보존 | data SHA256 manifest 일치 | 보조 감사 | PASS |
| TC-08 | test 미사용 | test 미로드 | 코드·metrics | PASS |
| TC-09 | 일반화 성능 | 독립 데이터 평가 | test 보존 | 미평가 |

> **주의:** 숫자의 마지막 자릿수는 라이브러리·장치에 따라 달라질 수 있다. 시간은 정답으로 고정하지 않으며 측정하지 않았다.

## 5. 리스크 및 개선 과제

| 우선순위 | 이슈 / 잠재 영향 | 권고 조치 | 검증 산출물 |
|:--:|:--|:--|:--|
| P0 | valid로 epoch 선택과 평가를 함께 함 → 낙관적 지표 | 설계 확정 후 test 1회 평가로 별도 보고 | test 지표 |
| P0 | valid 9장 → 1장 차이가 11%p | 데이터 확대, 신뢰구간 또는 반복 실험 | 반복 결과표 |
| P1 | 8 epoch에서도 손실 감소 중 | epoch 범위를 넓혀 best 위치 확인 | 확장 history |
| P1 | square_002 확신도 0.763 → 경계 사례 | 확신도 하한 기준·검토 대상 표시 | 저확신 목록 |
| P2 | 모델 파일 미저장 → 같은 모델로 재추론 불가 | 본인 생성 파일만 `torch.save` / `weights_only=True`로 복원 | 모델 파일·해시 |
| P2 | 환경 버전이 README 검증 환경과 다름 | 실행 환경을 결과와 함께 저장 | 환경 기록 |

## 6. 다음 단계: 현업 적용 승인 전 검증 계획

| 단계 | 검증 항목 | 승인 기준 |
|:--:|:--|:--|
| 1 | 재현성 | 같은 환경 재실행 시 결과 파일 동일(이번 PASS) |
| 2 | 데이터 규모 | 클래스당 표본을 늘린 데이터(예: 응용 300장)에서 같은 절차 |
| 3 | 반복 실험 | seed 여러 개에서 Macro F1 평균·범위 보고 |
| 4 | 최종 평가 | 설계 확정 후 test 1회, 결과를 valid와 분리 보고 |
| 5 | 오류 관리 | 저확신·오답 사례 목록과 CAM(target 명시) 저장 |
| 6 | 출처 비교 | 공식 ResNet18 Weight를 준비한 환경에서 별도 비교 |

## 7. 최종 판단

- **기능 검증:** TODO B07·B08·B09를 완성해 학습·예측·JSON 저장이 동작했고, 출력이 README와 같다(Accuracy 1.0, Macro F1 1.0, best_epoch 8, 오답 0).
- **재현성:** 같은 설정으로 다시 실행해 결과 파일이 같았고 원본 데이터는 바뀌지 않았다.
- **한계:** valid 9장으로 선택·평가를 함께 했으므로 1.0은 일반화 성능이 아니다. test는 최종 평가용으로 남겨 두었다.
- **권고:** 본 실습은 **완료(PASS) 및 후속 검증 진행**으로 분류한다. 현업 판단은 더 큰 데이터·반복 실험·test 1회 평가 이후에 한다.

---

## 부록 A. 근거 파일 및 검증 이력

| 파일 | 역할 |
|:--|:--|
| `07_project.py` | TODO B07·B08·B09 완성 + 전체 주석 |
| `metrics.csv`, `metrics.json` | **실제 실행 지표** |
| `predictions.csv` | **실제 실행 valid 예측 9행** |
| `history.csv` | **실제 실행 epoch별 valid 손실** |
| `confusion.csv` | **실제 실행 혼동행렬** |
| `project_cam.png` | **실제 실행 최종 모델 CAM** |
| `logs/07_project.log`, `logs/07_project_rerun.log` | 1차·2차 실행 표준 출력 |
| `logs/run1/` | 재현 비교용 1차 실행 결과 사본 |
| `../README.md` | 실습 07·08 요구사항·정답 출력(참고값) |

- **검증 방식:** `python 07_project.py`를 실제 실행해 stdout을 README와 대조하고, 결과를 `logs/run1/`에 복사한 뒤 다시 실행해 `cmp`(바이트)와 픽셀 비교로 대조했다. 원본 data SHA256은 별도 보조 감사다.
- **제약:** train 21장·valid 9장, 단일 seed(14), CPU 실행.
- **재현 명령:** `python 07_project.py` (실습 08: 같은 명령 재실행 후 결과 파일 비교)
- **보고 원칙:** 측정된 사실, 그 사실에서 계산한 값, 검증되지 않은 가정을 구분해 작성했다.

*교육용 실습 결과 보고서 — 2026-11-04 실제 실행 산출물과 README.md를 근거로 작성했으며 운영 승인 문서가 아니다.*
