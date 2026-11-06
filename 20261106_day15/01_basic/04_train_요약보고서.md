# Day15 Detection | YOLO 객체 생성·속성 확인과 smoke test 결과요약보고서

> **문서 유형:** 기본필수 실습 04 · Ultralytics YOLO 학습 파이프라인 연결(smoke test) 검증 보고서  
> **보고서 ID:** D15-B04-TRN  
> **작성 기준일:** 2026-11-06 (PC 실행 결과 기준 재작성)  
> **검증 범위:** `04_train.py`(TODO B05), `p04_dataset.yaml`, `p04_runs/smoke/`, 보조 기록 `logs/04_train.log`·`logs/smoke_epochs_1to3/`  
> **판정:** **smoke test 통과(PASS) — PC에서 학습·저장·자동 검증까지 연결 성공, task·names·device 출력 정상 / P·R·mAP 0은 1 epoch smoke의 예상 결과로 품질 판단 대상 아님**  
> **모델 출처 표기:** `yolo26n.yaml` 구조에서 무작위 초기화한 smoke 모델(사전학습 weight 전이 없음). demo.pt 미사용

## 1. Executive Summary

**검증 목적.** 본격 학습 전에 **"YOLO 객체 → 데이터 설정 → 학습 → 저장 → 자동 검증"이 끝까지 연결되는지**를 가장 작은 비용으로 확인한다(smoke test). 확인 항목은 세 가지다. ① `YOLO("yolo26n.yaml")`로 객체를 만든다. ② 학습 뒤 `model.task`·`model.names`·`model.device`를 출력해, 모델이 이번 데이터에 맞게 설정됐는지 본다. ③ 소형 데이터(train 12장)로 1 epoch만 돌려 runs 폴더와 weights가 생기는지 본다. TODO B05에서 `model.train(data=str(config_path), epochs=1, imgsz=160, batch=4, device="cpu", workers=0, seed=42, project=str(BASE/"p04_runs"), name="smoke", exist_ok=True)`를 완성했다.

**핵심 결과**

| 확인 항목 | 결과 |
|:--|:--|
| 학습 완료 | `1 epochs completed in 0.001 hours`, 3/3 iteration, 1.3s |
| 출력 task | **detect** |
| 출력 names | **{0: 'square', 1: 'circle', 2: 'triangle'}** |
| 출력 device | **cpu** |
| weights | `best.pt`·`last.pt` 각 5.3MB (optimizer 제거) |
| 자동 검증 | val 6장·18객체, P·R·mAP50·mAP50-95 모두 0 |

**README 질문 요약 (2.3절):**
- **1 epoch 연결 성공 ≠ 탐지 품질:** smoke test는 "파이프라인이 돈다"만 증명한다. 무작위 weight를 3번 갱신한 정도로는 물체를 찾지 못하므로 mAP 0이 정상이다.
- **`.yaml`과 `.pt`:** yaml은 층 구성 설계도라 weight가 없다. pt는 학습된 weight와 names·학습 인자를 담는다.

**최종 판단:** smoke test **PASS**. 지표 0은 실패가 아니며, 품질은 충분히 학습한 모델(실습 06·07)로 판단한다.

## 2. 목적 및 검증 범위

### 2.1 실험 범위와 비교 조건

| 항목 | 값 |
|:--|:--|
| YOLO 객체 | `YOLO("yolo26n.yaml")` — 패키지에 든 구조 설정으로 무작위 초기화, 다운로드 없음 |
| 데이터 설정 | `p04_dataset.yaml`: train·val 경로와 names 3개, **test 키 없음** |
| 학습 인자 (B05) | epochs 1, imgsz 160, batch 4, device cpu, workers 0, seed 42, project `p04_runs`, name `smoke`, exist_ok True |
| 제공 기본값 유지 | optimizer=auto, warmup_epochs 3, close_mosaic 10 등 나머지는 Ultralytics 기본값 |
| 반복 수 | train 12장 / batch 4 = epoch당 **3 iteration** (로그 `3/3`) |
| 속성 출력 | 학습 뒤 `print("task:", model.task, "names:", model.names, "device:", model.device)` |
| 실행 환경 | 사용자 PC (재현 절 참고) |
| 판정 기준 | 학습 완료·파일 생성·속성 값. **지표 값은 기준에 넣지 않음** |

### 2.2 데이터 분포

| split | 이미지 | 비율 | 객체 | square | circle | triangle | negative | group |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| train | 12 | 50.0% | 36 | 12 | 12 | 12 | 0 | 6 |
| val | 6 | 25.0% | 18 | 6 | 6 | 6 | 0 | 3 |
| test | 6 | 25.0% | 18 | 6 | 6 | 6 | 0 | 3 |
| **합계** | **24** | 100% | **72** | 24 | 24 | 24 | 0 | 12 |

smoke test는 이 중 train 12장으로 학습하고 val 6장으로 자동 검증한다. PC 로그의 검증 행 `all 6 18`과 클래스별 `6 6`이 이 표의 val 행과 같다. test 6장은 쓰지 않는다.

### 2.3 README 질문 답변

#### Q1. 1 epoch 연결 성공과 실제 탐지 품질이 다른 이유는?

| 구분 | smoke test가 증명하는 것 | 증명하지 못하는 것 |
|:--|:--|:--|
| 데이터 | YAML 경로·라벨 형식·클래스 수가 맞다 (names 3개 출력) | 라벨 위치가 정확한지 |
| 학습 | 순전파·손실·역전파·optimizer 갱신이 끝까지 돈다 (3/3 iteration) | 손실이 수렴하는지 |
| 저장 | best.pt·last.pt 생성, `Results saved to ...\p04_runs\smoke` | 저장된 weight가 쓸 만한지 |
| 검증 | 학습 직후 val 6장 평가가 돈다 | 탐지 성능 |

무작위 weight에서 **3번만 갱신**했다. 또한 기본 warmup이 3 epoch라 1 epoch는 학습률이 아주 작은 구간이다. 그래서 box_loss 3.105, cls_loss 4.262로 손실이 높게 남았고, val의 P·R·mAP는 클래스 3개 모두 0이다. 0이 나왔다는 사실은 **검증 단계까지 실행됐다는 증거**이지 실패가 아니다. 실제 탐지 품질은 충분히 학습한 demo.pt를 쓰는 실습 06·07에서 확인한다.

#### Q2. `.yaml`과 `.pt`는 무엇이 다른가?

| 파일 | 내용 | 결과에서 |
|:--|:--|:--|
| `yolo26n.yaml` | 층 종류·연결·채널 수 같은 **구조 설계도**, weight 없음 | `YOLO("yolo26n.yaml")`로 무작위 초기화 모델 생성 |
| `p04_runs\smoke\weights\best.pt`·`last.pt` | 학습된 **weight + names·학습 인자**, 추론용으로 optimizer 제거 | "Optimizer stripped ... 5.3MB" 후 저장 |
| `demo.pt` (실습 06·07) | 충분히 학습한 weight | 이 실습에서는 쓰지 않음 |

같은 yaml로 만든 모델도 weight가 다르면 예측이 완전히 다르다. 그래서 보고서에는 pt 파일의 출처와 SHA-256을 함께 기록한다. 주의: optimizer가 제거된 best.pt·last.pt는 추론용이라 학습 재개(resume)에 쓸 수 없다.

#### 추가 질문. task·class names·device를 출력하는 이유는?

| 속성 | PC 출력 | 확인하는 것 | 틀렸다면 생기는 일 |
|:--|:--|:--|:--|
| `model.task` | detect | 탐지 모델(박스 출력)인지 | 분류·분할 모델로 학습·추론하면 출력 형식이 다르다 |
| `model.names` | {0: 'square', 1: 'circle', 2: 'triangle'} | 클래스 번호→이름 매핑이 데이터 YAML과 같은지 | 번호가 밀리면 예측 CSV의 class_id 해석이 바뀐다 |
| `model.device` | cpu | 의도한 장치에서 실행됐는지 | GPU 없는 PC에서 cuda 지정 시 오류, 시간 비교가 무의미 |

yaml로 막 만든 객체의 names는 80개 기본값('0'~'79')이고, 학습이 데이터 YAML을 읽은 뒤에 3개 이름으로 바뀐다(아래 4절 보조 기록). 따라서 학습 뒤 names가 3클래스로 출력됐다는 것은 **학습이 이번 데이터 설정을 실제로 읽었다**는 가장 간단한 증거다. PC는 CUDA 빌드 torch(2.14.0+cu130)를 쓰지만 로그 첫 줄과 device 출력이 모두 CPU라, `device="cpu"` 지정이 그대로 적용됐다.

## 3. 처리 결과 상세 (사용자 PC 실행 로그)

### 3.1 학습

```text
Starting training for 1 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
        1/1         0G      3.105      4.262    0.04568         12        160: 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.3s
```

| 항목 | 값 | 해석 |
|:--|:--|:--|
| GPU_mem | 0G | CPU 실행 |
| iteration | 3/3, 2.2 it/s, 1.3s (참고값) | 12장 / batch 4 = 3번 갱신 |
| box_loss | 3.105 | 박스 위치 손실, 아직 높음 |
| cls_loss | 4.262 | 클래스 손실, 아직 높음 |
| l1_loss | 0.04568 | 이 Ultralytics 버전에서 표시되는 세 번째 박스 손실 항목 |
| Instances / Size | 12 / 160 | 마지막 batch의 물체 수, 입력 크기 160 |
| `Closing dataloader mosaic` | 출력됨 | close_mosaic 10 기본값에 따라 mosaic 증강을 끈 상태로 학습 |

### 3.2 저장과 자동 검증

| 항목 | 값 |
|:--|:--|
| 저장 | `last.pt`·`best.pt` 각 5.3MB, "Optimizer stripped" |
| 검증 대상 | `best.pt` (fused) — 120 layers, 2,375,421 parameters, 0 gradients, 5.3 GFLOPs |
| val 결과 | all 6장·18객체: P 0 · R 0 · mAP50 0 · mAP50-95 0 |
| 클래스별 | square 6 · circle 6 · triangle 6 객체, 모두 0 |
| 속도(참고값) | preprocess 0.0ms · inference 6.3ms · postprocess 0.4ms per image |
| 결과 폴더 | `C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke` |

### 3.3 속성 출력

```text
task: detect names: {0: 'square', 1: 'circle', 2: 'triangle'} device: cpu
학습 결과: C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke
```

## 4. 확인 사항

### 4.1 자동 선택된 optimizer와 학습 시 파라미터 수 (`logs/04_train.log`)

| 항목 | 값 |
|:--|:--|
| 데이터 반영 | "Overriding model.yaml nc=80 with nc=3" |
| 학습 시 모델 | 2,504,970 parameters (검증용 fused 모델은 2,375,421) |
| optimizer | `optimizer=auto` → lr0·momentum을 무시하고 **AdamW(lr=0.001429, momentum=0.9)** 자동 선택 |
| 1 epoch 학습률 | 2.858e-05 (warmup 구간) |
| AMP·pretrained | args에는 amp=True·pretrained=True 기본값이 남지만 CPU·yaml 모델이라 AMP 미사용, weight 전이 기록 없음 |

### 4.2 results.csv 열 (`p04_runs/smoke/results.csv`)

생성되는 열은 `epoch, time, train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0~2`이며, 1 epoch smoke에서는 1행만 생긴다.

### 4.3 YOLO 객체의 names: 학습 전과 후 (`logs/smoke_epochs_1to3/`)

| 시점 | task | names | device |
|:--|:--|:--|:--|
| `YOLO("yolo26n.yaml")` 직후 | detect | 80개 기본값 {0: '0', 1: '1', …} | cpu |
| `model.train(...)` 뒤 | detect | {0: 'square', 1: 'circle', 2: 'triangle'} | cpu |

### 4.4 smoke 범위 1~3 epoch (`logs/smoke_epochs_1to3/`)

`04_train.py`와 같은 설정에서 epochs만 1·2·3으로 바꾼 보조 실행이다. 원본·주석본 스크립트는 바꾸지 않았다.

| 실행 | 마지막 epoch box / cls / dfl | P·R·mAP | weights |
|:--|:--|:--|:--|
| 1 epoch | 3.085 / 4.258 / 0.0396 | 모두 0 | best·last |
| 2 epoch | 3.501 / 4.285 / 0.0490 | 모두 0 | best·last |
| 3 epoch | 3.266 / 4.233 / 0.0381 | 모두 0 | best·last |

세 경우 모두 학습·저장·검증이 끝났고 3 epoch(9 iteration)에서도 mAP는 0이었다. 갱신 횟수가 적어 손실이 오르내리며 추세를 말할 수 없다. 같은 seed라도 전체 epochs 수가 다르면 첫 epoch 손실이 달랐는데(3.085 vs 3.394), 증강 일정이 epochs에 따라 달라지기 때문으로 보인다(가설, 미검증).

## 5. 품질 검증 및 합격 기준

| TC ID | 검증 내용 | 합격 기준 | 관찰 근거 | 결과 |
|:--|:--|:--|:--|:--:|
| TC-01 | YOLO 객체 생성 | yaml로 생성, 다운로드 없음 | 오류 없이 학습 시작 | PASS |
| TC-02 | task 출력 | detect | 마지막 출력 | PASS |
| TC-03 | names 출력 | {0: square, 1: circle, 2: triangle} | 마지막 출력 | PASS |
| TC-04 | device 출력 | cpu | 마지막 출력, GPU_mem 0G | PASS |
| TC-05 | 학습 인자 | epochs 1·imgsz 160·batch 4·cpu | `1/1`, Size 160, 3/3 | PASS |
| TC-06 | test 미전달 | YAML에 test 키 없음 | `p04_dataset.yaml` | PASS |
| TC-07 | 학습 완료 | `1 epochs completed` | 로그 | PASS |
| TC-08 | 저장 | best.pt·last.pt | "Optimizer stripped" 2줄 | PASS |
| TC-09 | 자동 검증 실행 | val 6장·18객체 평가 출력 | 검증 표 | PASS |
| TC-10 | 탐지 품질 | 판단 대상 아님 | P·R·mAP 0 | N/A |

## 6. 리스크 및 개선 과제

| 우선순위 | 이슈 | 권고 조치 |
|:--:|:--|:--|
| P1 | mAP 0을 "모델 실패"로 오해 | 보고서·발표에 smoke 목적·epoch 수·iteration 수를 함께 표기 |
| P1 | optimizer=auto가 lr0·momentum을 무시하고 자동 선택 | 비교 실험은 응용 04처럼 `optimizer="SGD"`·lr0를 명시 |
| P2 | Ultralytics 버전에 따라 로그 열 이름·layer 수 표기가 다를 수 있음 | 보고서에 버전(8.4.203)을 함께 기록 |
| P2 | 시간(1.3s, 6.3ms)은 CPU·첫 실행 비용 포함 | 시간은 참고값으로만 사용 |
| P2 | Ultralytics가 `data/labels/*.cache` 생성 | 제출 전 cache 삭제 |

## 7. 다음 단계

| 단계 | 내용 | 기준 |
|:--:|:--|:--|
| 1 | 학습 기록 읽기(실습 05) | best 선택 기준 이해 |
| 2 | 충분히 학습한 모델로 val 추론(실습 06) | 탐지 결과 확인 |
| 3 | 조건을 고정한 2 epoch smoke와 CLI 기록(응용 04) | optimizer·seed 명시 |

## 8. 최종 판단

- **연결:** PC에서 YOLO 객체 생성 → 학습 → 저장 → 자동 검증이 끝까지 실행됐다.
- **속성:** task detect, names 3클래스, device cpu로 데이터 설정과 장치 지정이 그대로 반영됐다.
- **품질:** P·R·mAP 0은 1 epoch smoke test에서 예상된 값이며 판단 대상이 아니다.
- **판정:** **smoke test PASS.**

---

## 부록 A. 근거 파일 및 검증 이력

| 파일 | 역할 |
|:--|:--|
| `04_train.py` | TODO B05 + 전체 주석 |
| `logs/04_train.log`, `p04_runs/smoke/` | 4.1·4.2절 근거 (보고서 작성 환경 실행) |
| `logs/smoke_epochs_1to3/` | 4.3·4.4절 근거: `smoke_epochs.py`·`smoke_epochs.log`·`ep*_results.csv`·`ep*_args.yaml` |
| `../04_train.ipynb`, `../outputs_notebook/p04_runs/` | 학습 노트북 |

## 재현

| 항목 | 값 |
|:--|:--|
| 실행 명령 | `python 04_train.py` (README가 있는 `01_basic` 폴더에서) |
| 환경 | Windows, Intel Core Ultra 9 275HX, CPU 실행 |
| Python | 3.13.16 |
| 주요 패키지 | Ultralytics 8.4.203, torch 2.14.0+cu130 (CUDA 빌드지만 device=cpu로 실행) |
| 실행 위치 | `C:\workspace\practice_assignment\20261106\01_basic` |
| seed·스레드 | seed 42, deterministic(기본값), workers 0, `torch.set_num_threads(1)` |
| 원본 보존 | 원본 `04_train.py` 미수정 |
