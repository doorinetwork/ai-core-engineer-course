# Day15 Detection | Confidence·NMS IoU 임계값 비교 결과요약보고서

> **문서 유형:** 응용 실습 02 · 고전 NMS 임계값 조건별 남는 박스 수·Precision·Recall 비교 보고서  
> **보고서 ID:** D15-I02-NMS  
> **작성 기준일:** 2026-11-06 (재작성)  
> **검증 범위:** `02_nms.py`(TODO A02), `data/boxes_fixture.csv`·`boxes_gt.csv`, `outputs/nms_comparison.csv`, 확장 격자 `outputs/nms_threshold_grid.csv`(fixture 49조합)·`outputs/yolo_val_conf_nms_grid.csv`(실제 YOLO val 40조합)  
> **판정:** **과제 3조건 README와 일치(PASS) / conf는 Precision·Recall을 함께 움직이고, NMS IoU는 Recall을 바꾸지 않은 채 중복 FP만 조절함을 확인**  
> **모델 출처 표기:** 본 과제는 `teaching_fixture`(손으로 정한 후보 4개, YOLO 예측 아님). 보조 격자는 `self_prepared_demo.pt` — 배포본에 없는 `data/demo.pt`를 이 환경에서 자체 준비(yolo26n.yaml 무작위 초기화 → 02_응용 train 36장 120 epoch, test 미사용). 강사용 `prepare_yolo.py`(별도 source 132장) 산출물이 아님

## 1. Executive Summary

**검증 목적.** 후보 박스에 두 임계값을 차례로 적용할 때 결과가 어떻게 바뀌는지 비교한다. 첫째는 **confidence 임계값(conf)**으로, 점수가 낮은 후보를 버린다. 둘째는 **NMS IoU 임계값**으로, 같은 클래스 후보끼리 이 값보다 많이 겹치면 점수가 낮은 쪽을 버린다. 각 조건에서 **남는 박스 수(kept)·TP/FP/FN·Precision·Recall**을 기록했다. TODO A02 `keep = ids[batched_nms(boxes[ids], scores[ids], classes[ids], nms_iou)]`를 완성했고, 과제의 3조건 외에 임계값 격자를 실제로 실행했다.

**핵심 결과 (과제 3조건, README와 일치).**

| 조건 | conf | NMS IoU | 남은 박스 | TP | FP | FN | Precision | Recall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 기준 | 0.25 | 0.50 | 3 | 2 | 1 | 1 | 0.6667 | 0.6667 |
| conf만 올림 | **0.50** | 0.50 | 2 | 2 | 0 | 1 | **1.0000** | 0.6667 |
| NMS IoU만 올림 | 0.25 | **0.95** | 4 | 2 | 2 | 1 | **0.5000** | 0.6667 |

**임계값별 효과 요약.**

| 바꾼 임계값 | 남는 박스 | Precision | Recall | 결정 경계(fixture) |
|:--|:--|:--|:--|:--|
| conf ↑ | 줄어듦 (3 → 2 → 1) | 오름 (0.667 → 1.0) | **떨어질 수 있음** (0.667 → 0.333, conf > 0.80) | 후보 점수 0.40·0.80·0.85 |
| NMS IoU ↑ | 늘어남 (3 → 4) | 내림 (0.667 → 0.5) | **변화 없음** (0.667 유지) | 중복 쌍의 서로 IoU 0.90 |

**보조 확인 (실제 YOLO val 12장, 30객체).** 모델 출력에 클래스별 NMS를 더하면 conf 0.25에서 NMS IoU ≤ 0.85일 때 FP가 9 → **0**이 되고, Recall은 0.9667로 그대로였다. 같은 이미지·같은 클래스의 중복 후보 쌍 20개가 서로 IoU 0.722~0.949로 겹쳐 있었기 때문이다.

**README 질문 요약 (2.3절):** 한 번에 임계값 하나만 바꾸었다. NMS는 정답을 보지 않는 후처리라 **놓친 정답(FN)을 되살리지 못한다.** 후보 4개 fixture와 겹친 물체가 없는 합성 val로는 운영 임계값을 단정할 수 없다.

**최종 판단:** 과제는 PASS. 두 임계값의 역할 차이는 fixture와 실제 YOLO 출력 양쪽에서 같은 방향으로 확인했다. 최적 조합은 결론 내리지 않는다.

## 2. 목적 및 검증 범위

### 2.1 실험 범위와 비교 조건

| 항목 | 값 |
|:--|:--|
| 처리 순서 | ① `scores >= conf` 필터 → ② `batched_nms(boxes, scores, classes, nms_iou)` → ③ 정답과 IoU 0.5 일대일 매칭 |
| 고정 | 매칭 IoU 0.5, 같은 클래스·정답 1회 사용, 입력 후보, 클래스별 억제(다른 클래스끼리는 억제하지 않음) |
| 과제 조건 | (0.25, 0.50) 기준 / (0.50, 0.50) conf만 변경 / (0.25, 0.95) NMS IoU만 변경 |
| 확장 격자 A (fixture) | conf 0.25·0.40·0.41·0.50·0.85·0.86·0.95 × NMS IoU 0.30·0.50·0.70·0.85·0.89·0.90·0.95 = 49조합 |
| 확장 격자 B (YOLO val) | conf 0.15·0.25·0.30·0.40·0.50 × NMS IoU 0.30·0.50·0.70·0.80·0.85·0.90·0.95·1.00 = 40조합 |
| 경계 규칙 | conf는 `>=`(같은 값 통과), NMS는 IoU **>** 임계값일 때만 제거(같은 값 보존). NMS IoU 1.00 = 억제 없음 |
| 함수 | `torchvision.ops.batched_nms`, `box_iou` |
| test 사용 | 사용 안 함 |

### 2.2 데이터

| 자료 | 행 수 | 클래스별 | 성격 |
|:--|--:|:--|:--|
| `boxes_gt.csv` (정답) | 3 | square 1 · circle 1 · triangle 1 | 기하 정답, confidence 없음 |
| `boxes_fixture.csv` (예측) | 4 | square 2(0.95·0.85) · circle 1(0.80) · triangle 1(0.40) | 손으로 정한 원리 연습값, YOLO 예측 아님 |

**fixture 후보 간 관계 (임계값 경계를 정하는 값).**

| 후보 | class | conf | 정답과 IoU | 다른 후보와 IoU | 역할 |
|:--|:--|--:|--:|:--|:--|
| P0 | square | 0.95 | 0.9025 | P1과 **0.90** (교집합 1368 / 합집합 1520) | 정답 |
| P1 | square | 0.85 | 0.8162 | P0과 0.90 | 중복 |
| P2 | circle | 0.80 | 1.0000 | — | 정답 |
| P3 | triangle | 0.40 | 0.0000 | — | 위치 오류 |

**YOLO val (보조).** val 12장, 객체 30개(클래스별 10), negative 2장. 모델 출력을 conf 0.05로 받은 후보 84개(`yolo_val_candidates_conf005.csv`)에 위 처리 ①~③을 다시 적용했다. conf ≥ 0.15 후보는 47개, ≥ 0.25는 38개다. 모델 출처는 부록 B에 있다.

### 2.3 README 질문 답변

#### Q. 무엇을 고정했고 무엇을 바꾸었는가? 결과만으로 단정할 수 없는 부분은?

| 구분 | 내용 |
|:--|:--|
| 고정 | 후보·정답, 매칭 IoU 0.5, 일대일 규칙, 클래스별 억제 |
| 바꾼 것 | 과제 조건마다 **임계값 하나만**: conf(0.25 → 0.50) 또는 NMS IoU(0.50 → 0.95). 확장 격자도 한 축씩 읽는다 |
| 관찰 ① conf | 올리면 FP와 TP가 **함께** 줄 수 있다. 0.40 → 0.41에서 P3(오검출)가 빠져 Precision 1.0이 되고, 0.85 → 0.86 근처를 넘으면 P2(circle 정답, 0.80)까지 빠져 Recall이 0.333으로 떨어진다 |
| 관찰 ② NMS IoU | 중복 쌍의 서로 IoU 0.90 **이하** 임계값(0.30~0.89)에서는 P1을 지우고, 0.90 이상에서는 남긴다. TP는 그대로라 Recall 변화가 없다 |
| 관찰 ③ 결합 | conf가 중복 P1(0.85)보다 높으면(0.86 이상) NMS IoU와 무관하게 결과가 같다. 먼저 걸러진 후보에는 NMS가 할 일이 없다 |
| 단정할 수 없는 것 | ① 운영 conf·NMS IoU 최적값(후보 4개, 겹친 물체 없는 합성 val) ② **같은 클래스 물체가 실제로 겹쳐 있는 장면**에서 낮은 NMS IoU가 진짜 물체를 지우는 손실(이번 데이터에 그런 장면이 없어 미측정) ③ YOLO26의 기본 end-to-end 출력에 NMS를 추가하는 것이 일반적으로 이로운지 ④ test·실제 데이터에서 같은 경향인지 |

**NMS와 매칭의 분리:** NMS는 정답을 보지 않고 **후보끼리** 비교하는 추론 후처리이고, 매칭은 **후보와 정답**을 비교하는 평가 단계다. 두 단계 모두 IoU를 쓰지만 비교 대상이 다르다. 그래서 NMS IoU를 아무리 바꿔도 정답과 겹치는 후보가 원래 없던 triangle의 FN은 사라지지 않는다.

## 3. 처리 결과 상세

### 3.1 과제 3조건의 단계별 후보 (`outputs/nms_comparison.csv`)

| 조건 | conf 통과 ids | NMS 후 keep | 매칭 결과 |
|:--|:--|:--|:--|
| 0.25 / 0.50 | [0,1,2,3] | [0,2,3] | P0 TP · P2 TP · P3 FP |
| 0.50 / 0.50 | [0,1,2] | [0,2] | P0 TP · P2 TP |
| 0.25 / 0.95 | [0,1,2,3] | [0,1,2,3] | P0 TP · P1 FP(중복) · P2 TP · P3 FP |

```text
 conf  nms_iou  kept  tp  fp  fn  precision  recall
 0.25     0.50     3   2   1   1     0.6667  0.6667
 0.50     0.50     2   2   0   1     1.0000  0.6667
 0.25     0.95     4   2   2   1     0.5000  0.6667
```

### 3.2 확장 격자 A — fixture 49조합 (`outputs/nms_threshold_grid.csv`)

같은 행 안의 값이 같으면 열을 묶어 표시했다. 칸은 **kept / Precision / Recall**이다.

| conf \ NMS IoU | 0.30 · 0.50 · 0.70 · 0.85 · 0.89 | 0.90 · 0.95 |
|:--|:--|:--|
| 0.25 · 0.40 | 3 / 0.667 / 0.667 | 4 / 0.500 / 0.667 |
| 0.41 · 0.50 | 2 / 1.000 / 0.667 | 3 / 0.667 / 0.667 |
| 0.85 | 1 / 1.000 / 0.333 | 2 / 0.500 / 0.333 |
| 0.86 · 0.95 | 1 / 1.000 / 0.333 | 1 / 1.000 / 0.333 |

| 경계 | 원인 | 바뀐 것 |
|:--|:--|:--|
| conf 0.40 → 0.41 | P3(0.40)가 conf 미달로 제거 | kept −1, FP −1, Precision ↑ |
| conf 0.80 초과(표에선 0.85) | P2(circle 정답, 0.80) 제거 | TP −1, FN +1, Recall 0.667 → 0.333 |
| conf 0.85 → 0.86 | P1(0.85) 제거 → NMS가 지울 중복이 없어짐 | NMS IoU 열 차이 사라짐 |
| NMS IoU 0.89 → 0.90 | P0·P1 서로 IoU 0.90이 임계값보다 크지 않음 | 중복 P1 보존, FP +1, Precision ↓ |

주의: NMS IoU 0.90에서 P1이 남은 것은 "IoU > 임계값일 때만 제거"하는 경계 규칙 때문이다(같은 값은 보존).

### 3.3 확장 격자 B — 실제 YOLO val 40조합 (`outputs/yolo_val_conf_nms_grid.csv`, 보조)

칸은 **kept / FP / Recall**이다. NMS IoU 1.00은 억제 없음, 즉 모델 출력 그대로이며 응용 06의 결과와 같다(conf 0.15: 47 / 17 / 1.000, conf 0.25: 38 / 9 / 0.967).

| conf \ NMS IoU | 0.30 · 0.50 · 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 (억제 없음) |
|:--|:--|:--|:--|:--|:--|:--|
| 0.15 | 30 / 0 / 1.000 | 31 / 1 / 1.000 | 33 / 3 / 1.000 | 39 / 9 / 1.000 | 47 / 17 / 1.000 | 47 / 17 / 1.000 |
| 0.25 | 29 / 0 / 0.967 | 29 / 0 / 0.967 | 29 / 0 / 0.967 | 34 / 5 / 0.967 | 38 / 9 / 0.967 | 38 / 9 / 0.967 |
| 0.30 | 29 / 0 / 0.967 | 29 / 0 / 0.967 | 29 / 0 / 0.967 | 31 / 2 / 0.967 | 33 / 4 / 0.967 | 33 / 4 / 0.967 |
| 0.40 | 24 / 0 / 0.800 | 24 / 0 / 0.800 | 24 / 0 / 0.800 | 24 / 0 / 0.800 | 24 / 0 / 0.800 | 24 / 0 / 0.800 |
| 0.50 | 19 / 0 / 0.633 | 19 / 0 / 0.633 | 19 / 0 / 0.633 | 19 / 0 / 0.633 | 19 / 0 / 0.633 | 19 / 0 / 0.633 |

| 관찰 | 수치 근거 |
|:--|:--|
| NMS IoU는 Recall을 바꾸지 않음 | 각 conf 행에서 Recall이 모든 NMS IoU에서 같음 |
| 중복 쌍의 서로 IoU 0.722~0.949 (conf ≥ 0.15, 같은 이미지·클래스 20쌍) | NMS IoU ≤ 0.70이면 중복 전부 제거, 0.95 이상이면 전부 남음 |
| conf로 FP를 0으로 만들면 Recall 손실 | NMS 없이 FP 0이 되는 첫 conf 0.40에서 Recall 0.800(conf 0.15 대비 TP 6개 손실) |
| conf 0.40 이상에서는 NMS가 할 일이 없음 | 중복 후보 점수가 모두 0.40 미만 |
| 다른 클래스끼리 IoU > 0.3으로 겹친 후보 | 0쌍 → 클래스별 억제와 전체 억제의 차이는 이 데이터에서 관찰 불가 |

**해석 주의:** 이 val의 물체들은 서로 겹치지 않는다. 그래서 낮은 NMS IoU가 겹쳐 있는 진짜 물체를 지우는 부작용이 나타날 수 없고, "NMS IoU 0.30도 손실 없음"은 이 데이터에서만 성립한다.

## 4. 품질 검증 및 합격 기준

| TC ID | 검증 내용 | 합격 기준 | 관찰 근거 | 결과 |
|:--|:--|:--|:--|:--:|
| TC-01 | 클래스별 억제 | idxs=classes 전달 | A02 코드 | PASS |
| TC-02 | index 복원 | `ids[...]`로 원본 index | A02 코드 | PASS |
| TC-03 | 과제 표 일치 | README 3행과 동일 | stdout | PASS |
| TC-04 | 단일 변수 변경 | 조건마다 임계값 하나만 다름 | 2.1절 | PASS |
| TC-05 | 격자 A 정합 | 과제 3조건 칸이 3.1절과 같음 | `nms_threshold_grid.csv` | PASS |
| TC-06 | 격자 B 정합 | NMS 1.00 칸이 응용 06 결과와 같음 | conf .15·.25 행 | PASS |
| TC-07 | 재실행 | 노트북 `nms_comparison.csv` 바이트 동일, 격자 B 재실행 CSV 바이트 동일 | `cmp` | PASS |
| TC-08 | 운영 임계값 확정 | 비용 기준·겹친 물체 장면 필요 | — | 보류 |

## 5. 리스크 및 개선 과제

| 우선순위 | 이슈 | 권고 조치 |
|:--:|:--|:--|
| P0 | 겹친 같은 클래스 물체가 없는 데이터라 NMS IoU를 낮출 때의 손실 미측정 | 물체가 겹친 장면을 추가해 NMS IoU 0.5·0.7 비교 |
| P1 | conf로만 중복을 지우면 Recall 손실(격자 B: conf 0.40에서 conf 0.15 대비 TP 6개 손실) | 낮은 conf + NMS 조합을 val에서 비교한 뒤 test 1회 확인 |
| P1 | 후보 4개 fixture로 임계값 결론 불가 | 실제 예측 격자(이번 보조 분석)를 함께 제시 |
| P1 | 경계값(IoU == 임계값)의 처리 | 구현별 `>` / `>=` 규칙을 문서화(특심화 X03은 == 보존) |
| P2 | 자체 준비 모델의 중복 경향 | 강사 모델로 격자 B 재실행 |

## 6. 다음 단계

| 단계 | 내용 | 기준 |
|:--:|:--|:--|
| 1 | 실제 모델의 conf 후보 비교(응용 06) | val에서만 결정 |
| 2 | NMS 직접 구현·동점·경계 계약(특심화 X03) | 테스트 통과 |
| 3 | 겹친 물체 장면으로 NMS IoU 하한 탐색 | Recall 손실 0인 최소 IoU |

## 7. 최종 판단

- **기능:** A02 완성, 과제 3조건이 README와 같다.
- **conf:** 올리면 남는 박스가 줄고 Precision이 오르지만, 정답 후보 점수를 넘는 순간 Recall이 떨어진다.
- **NMS IoU:** 낮추면 중복 FP만 줄고 Recall은 변하지 않는다. 효과는 중복 쌍의 서로 IoU(fixture 0.90, YOLO 0.722~0.949)를 기준으로 갈린다.
- **판정:** **PASS.** 운영 임계값은 겹친 물체 장면과 오류 비용 기준 없이 정하지 않는다.

---

## 부록 A. 근거 파일 및 검증 이력

| 파일 | 역할 |
|:--|:--|
| `02_nms.py` | TODO A02 + 전체 주석 |
| `outputs/nms_comparison.csv`, `outputs/logs/02_nms.log` | 과제 3조건 실제 실행 결과 |
| `outputs/nms_threshold_grid.csv`, `outputs/logs/nms_grid_fixture.log` | 격자 A (fixture 49조합) |
| `outputs/yolo_val_candidates_conf005.csv` | 격자 B 입력: demo.pt val 후보 84개 |
| `outputs/yolo_val_conf_nms_grid.csv`, `outputs/logs/nms_grid_yolo.log` | 격자 B (YOLO val 40조합) |
| `outputs/logs/nms_threshold_grid.py` | 격자 A·B 실행 코드(`02_nms.py`와 같은 처리 순서) |
| `../02_nms.ipynb` | 단계별 keep 확인 노트북 |

## 부록 B. 격자 B 모델 출처

| 항목 | 값 |
|:--|:--|
| 준비 이유 | 배포 ZIP에 `data/demo.pt`가 없고 강사용 `운영/prepare_yolo.py`도 없음 → 실제 YOLO 경로를 실행하려고 자체 준비 |
| 구조 | `yolo26n.yaml`(무작위 초기화, `pretrained=False`, COCO weight 미사용), 2,375,421 파라미터(fused) |
| 학습 데이터 | `02_응용/data/images/train` 36장(정상 30·negative 6) |
| epoch 선택 | `02_응용/data/images/val` 12장으로 Ultralytics fitness 최고 epoch(best.pt) 선택 |
| test 사용 | 사용 안 함 |
| 설정 | epochs 120, imgsz 160, batch 8, SGD lr0 0.01, seed 42, deterministic, workers 0, CPU |
| 파일 | 5,345,285 bytes, SHA-256 `66a9a244ba353bbdbd7629c0cc84427b3c38964065b6093b4efd6d5bf80fcad1` |
| 준비 시 val(참고) | P 0.895 / R 0.847 / mAP50 0.970 / mAP50-95 0.873 |
| 근거 | `20261106_day15/_demo_model/`(prepare_demo.py·demo_model_provenance.json·results.csv) |

주의: 강사 준비 경로는 목표 데이터와 분리된 source 132장으로 학습한다. 이 모델은 응용·심화·특심화와 같은 생성 규칙의 train으로 학습했으므로, 같은 계열 val의 지표는 강사 경로보다 낙관적일 수 있다.

## 재현

| 항목 | 값 |
|:--|:--|
| 실행 명령 | `python 02_nms.py` · `python outputs/logs/nms_threshold_grid.py fixture` · `python outputs/logs/nms_threshold_grid.py yolo` |
| README 검수 환경(참고) | Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu, OpenCV 4.13.0, NumPy 2.3.5, pandas 2.2.3 |
| seed·스레드 | 난수 미사용(NMS·매칭 결정적, YOLO 추론 결정적) |
| 재실행 일치 | 노트북 `nms_comparison.csv` 바이트 동일, 격자 B 2회 실행 CSV 바이트 동일 |
| 모델(격자 B) | `data/demo.pt` SHA-256 `66a9a244ba353bbdbd7629c0cc84427b3c38964065b6093b4efd6d5bf80fcad1` |
| 원본 보존 | 원본 `02_nms.py` 미수정 |
