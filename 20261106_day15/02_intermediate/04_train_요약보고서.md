# Day15 Detection | Python API·CLI 동일 설정과 runs·best/last 경로 기록 결과요약보고서

> **문서 유형:** 응용 실습 04 · 학습 설정 고정·기록과 2 epoch smoke 학습 검증 보고서  
> **보고서 ID:** D15-I04-TRN  
> **작성 기준일:** 2026-11-06
> **검증 범위:** `04_train.py`(TODO A04), `outputs/dataset_active.yaml`, `outputs/same_settings_cli.txt`, `outputs/smoke/`
> **판정:** **API 학습·경로 기록 통과(PASS) — 지정한 설정이 그대로 적용되고 runs·best/last 경로가 확정됨 / CLI는 기록만 하고 미실행 / 2 epoch val mAP 0은 smoke의 예상 결과**  
> **모델 출처 표기:** `yolo26n.yaml` 무작위 초기화(`pretrained=False`). demo.pt 미사용

## 1. Executive Summary

**검증 목적.** 같은 학습을 **Python API**(`model.train(...)`)와 **CLI**(`yolo detect train ...`)로 할 때 설정이 같은지 비교할 수 있게 한다. 그리고 학습 결과가 저장되는 **runs 폴더와 best.pt·last.pt 경로**를 기록한다. TODO A04에서 `model.train(data=str(path), epochs=2, imgsz=160, batch=4, workers=0, device="cpu", amp=False, pretrained=False, optimizer="SGD", lr0=0.01, seed=42, plots=False, project=str(out), name="smoke", exist_ok=True)`를 완성했다. 스크립트는 같은 값의 CLI 명령을 `outputs/same_settings_cli.txt`에 저장한다.

**핵심 결과**

| 확인 항목 | 결과 |
|:--|:--|
| API에 준 15개 인자 | trainer 인자 줄에 **모두 같은 값으로 기록됨**(2.3절 표) |
| optimizer | 로그 `SGD(lr=0.01, momentum=0.937)` — auto 선택이 아니라 **지정값 그대로** |
| runs 폴더(save_dir) | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke` |
| best / last | `...\outputs\smoke\weights\best.pt`·`last.pt`, 각 5.3MB, optimizer 제거 |
| 학습 | 2 epoch 완료, epoch당 9 iteration |
| val | 12장·30객체, 두 epoch와 best.pt 검증 모두 P·R·mAP 0 |
| CLI | 같은 설정 명령을 파일로 기록. 로그 마지막 줄 "CLI 비교는 선택 실행입니다" — **CLI 자체는 미실행** |

**README 질문 요약 (2.3절):**
- **연결 성공 ≠ 탐지 품질:** 2 epoch(18 iteration) smoke는 설정·경로·저장이 맞는지를 보여 줄 뿐이고, mAP 0은 예상된 결과다.
- **`.yaml`과 `.pt`:** yaml은 구조 설계도(weight 없음), pt는 학습된 weight와 names·학습 인자를 담은 파일이다.

**최종 판단:** API 학습과 경로 기록은 **PASS**다. API와 CLI의 결과 비교는 CLI를 실행하지 않아 **미실행**이다. 품질 판단은 하지 않는다.

## 2. 목적 및 검증 범위

### 2.1 실험 범위와 비교 조건

| 항목 | 값 |
|:--|:--|
| 모델 | `YOLO("yolo26n.yaml")`, pretrained False → "Overriding model.yaml nc=80 with nc=3" |
| 데이터 | `outputs/dataset_active.yaml` (train·val 경로 + names 3개, test 키 없음) |
| 고정 인자 | epochs 2, imgsz 160, batch 4, workers 0, device cpu, amp False, optimizer SGD, lr0 0.01, seed 42, plots False |
| 저장 위치 인자 | project `outputs`, name `smoke`, exist_ok True → `outputs/smoke/` |
| CLI 기록 | 같은 값, **name만 `cli_smoke`**(API 결과를 덮어쓰지 않도록 분리) |
| 반복 수 | 36장 / batch 4 = epoch당 9 iteration (로그 `9/9`) |
| 실행 환경 | 사용자 PC (재현 절 참고) |

### 2.2 데이터 분포

| split | 이미지 | 비율 | 객체 | square | circle | triangle | negative | group |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| train | 36 | 60.0% | 90 | 30 | 30 | 30 | 6 | 18 |
| val | 12 | 20.0% | 30 | 10 | 10 | 10 | 2 | 6 |
| test | 12 | 20.0% | 30 | 10 | 10 | 10 | 2 | 6 |
| **합계** | **60** | 100% | **150** | 50 | 50 | 50 | 10 | 30 |

`train: ... 36 images, 6 backgrounds, 0 corrupt`와 `val: ... 12 images, 2 backgrounds, 0 corrupt`가 이 표의 train·val 행(이미지 수, negative 수)과 같다. backgrounds는 빈 라벨(정상 negative) 이미지다. test는 쓰지 않는다.

### 2.3 README 질문 답변과 주제 분석

#### 주제 ① Python API와 CLI의 동일 설정 비교

API 호출 인자, 스크립트가 기록한 CLI 토큰, PC 로그의 trainer 인자 줄을 나란히 놓았다.

| 설정 | API (`model.train`) | CLI (`same_settings_cli.txt`) | PC trainer 인자 | 일치 |
|:--|:--|:--|:--|:--:|
| 모델 | `YOLO("yolo26n.yaml")` | `model=yolo26n.yaml` | `model=yolo26n.yaml` | O |
| 데이터 | `data=str(path)` | `data="<BASE>/outputs/dataset_active.yaml"` | `data=C:\workspace\practice_assignment\20261106\02_intermediate\outputs\dataset_active.yaml` | O |
| epochs | 2 | `epochs=2` | `epochs=2` | O |
| imgsz | 160 | `imgsz=160` | `imgsz=160` | O |
| batch | 4 | `batch=4` | `batch=4` | O |
| workers | 0 | `workers=0` | `workers=0` | O |
| device | "cpu" | `device=cpu` | `device=cpu` | O |
| amp | False | `amp=False` | `amp=False` | O |
| pretrained | False | `pretrained=False` | `pretrained=False` | O |
| optimizer | "SGD" | `optimizer=SGD` | `optimizer=SGD` | O |
| lr0 | 0.01 | `lr0=0.01` | `lr0=0.01` | O |
| seed | 42 | `seed=42` | `seed=42` | O |
| plots | False | `plots=False` | `plots=False` | O |
| project | `str(out)` | `project="<BASE>/outputs"` | `project=C:\workspace\practice_assignment\20261106\02_intermediate\outputs` | O |
| name | "smoke" | `name=cli_smoke` | `name=smoke` | **의도적 차이** |
| exist_ok | True | `exist_ok=True` | `exist_ok=True` | O |

| 관찰 | 해석 |
|:--|:--|
| 지정한 값이 trainer 인자에 그대로 남음 | API 호출이 의도대로 해석됐다. CLI도 같은 키=값을 쓰므로 같은 설정이 된다 |
| optimizer 줄이 `SGD(lr=0.01, momentum=0.937)` | `optimizer="SGD"`를 명시해 lr0가 실제로 적용됐다. auto였다면 lr0·momentum을 무시하고 자동 선택한다 |
| 지정하지 않은 값(momentum 0.937, warmup_epochs 3, close_mosaic 10, mosaic 1.0 등) | API·CLI 모두 같은 기본값을 쓴다. 단, Ultralytics 버전이 바뀌면 기본값이 바뀔 수 있으므로 비교는 같은 버전끼리 한다 |
| name만 다름 | 저장 폴더만 `smoke`와 `cli_smoke`로 갈리고 학습 조건에는 영향이 없다 |
| CLI 미실행 | 같은 설정이라는 것은 **인자 문자열 대조로만** 확인했다. CLI 결과 손실·지표가 같은지는 미실행이다 |

주의: `data`·`project`의 `<BASE>`는 스크립트 위치에서 계산한 절대 경로다. 다른 PC나 폴더로 옮기면 CLI 문자열의 경로가 바뀐다. 따라서 txt 파일은 그 실행 환경 기준 기록이다.

#### 주제 ② runs 폴더·best/last 경로 기록

| 항목 | 경로 | 정하는 인자 |
|:--|:--|:--|
| 데이터 설정 | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\dataset_active.yaml` | 스크립트 `path` |
| runs 폴더(save_dir) | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke` ("Logging results to") | project + name |
| last.pt | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke\weights\last.pt` (5.3MB) | save_dir/weights |
| best.pt | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke\weights\best.pt` (5.3MB) | save_dir/weights |
| 스크립트 출력 | `best/last: C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke\weights` | `out/"smoke/weights"` |
| CLI 기록 파일 | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\same_settings_cli.txt` (코드상 위치, 로그에 내용 없음) | `out` |
| CLI를 실행할 경우 | `C:\workspace\practice_assignment\20261106\02_intermediate\outputs\cli_smoke\weights\` (미실행) | name=cli_smoke |
| 라벨 cache | `C:\workspace\practice_assignment\20261106\02_intermediate\data\labels\train.cache`, `val.cache` 새로 생성 | Ultralytics 자동 |

| 관찰 | 해석 |
|:--|:--|
| `exist_ok=True` | 다시 실행해도 `smoke2`, `smoke3`이 생기지 않고 같은 `smoke` 폴더를 덮어쓴다. 경로가 고정되어 기록·비교가 쉽지만 이전 결과는 사라진다 |
| best.pt와 last.pt가 같은 크기 | 두 epoch 모두 지표가 0이라 fitness가 같다. best는 첫 최고 epoch, last는 마지막 epoch의 weight다 |
| "Optimizer stripped" | 두 파일 모두 추론용으로 optimizer가 제거됐다. 학습 재개(resume)에는 쓸 수 없다 |
| 검증 대상 | 학습 직후 `best.pt`로 다시 검증("Validating ...\best.pt") |

#### Q1. 1 epoch(이번엔 2 epoch) 연결 성공과 실제 탐지 품질이 다른 이유는?

| 관점 | 2 epoch smoke | 의미 |
|:--|:--|:--|
| 갱신 횟수 | 9 + 9 = 18 iteration | 무작위 weight를 아주 조금 움직였다 |
| 학습률 구간 | warmup 3 epoch 안 | 학습률이 아직 작다 |
| 손실 | box 3.133 → 3.192, cls 4.303 → 4.378 | 줄지 않았다(warmup 중 학습률이 커지며 흔들림) |
| val | 두 epoch 모두 P·R·mAP 0 | 아직 물체를 찾지 못한다 |

smoke 성공은 "데이터·인자·저장 경로가 맞아 다음 단계로 가도 된다"를 뜻하고, 품질 확인은 "써도 된다"를 뜻한다. 탐지 품질은 충분히 학습한 demo.pt(응용 05·06)로 확인한다.

#### Q2. `.yaml`과 `.pt`는 무엇이 다른가?

| 파일 | 내용 | 이번 실행 |
|:--|:--|:--|
| `yolo26n.yaml` | 층 구성·채널·연결 설계도, weight 없음 | 260 layers, 2,504,970 parameters 모델 생성 |
| `smoke\weights\best.pt`·`last.pt` | 학습된 weight + names·학습 인자, optimizer 제거 | 5.3MB, 검증 시 fused 120 layers·2,375,421 parameters |
| `smoke\args.yaml` | 실제 적용된 모든 인자 | 설정 재현의 근거(4절) |

`pretrained=False`로 두어 yaml 모델에 외부 weight가 섞이지 않게 했다.

## 3. 처리 결과 상세

### 3.1 준비 단계

| 로그 | 값 | 해석 |
|:--|:--|:--|
| 환경 | Ultralytics 8.4.203, Python 3.13.16, torch 2.14.0+cu130, CPU | GPU_mem 0G, device cpu |
| 모델 | 260 layers, 2,504,970 parameters, 5.9 GFLOPs | 학습 시 구조 |
| 데이터 스캔 | train 36장(6 backgrounds) · val 12장(2 backgrounds), corrupt 0 | 라벨 형식 이상 없음 |
| cache | `train.cache`, `val.cache` 새로 생성 | 다음 실행부터 스캔 생략 |
| 경고 | Slow image access (read 7.1 / 6.7 MB/s) | 원격·느린 저장소 경고. 소형 데이터라 학습 시간 영향은 작다(가설) |
| optimizer | SGD(lr=0.01, momentum=0.937), 114/126/126 parameter groups | 지정값 적용 |
| 기타 | `Using 36 train, 12 val images for fraction=1.0 at imgsz=160`, `Using 0 dataloader workers`, `Closing dataloader mosaic` | 전체 데이터·메인 프로세스 로딩·mosaic 끈 상태 |

### 3.2 학습과 검증

| epoch | box_loss | cls_loss | l1_loss | iteration | 학습 시간(참고값) | val P / R / mAP50 / mAP50-95 |
|--:|--:|--:|--:|:--|--:|:--|
| 1/2 | 3.133 | 4.303 | 0.04265 | 9/9, 9.7 it/s | 0.9s | 0 / 0 / 0 / 0 |
| 2/2 | 3.192 | 4.378 | 0.04148 | 9/9, 11.4 it/s | 0.8s | 0 / 0 / 0 / 0 |

| best.pt 최종 검증 | Images | Instances | P | R | mAP50 | mAP50-95 |
|:--|--:|--:|--:|--:|--:|--:|
| all | 12 | 30 | 0 | 0 | 0 | 0 |
| square · circle · triangle | 각 10 | 각 10 | 0 | 0 | 0 | 0 |

속도(참고값): preprocess 0.1ms · inference 4.9ms · postprocess 0.3ms per image. 전체 학습은 `2 epochs completed in 0.001 hours`다.

### 3.3 스크립트 마지막 출력

```text
best/last: C:\workspace\practice_assignment\20261106\02_intermediate\outputs\smoke\weights CLI 비교는 선택 실행입니다.
```

## 4. 확인 사항

### 4.1 기록되는 CLI 명령 (`outputs/same_settings_cli.txt`)

```text
yolo detect train model=yolo26n.yaml data="<BASE>/outputs/dataset_active.yaml" epochs=2 imgsz=160 batch=4 workers=0 device=cpu amp=False pretrained=False optimizer=SGD lr0=0.01 seed=42 plots=False project="<BASE>/outputs" name=cli_smoke exist_ok=True
```

`<BASE>`는 `04_train.py`가 있는 폴더의 절대 경로다(PC에서는 `C:\workspace\practice_assignment\20261106\02_intermediate`). CLI 비교를 할 때는 이 명령을 같은 가상환경에서 실행한 뒤 `outputs/cli_smoke/results.csv`의 손실과 `outputs/smoke/results.csv`를 대조한다.

### 4.2 runs 폴더 안의 파일

| 경로 | 내용 |
|:--|:--|
| `outputs/smoke/results.csv` | epoch별 손실·P·R·mAP·lr. 2 epoch라 2행 (`time`은 누적 초) |
| `outputs/smoke/args.yaml` | 실제 적용된 모든 인자. API·CLI 설정 대조의 파일 근거 |
| `outputs/smoke/weights/best.pt`, `last.pt` | 학습된 weight |
| 그림 파일 | `plots=False`라 곡선·batch 그림은 만들지 않음 |

### 4.3 재실행 확인 (노트북)

`../04_train.ipynb`에서 같은 설정으로 다시 학습했을 때 train 손실 값이 같게 나왔다(time만 다름). 같은 장치·같은 버전에서는 seed 42·deterministic·workers 0 설정으로 손실이 재현된다.

## 5. 품질 검증 및 합격 기준

| TC ID | 검증 내용 | 합격 기준 | 관찰 근거| 결과 |
|:--|:--|:--|:--|:--:|
| TC-01 | API 설정 적용 | 지정한 15개 인자가 trainer 인자 줄과 같음 | trainer 인자 줄 | PASS |
| TC-02 | optimizer 고정 | `SGD(lr=0.01, ...)` | optimizer 줄 | PASS |
| TC-03 | runs 폴더 | `outputs\smoke` | "Logging results to" | PASS |
| TC-04 | best/last 경로 | `smoke\weights\best.pt`·`last.pt` 생성 | "Optimizer stripped" 2줄, 마지막 출력 | PASS |
| TC-05 | 2 epoch 완료 | `2 epochs completed` | 로그 | PASS |
| TC-06 | test 미사용 | YAML에 test 키 없음 | `dataset_active.yaml` | PASS |
| TC-07 | API·CLI 설정 문자열 일치 | name 외 같은 값 | 2.3절 표 | PASS |
| TC-08 | CLI 실행 결과 비교 | — | 선택 실행, 로그 없음 | 미실행 |
| TC-09 | 탐지 품질 | 판단 대상 아님 | P·R·mAP 0 | N/A |

## 6. 리스크 및 개선 과제

| 우선순위 | 이슈 | 권고 조치 |
|:--:|:--|:--|
| P1 | CLI 미실행이라 결과 동일성은 미검증 | `same_settings_cli.txt` 명령을 실행해 `cli_smoke/results.csv` 손실을 `smoke`와 대조 |
| P1 | `exist_ok=True`로 재실행 시 이전 결과를 덮어씀 | 비교 실험은 name을 바꾸거나 결과를 별도 폴더에 보관 |
| P1 | smoke 결과를 성능으로 오해 | 보고서에 epoch·iteration 수·목적 표기 |
| P2 | CLI 문자열에 절대 경로가 들어감 | 다른 PC로 옮기면 경로를 다시 생성 |
| P2 | 지정하지 않은 값은 버전 기본값을 따름 | args.yaml을 결과와 함께 보관, 같은 버전끼리 비교 |
| P2 | Slow image access 경고 | 데이터를 로컬 디스크에 두고 실행 |
| P2 | `data\labels\*.cache` 생성 | 제출 전 삭제 |

## 7. 다음 단계

| 단계 | 내용 |
|:--:|:--|
| 1 | 학습된 모델의 실제 val 지표(응용 05) |
| 2 | imgsz만 바꾼 비교(심화 04) |
| 3 | CLI 실행 후 API 결과와 손실 대조(선택) |

## 8. 최종 판단

- **API·CLI 설정:** API에 준 값이 그대로 적용됐고, CLI 명령은 name을 뺀 모든 값이 같다. 실행 결과 비교는 미실행이다.
- **경로:** runs 폴더 `outputs\smoke`와 best/last 경로 `outputs\smoke\weights\`가 확정되고 출력됐다.
- **품질:** 2 epoch smoke라 mAP 0이며 판단 대상이 아니다.
- **판정:** **PASS**(설정 적용·경로 기록) / CLI 결과 비교 **미실행**.

---

## 부록 A. 근거 파일 및 검증 이력

| 파일 | 역할 |
|:--|:--|
| `04_train.py` | TODO A04 + 전체 주석 |
| `outputs/same_settings_cli.txt` | 같은 설정 CLI 명령 |
| `outputs/smoke/` | results.csv·args.yaml·weights |
| `../04_train.ipynb`, `../outputs_notebook/smoke/` | 학습 노트북 |

## 재현

| 항목 | 값 |
|:--|:--|
| 실행 명령 | `python 04_train.py` (README가 있는 `02_intermediate` 폴더에서) |
| 환경 | 사용자 PC (Windows, Intel Core Ultra 9 275HX, CPU 실행, `.gpuvenv` 가상환경) |
| Python | 3.13.16 |
| 주요 패키지 | Ultralytics 8.4.203, torch 2.14.0+cu130 (device=cpu로 실행) |
| seed·스레드 | seed 42, deterministic=True, workers 0, `torch.set_num_threads(1)` |
| 재실행 일치 | PC 재실행 결과 미제공(미실행). 다른 장치·버전과 bitwise 동일은 요구하지 않음 |
| 원본 보존 | 원본 `04_train.py` 미수정 |
