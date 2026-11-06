# Model Card

## 데이터·출처

| 항목 | 값 |
|:--|:--|
| class_to_idx | `{'circle': 0, 'square': 1, 'triangle': 2}` (폴더 이름 알파벳 순) |
| 입력 | RGB, Resize(32,32) → ToTensor, float [0,1], 정규화 없음 |
| source | TinyShapeCNN_3-8-16_GAP4, 합성 source train 240·valid 60장, 50 epoch 중 48 선택 |
| ImageNet 여부 | 아님 (`imagenet: false`, `target_used: false`) |

| split | circle | square | triangle | 합계 | 용도 |
|:--|--:|--:|--:|--:|:--|
| train | 120 | 40 | 20 | 180 | 학습 |
| valid | 30 | 30 | 30 | 90 | epoch·후보 선택 |
| test | 10 | 10 | 10 | 30 | 미사용 |

| 중복·그룹 점검 | 결과 |
|:--|:--|
| sha256 중복 (전체 300행) | 0건 |
| train·valid group_id 중복 | 0건 |

## 실험 조건

| 항목 | 값 |
|:--|:--|
| 후보 (동률 시 우선순위 순) | head·none → partial·none → head·weight → head·sampler |
| scratch 보조 후보 | 미실행 |
| seed / thread | 14 / CPU 1-thread |
| epoch 예산 | 후보 모두 8, best epoch = valid loss 최소 |
| optimizer | AdamW, weight_decay 1e-4 |
| 학습률 | Head 0.01, Backbone(partial) 0.001 |
| batch | train 16, valid 32 |
| BN 정책 | running 통계 고정 |
| 데이터 순서 | train: seed 14 셔플(sampler는 복원 추출) / valid: 고정 순서 |

| 후보 | 학습 범위 | 불균형 대응 | 학습 파라미터 |
|:--|:--|:--|--:|
| head·none | Head | 없음 | 771 |
| partial·none | 마지막 Conv+BN affine+Head | 없음 | 1,955 |
| head·weight | Head | 클래스 가중 손실 [0.3, 0.9, 1.8] | 771 |
| head·sampler | Head | WeightedRandomSampler | 771 |

## 선택 근거

선택 규칙: Macro F1 최대 → valid loss 최소 → 후보 순서. test 사용 False.

| 후보 | Macro F1 | Recall (circle / square / triangle) | valid loss | best epoch | 파라미터 | 시간(초, 참고) | 결과 |
|:--|--:|:--|--:|--:|--:|--:|:--|
| head·none | 1.0 | 1.0 / 1.0 / 1.0 | 0.012641 | 8 | 771 | 38.48* | |
| partial·none | 1.0 | 1.0 / 1.0 / 1.0 | 0.008148 | 8 | 1,955 | 2.52 | |
| head·weight | 1.0 | 1.0 / 1.0 / 1.0 | 0.006550 | 8 | 771 | 2.67 | 2위 |
| **head·sampler** | **1.0** | 1.0 / 1.0 / 1.0 | **0.006475** | 8 | 771 | 2.80 | **선택** |

\* 첫 실행에서 이미지를 처음 읽는 비용이 섞인 값. 시간은 선택에 쓰지 않는다.

| epoch | head·none | partial·none | head·weight | head·sampler |
|--:|--:|--:|--:|--:|
| 1 | 0.1587 / 0.956 | 0.1918 / 0.945 | 0.0825 / 0.978 | 0.0590 / 1.000 |
| 4 | 0.0210 / 1.000 | 0.0150 / 1.000 | 0.0098 / 1.000 | 0.0198 / 0.989 |
| 8 | 0.0126 / 1.000 | 0.0081 / 1.000 | 0.0066 / 1.000 | 0.0065 / 1.000 |

(칸 = valid loss / valid Macro F1)

- 네 후보 모두 F1 1.0 동률 → loss 최소인 head·sampler 선택.
- 2위와 loss 차이는 0.000075로 매우 작다. 단일 seed라 우열이 확정적이지 않다.

## 오류와 설명

| 항목 | 결과 |
|:--|:--|
| 혼동행렬 (행=실제, 열=예측) | [[30,0,0], [0,30,0], [0,0,30]] |
| 실제 오답 | **0건** |
| error_*.png, cam_wrong.png | 생성 안 함 (없는 오류를 만들지 않음) |

## 한계

| 한계 | 영향 |
|:--|:--|
| 합성 도형 3클래스 | 실제 사진에 그대로 일반화된다고 볼 수 없음 |
| valid 클래스당 30장 | 1장 오답이 Recall 3.3%p 변화 |
| 같은 valid로 4후보 반복 선택 | valid 1.0은 낙관적 추정 |
| 모든 후보 valid 1.0 포화 | 지표가 후보를 구분하지 못하고 loss 차이로 선택 |
| 단일 seed | 후보 순위가 seed에 따라 바뀔 수 있음 |
| Grad-CAM | 선택 logit의 반응 지도일 뿐 인과 증거 아님 |
| test 평가 | 미실행 |
| ImageNet ResNet18 (`resnet18_official.py`) | 미실행·미검증 (공식 가중치 다운로드·CUDA 코드 포함) |
| 실제 서비스 데이터 검증 | 미실행 |

## 재현

| 항목 | 값 |
|:--|:--|
| 실행 명령 | `python main.py` → `python restore.py` |
| 환경 | Windows 11, Python 3.13.16, torch 2.14.0+cu130(CUDA 없음, CPU 실행), torchvision 0.29.0, scikit-learn 1.8.0, OpenCV 4.13.0.92 |
| Git 변경 기록 | Git 저장소 없음 |
| 변경 내용 | D07 후보 4개 / D08 선택 키 `(macro_f1, -loss)` / D09 state_dict·class_to_idx·입력 계약·선택 요약 저장 / D10 `load_state_dict(saved["model"])` |
| best_model.pt | 14,201 bytes, SHA-256 `c966e0ebf447b10e2d7ec9f529ef893a1f1285f2bdb8e482083c27772c473c34` |

| 복원 확인 (새 프로세스) | 결과 |
|:--|:--|
| ID 순서 | True |
| 확률 일치 (atol 1e-6) | True |
| 예측 일치 | True |

| expected 대조 | 결과 |
|:--|:--|
| metrics.csv / history.csv / predictions.csv | 일치 (수치 최대 차 1.5e-07, seconds 제외) |
| confusion.csv / cam_index.json / metrics.json report | 일치 |
| cam_correct.png | 픽셀 일치 |
