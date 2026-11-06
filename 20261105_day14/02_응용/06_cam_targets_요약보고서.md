# Grad-CAM target 비교 실행 결과 보고서

> **프로젝트:** Day 14 전이학습 응용 실습 06 — 같은 이미지에서 target(circle / square)을 바꾼 Grad-CAM  
> **문서 유형:** 기술 검증 결과 요약 / 교육용 실습 평가  
> **작성일:** 2026-11-04  
> **판정:** **기능 요구사항 충족(PASS) / README 정답 실행 예 2줄 일치.**  
> **평가 대상:** `06_cam_targets.py`(TODO A06), `outputs/target_0_cam.png`, `outputs/target_1_cam.png`

---

## 1. Executive Summary

본 실습의 목적은 Grad-CAM이 **선택한 logit 하나에 대한 반응 지도**라는 점을 확인하는 것이다. 같은 valid 이미지에서 target을 circle(0)과 square(1)로 바꿔 지도가 달라지는지 본다. 이를 위해 TODO A06에서 `gradient = torch.autograd.grad(out[0, target], features)[0]`을 완성하였다.

**실제 실행 결과 기준**, valid 0번(circle_000.png, 초기 모델 예측 circle)에서 **circle logit 1.5821, mean CAM 0.1275**, **square logit −1.351, mean CAM 0.1719**였다. 두 지도의 픽셀 평균 절대 차이는 0.2794로 지도가 분명히 달랐다(노트북 측정). circle 지도는 원의 왼쪽 아래·오른쪽 아래 가장자리에, square 지도는 이미지 위·아래 가장자리와 원의 위·오른쪽 일부에 반응했다(그림 관찰).

**README 질문 요약 (2.3절 상세):**
- **Heatmap의 target:** 미분한 logit의 클래스다. 이 실습은 예측과 상관없이 target 0(circle), 1(square)을 직접 지정했다.
- **target이 바뀌면 지도가 달라지는가:** 달라질 수 있고, 이번 실행에서 실제로 달라졌다. 채널 가중치가 target logit의 gradient로 정해지기 때문이다.

**최종 판단:** target별 Grad-CAM 계산의 실습 완료 조건을 충족했다(PASS). 단, Grad-CAM은 원인·정답·탐지 박스가 아니다.

## 2. 목적 및 검증 범위

### 2.1 검증 목적

- Sequential을 features(model[:7]) 앞뒤로 나누고 `autograd.grad`로 target logit의 gradient를 구한다.
- 채널 가중치(gradient 공간 평균) × features → ReLU → 32×32 확대로 CAM을 만든다.
- 같은 이미지에서 target만 바꿔 지도·logit·평균 CAM을 비교한다.

### 2.2 검증 범위

| 구분 | 포함 | 제외 |
|:--|:--|:--|
| 모델 | 05와 같은 초기 Head-only(lr 0.01, 8회) | 07 최종 모델 |
| 이미지 | valid 0번 `circle_000.png` 1장 | 다른 이미지, test |
| target | 0(circle), 1(square) | 2(triangle) |
| 기준값 | README 실습 06 정답 실행 예(참고값) | 정답 위치(없음) |

**TODO 정답:** A06 `gradient = torch.autograd.grad(out[0, target], features)[0]`. 원본 `torch.zeros_like(features)`는 채널 가중치가 모두 0이라 CAM이 항상 0이고 mean CAM도 0이 된다. `out[0, target]`은 scalar 하나여야 하며, `autograd.grad`는 파라미터 `.grad`에 누적하지 않고 (1,16,16,16) gradient tuple을 돌려준다. target마다 순전파를 새로 하므로 그래프 재사용 문제가 없다. TODO 외 코드는 원본을 유지하고 줄 나눔만 했다(AST 동일).

### 2.3 README 질문 답변

#### Q1. Heatmap의 target은 무엇인가?

**미분 대상으로 고른 logit의 클래스**다. 이 코드에서는 `out[0, target]`, 즉 valid 0번 이미지의 target 클래스 점수 하나를 features로 미분한다. target은 예측 클래스일 필요가 없다. 이 실습은 `for target in [0, 1]`로 circle과 square를 직접 지정했다. 07에서는 `logits_one.argmax(1)`, 즉 **예측 클래스**를 target으로 쓴다. 따라서 지도를 보고할 때는 반드시 "어느 target에 대한 지도인지"를 함께 적어야 한다. 지도는 "이 logit을 키우는 방향으로 양의 기여를 한 특징 위치"를 보여 줄 뿐이며, 원인·정답·탐지 박스가 아니다.

#### Q2. 같은 이미지에서 target이 바뀌면 지도가 달라질 수 있는가?

달라질 수 있다. CAM = ReLU(Σₖ αₖ·Aₖ)에서 features Aₖ는 target과 무관하게 같지만, 채널 가중치 αₖ = gradient의 공간 평균은 **target logit을 미분해 구하므로 target마다 다르다**. Head가 선형층이므로 αₖ는 사실상 Head에서 그 target 행의 가중치를 공간 평균한 값이 되어, 어떤 채널은 circle에는 양(+), square에는 음(−)이 된다. 이번 실행에서도 두 지도의 평균 절대 차이가 0.2794였고 반응 위치도 달랐다(아래 3.2). square logit은 −1.351로 음수였지만, 지도는 "square 점수를 올리는 쪽으로 기여한 위치"를 보여 주므로 0이 아니었다(mean CAM 0.1719). 반대로 어떤 target에 양의 기여가 전혀 없으면 지도가 전부 0일 수 있으며, 이를 예쁜 지도로 바꾸지 않는다.

## 3. 처리 결과 상세

### 3.1 처리 흐름

| 순서 | 처리 | 적용 설정 | 기대 기능 / 효과 |
|:--:|:--|:--|:--|
| 1 | 초기 모델 학습 | 05와 동일(Head-only, lr 0.01, 8회) | 예측 circle |
| 2 | 순전파 분할 | `features = model[:7](image)`, `out = model[7:](features)` | (1,16,16,16), (1,3) |
| 3 | gradient | `autograd.grad(out[0, target], features)[0]` (TODO A06) | (1,16,16,16) |
| 4 | CAM | `mean((2,3))` 가중치 × features → sum(1) → ReLU | (1,16,16) |
| 5 | 확대·정규화 | bilinear 32×32, ÷ max(최댓값, 1e-8) | [0,1] 지도 |
| 6 | 합성·저장 | JET 열지도 0.4 + 원본 0.6, 256×256 | `target_{0,1}_cam.png` |

### 3.2 target별 결과

| target | logit | mean CAM | 반응 위치(그림 관찰) |
|:--|--:|--:|:--|
| 0 circle | 1.5821 | 0.1275 | 원의 왼쪽 아래 가장자리(가장 강함)와 오른쪽 아래 가장자리 |
| 1 square | −1.351 | 0.1719 | 이미지 위·아래 가장자리 띠, 원의 위쪽·오른쪽 일부 |

| target 0 (circle) | target 1 (square) |
|:--:|:--:|
| ![target 0 CAM](outputs/target_0_cam.png) | ![target 1 CAM](outputs/target_1_cam.png) |

- circle 지도는 둥근 경계의 아래쪽 곡선에 몰려 있다. "곡선 경계가 circle 점수를 올렸다"는 **가설**과 맞지만, 원의 위쪽 곡선에는 반응이 약해 경계 전체를 본다고 단정할 수 없다.
- square 지도는 도형이 아닌 이미지 테두리 쪽에도 넓게 퍼져 mean CAM이 더 크다. mean CAM이 크다고 더 "중요한" 지도인 것은 아니며, 값은 각 지도의 최댓값으로 정규화한 상대값이다.
- 두 지도의 평균 절대 차이 0.2794(노트북 측정).

### 3.3 표준 출력 (`outputs/logs/outputs_06_cam_targets.log`)

```text
target: circle logit: 1.5821 mean CAM: 0.1275
target: square logit: -1.351 mean CAM: 0.1719
```

README 실습 06 정답 실행 예와 같다.

## 4. 품질 검증 및 합격 기준

### 4.1 합격 기준 대비 결과

| 항목 | 합격 기준(README) | 실제 실행 결과 | 판정 |
|:--|:--|:--|:--:|
| circle logit / mean CAM | 1.5821 / 0.1275 | **동일** | PASS |
| square logit / mean CAM | −1.351 / 0.1719 | **동일** | PASS |
| 생성 파일 | `target_0_cam.png`, `target_1_cam.png` | **2개, 256×256** | PASS |
| 좌표 일치 | 32×32 지도를 같은 32×32 이미지에 합성 | **동일 좌표 합성** | PASS |
| 지도 품질 | 기준 없음 | 정답 위치 없음 | 미평가 |

> **주의:** `inference_mode()` 안에서는 Grad-CAM을 계산할 수 없다. `eval()` 상태에서는 미분이 가능하다.

### 4.2 기능 검증 (Acceptance Test)

| TC ID | 검증 내용 | 관찰 근거 | 결과 |
|:--|:--|:--|:--:|
| TC-01 | 표준 출력 2줄 | 로그 = README 정답 예 | PASS |
| TC-02 | gradient 비영(非零) | mean CAM > 0 (원본 zeros_like면 0) | PASS |
| TC-03 | target별 차이 | 두 지도 평균 절대 차이 0.2794 | PASS |
| TC-04 | 파라미터 grad 누적 없음 | `autograd.grad` 사용(`.backward()` 아님) | PASS |
| TC-05 | 로직 보존 | 원본 대비 TODO 1줄 외 AST 동일 | PASS |

## 5. 리스크 및 개선 과제

| 우선순위 | 이슈 / 잠재 영향 | 개선 과제 | 검증 산출물 |
|:--:|:--|:--|:--|
| P0 | 수업 venv에서 결과 확인 안 함 | `python 06_cam_targets.py` 실행 | 출력 2줄 |
| P1 | 지도를 원인으로 오해할 위험 | 보고서에 target·logit·정규화 방식 함께 기록 | 지도 설명표 |
| P1 | 16×16 features → 32×32 확대로 해상도 낮음 | 더 이른 층 지도와 비교(심화) | 층별 비교 그림 |
| P2 | 이미지 1장만 확인 | 클래스별 여러 이미지에서 반복 | 지도 모음 |
| P2 | 이미지 테두리 반응(square) 원인 미확인 | 테두리를 바꾼 입력으로 반응 변화 확인 | 비교 실험 기록 |

## 6. 다음 단계: 현업 적용 승인 전 검증 계획

| 단계 | 검증 항목 | 방법 | 통과 조건 |
|:--:|:--|:--|:--|
| 1 | 환경 재현 | 수업 venv 실행 | 2줄 동일 |
| 2 | 다수 사례 | 클래스별 10장 이상 지도 | 반응 위치 경향 기록 |
| 3 | 편향 가설 검증 | 배경·테두리를 바꾼 입력으로 logit 변화 측정 | 가설 채택/기각 근거 |
| 4 | 정량 점검 | 도형 마스크와 CAM 겹침 비율(마스크가 있을 때) | 기준을 사전에 정해 기록 |
| 5 | 보고 기준 | 모든 지도에 target·모델·정규화 표기 | 보고서 검토 통과 |

## 7. 최종 판단

- **기능 검증:** TODO A06을 완성해 target별 Grad-CAM을 계산했고 출력이 README와 같았다.
- **관찰:** 같은 이미지에서 target만 바꿔도 지도가 달라졌다(평균 절대 차이 0.2794).
- **한계:** 이미지 1장, 정답 위치 없음. 지도는 반응 지도이며 원인이 아니다.
- **판단:** 본 실습은 **완료(PASS)**로 분류한다.

---

## 부록 A. 근거 파일 및 검증 이력

### A.1 근거 파일

| 파일 | 역할 |
|:--|:--|
| `06_cam_targets.py` | TODO A06 완성 + 전체 주석 |
| `outputs/target_0_cam.png`, `outputs/target_1_cam.png` | **실제 실행 CAM 합성 이미지** |
| `outputs/logs/outputs_06_cam_targets.log` | **실제 실행 표준 출력** |
| `../06_cam_targets.ipynb` | 원본·두 지도 나란히 보기, 지도 차이 측정 |
| `../README.md` | 실습 06 정답 실행 예와 설명 질문(참고값) |

### A.2 검증 이력

| 일자 | 내용 | 결과 |
|:--|:--|:--|
| 2026-11-04 | `06_cam_targets.py` 실행, README 정답 예와 대조 | 2줄 일치 |
| 2026-11-04 | 노트북 실행: 두 지도 평균 절대 차이 | 0.2794 |
| 2026-11-04 | CAM PNG 육안 관찰 | 3.2절 기록 |

- **재현 명령:** `python 06_cam_targets.py`
- **보고 원칙:** **측정된 사실**, **그 사실에서 계산한 값**, **아직 검증되지 않은 가정**을 구분하여 작성했다.

*교육용 실습 결과 보고서 — 2026-11-04 실제 실행 산출물(`outputs/`)과 README 실습 06을 근거로 작성했으며 운영 승인 문서가 아니다.*
