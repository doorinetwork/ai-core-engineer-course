"""실습 05: 정답 인덱스와 logits로 손실을 계산합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
import torch
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
from torch import nn

# -----------------------------------------------------------------------------
# 입력 준비: 모델 출력(logits)과 정답(target)
# -----------------------------------------------------------------------------
# logits (type: torch.Tensor, shape (2, 2), dtype float32): 모델이 낸 클래스별 원점수이다.
#   행 = 샘플 2개, 열 = 클래스 2개(클래스 0, 클래스 1)이다.
#   1번째 샘플은 클래스 0 점수가 2.0으로 더 높고, 2번째 샘플은 클래스 1 점수가 2.0으로 더 높다.
#   logits는 softmax를 거치기 전의 값이라 합이 1이 아니며 음수도 될 수 있다.
logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
# target (type: torch.Tensor, shape (2,), dtype int64): 각 샘플의 정답 클래스 번호이다.
#   1번째 샘플의 정답은 클래스 0, 2번째 샘플의 정답은 클래스 1이다.
#   dtype=torch.long으로 지정하는 이유: CrossEntropyLoss는 정답을 "클래스 인덱스"로 받을 때
#   정수형(int64, long)을 요구한다. float으로 주면 인덱스가 아니라 확률 분포로 해석되거나 오류가 난다.
#   one-hot 벡터가 아니라 번호 하나만 주는 형식이므로 shape은 (샘플 수,)이다.
target = torch.tensor([0, 1], dtype=torch.long)
# loss_fn (type: nn.CrossEntropyLoss): 다중 분류용 교차 엔트로피 손실 함수 객체이다.
#   내부에서 log_softmax(logits)를 계산한 뒤, 정답 클래스 위치의 값에 마이너스를 붙여 꺼낸다(NLLLoss).
#   기본 reduction="mean"이므로 샘플별 손실을 평균 내어 스칼라 하나를 돌려준다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# 손실 계산
# -----------------------------------------------------------------------------
# loss (type: torch.Tensor, shape () 스칼라, dtype float32): 두 샘플의 평균 교차 엔트로피 손실이다.
#   인자 순서는 반드시 (예측 logits, 정답 target)이다. 순서를 바꾸면 shape·dtype이 맞지 않아 오류가 난다.
#   softmax를 미리 적용하지 않는 이유: CrossEntropyLoss가 내부에서 log_softmax를 이미 수행한다.
#   softmax를 한 번 더 씌우면 확률에 다시 softmax를 거는 셈이 되어 손실 값이 틀어지고 기울기도 약해진다.
loss = loss_fn(logits, target)

# -----------------------------------------------------------------------------
# 결과 확인 출력
# -----------------------------------------------------------------------------
# logits.shape (type: torch.Size)를 tuple로 바꿔 출력한다. 예상 출력: logits: (2, 2)
print("logits:", tuple(logits.shape))
# target의 shape과 dtype을 출력해 정답이 1차원 정수 인덱스인지 확인한다. 예상 출력: target: (2,) torch.int64
print("target:", tuple(target.shape), target.dtype)
# loss.item() (type: float): 스칼라 텐서에서 파이썬 숫자를 꺼낸다.
# round(..., 6) (type: float): 소수점 6자리로 반올림한다. 
print("loss:", round(loss.item(), 6))
