# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산·자동미분 라이브러리이다.
import torch
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
from torch import nn

# -----------------------------------------------------------------------------
# 재현 가능한 입력과 모델 준비
# -----------------------------------------------------------------------------
# PyTorch의 난수 시드를 42로 고정한다.
torch.manual_seed(42)
# X (type: Tensor, shape (4, 2), dtype float32): 샘플 4개, 특성 2개인 입력이다.
# 1.처럼 소수점을 붙여 float32로 만든다. nn.Linear의 weight가 float32라서 입력도 float여야 행렬곱이 된다.
X = torch.tensor([[1.,2.],[2.,1.],[3.,4.],[4.,3.]])
# bn (type: nn.BatchNorm1d): 8개 특성 각각을 batch 평균·분산으로 정규화한 뒤 learnable scale(weight)·shift(bias)를 적용하는 층이다.
# num_features=8은 바로 앞 Linear의 출력 특성 수(8)와 반드시 같아야 한다. 다르면 forward에서 shape 오류가 난다.
# 내부에 running_mean·running_var(각 shape (8,))를 버퍼로 가지며, 처음 값은 running_mean=0, running_var=1이다.
# 나중에 running_mean을 직접 확인하려고 모델 안에 넣기 전에 변수 bn으로 따로 잡아 둔다.
bn = nn.BatchNorm1d(num_features=8)
# model (type: nn.Sequential): Linear(2→8) → BatchNorm1d(8) → ReLU를 순서대로 실행하는 모델이다.
# BatchNorm은 Linear 뒤, 활성화 함수(ReLU) 앞에 둔다.
# 이렇게 해야 ReLU에 들어가기 전의 값 분포가 평균 0·분산 1 근처로 맞춰져 학습이 안정된다.
model = nn.Sequential(nn.Linear(2, 8), bn, nn.ReLU())
# 모델 전체를 train(학습) 모드로 둔다. Sequential에 호출하면 안의 bn까지 모두 train 모드가 된다.
# train 모드의 BatchNorm은 현재 batch의 평균·분산으로 정규화하고, 동시에 running_mean·running_var를 갱신한다.
model.train()

# -----------------------------------------------------------------------------
# train 모드에서 forward 실행하기 (B07)
# -----------------------------------------------------------------------------
# model에 X를 넣어 output을 구한다. 입력 (4,2)는 Linear에서 (4,8)이 된다.
# BatchNorm1d(8)은 8개 특성의 통계를 유지한다.
# output (type: Tensor, shape (4, 8)): 모델을 한 번 통과한 결과이다.
# model(X)는 model.forward(X)를 호출하며, 내부에서 다음 순서로 계산한다.
#   1) Linear: (4, 2) @ weight.T (2, 8) + bias → (4, 8)
#   2) BatchNorm1d: 8개 열마다 4개 샘플의 평균·분산으로 정규화 → shape는 (4, 8) 그대로
#      이때 train 모드이므로 running_mean = 0.9 * running_mean + 0.1 * batch 평균(momentum=0.1)으로 갱신된다.
#   3) ReLU: 음수를 0으로 바꿈 → shape는 (4, 8) 그대로
# batch_size가 4라서 batch 통계를 계산할 수 있다. batch_size가 1이면 train 모드 BatchNorm1d는 분산을 구할 수 없어 오류가 난다.
output = model(X)

# -----------------------------------------------------------------------------
# shape 확인하기
# -----------------------------------------------------------------------------
# X.shape (type: torch.Size)를 tuple로 바꿔 출력한다. 입력은 (4, 2), 출력은 (4, 8)이다.
# 샘플 수 4는 그대로 유지되고 특성 수만 2에서 8로 바뀌었다는 것을 확인한다.
print("입력:", tuple(X.shape), "출력:", tuple(output.shape))
# bn.running_mean (type: Tensor, shape (8,)): 특성 8개 각각의 누적 평균이다.
# 샘플 수(4)와 무관하게 특성 수(num_features=8)만큼의 길이를 가지므로 (8,)이 출력된다.
print("running_mean:", tuple(bn.running_mean.shape))

# -----------------------------------------------------------------------------
# eval 모드에서 running 통계가 유지되는지 확인하기
# -----------------------------------------------------------------------------
# before (type: Tensor, shape (8,)): train forward 직후의 running_mean 복사본이다.
# .clone()으로 값 자체를 복사해야 한다. clone 없이 대입하면 같은 메모리를 가리켜서,
# running_mean이 바뀌어도 before가 함께 바뀌므로 비교가 의미 없어진다.
before = bn.running_mean.clone()
# 모델 전체를 eval(평가) 모드로 바꾼다.
# eval 모드의 BatchNorm은 batch 통계 대신 저장된 running_mean·running_var로 정규화하고, running 통계를 갱신하지 않는다.
# eval()은 mode만 바꾸며 gradient 계산은 끄지 않는다.
model.eval()
# torch.no_grad() 문맥 안에서는 gradient를 계산·저장하지 않는다.
with torch.no_grad():
    # 같은 X를 eval 모드로 통과시킨다. 반환값은 쓰지 않고, running_mean이 바뀌는지만 본다.
    model(X)
# torch.equal(before, bn.running_mean) (type: bool): 두 Tensor의 shape와 모든 값이 같으면 True이다.
# eval 모드에서는 running_mean을 갱신하지 않으므로 True가 출력된다.
# 만약 eval()을 빼먹고 train 모드로 통과시켰다면 running_mean이 다시 갱신되어 False가 나온다.
print("eval에서 통계 유지:", torch.equal(before, bn.running_mean))
