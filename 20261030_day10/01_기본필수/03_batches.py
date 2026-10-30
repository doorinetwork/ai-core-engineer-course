# -----------------------------------------------------------------------------
# 라이브러리 불러오기와 실행 환경 고정
# -----------------------------------------------------------------------------
# 외부 라이브러리(PyTorch). 목적: Tensor 생성·Optimizer·난수 고정. 기능: GPU/CPU 텐서 연산과 자동 미분을 제공한다.
import torch
# 외부 라이브러리(PyTorch). 목적: train 데이터를 mini-batch로 나눠 공급하기. 
# 기능: TensorDataset은 (X, y) 쌍을 묶고, DataLoader는 그것을 batch 단위로 섞어서 꺼내 준다.
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# 실습용 데이터 만들기
# -----------------------------------------------------------------------------
# 학습용 공통 데이터와 별개인 '배치 개수만 확인하는' 20개 숫자이다.
# 이 실습은 모델을 학습하지 않고, batch가 어떻게 나뉘는지만 눈으로 확인하는 것이 목적이다.
# X (type: torch.Tensor, shape: (20, 2), dtype: float32): 20개 sample의 입력 특성 2개이다.
# torch.arange(40)은 0, 1, ..., 39의 40개 숫자를 만들고, reshape(20, 2)는 이를 20행 2열로 접는다.
# 그래서 0행은 [0, 1], 1행은 [2, 3], ..., 19행은 [38, 39]가 된다.
# dtype=torch.float32인 이유: 실제 학습에서 nn.Linear의 가중치가 float32이므로 입력도 float32로 맞추는 관례를 따른다.
X = torch.arange(40, dtype=torch.float32).reshape(20, 2)
# y (type: torch.Tensor, shape: (20,), dtype: int64): 20개 sample의 정답 class 번호이다.
# torch.arange(20)은 0~19의 정수이고, % 2(2로 나눈 나머지)를 하면 0, 1, 0, 1, ...이 번갈아 나온다.
# dtype을 지정하지 않으면 정수 arange는 int64가 되며, 이는 CrossEntropyLoss가 요구하는 정답 형식과 같다.
y = torch.arange(20) % 2
# dataset (type: TensorDataset, 길이: 20): X와 y를 행 번호끼리 짝지은 dataset이다.
# dataset[i]를 꺼내면 (X[i], y[i]) 튜플이 나온다. X와 y의 첫 번째 차원(20)이 같아야 만들 수 있다.
dataset = TensorDataset(X, y)

# -----------------------------------------------------------------------------
# DataLoader 만들기
# -----------------------------------------------------------------------------
# DataLoader(dataset, batch_size=6, shuffle=False)를 만든다.
# 20개를 순서대로 나누면 6·6·6·2개이다. dataset은 필수, batch_size와 shuffle은 선택 인자이다.
# loader (type: DataLoader): dataset을 batch 단위로 꺼내 주는 반복 가능한(iterable) 객체이다.
#   - dataset: 나눌 대상이다. 필수 인자이다.
#   - batch_size=6: 한 번(한 iteration)에 꺼내는 sample 개수이다. 지정하지 않으면 기본값은 1이다.
#   - shuffle=False: sample 순서를 섞지 않고 0번부터 차례대로 꺼낸다(기본값도 False이다).
#     섞지 않는 이유: 이 실습은 "6·6·6·2"로 나뉘는 모습을 확인하는 것이 목적이라 순서가 고정되어야
#     첫 batch가 0~5번, 둘째가 6~11번처럼 예측 가능하게 나온다.
#     실제 학습에서는 train loader만 shuffle=True로 두고, valid/test는 shuffle=False로 둔다.
# drop_last를 지정하지 않았으므로 기본값 False가 적용된다.
# 그래서 6개를 못 채운 마지막 2개도 버리지 않고 작은 batch 하나로 꺼낸다.
loader = DataLoader(dataset, batch_size=6, shuffle=False)

# -----------------------------------------------------------------------------
# batch 순회하며 shape 기록하기
# -----------------------------------------------------------------------------
# enumerate(loader, start=1): loader에서 batch를 하나씩 꺼내면서 1부터 시작하는 번호를 함께 붙인다.
# start=1인 이유: iteration 번호를 사람이 읽기 쉽게 0이 아닌 1부터 센다.
# iteration (type: int): 지금이 이 epoch에서 몇 번째 batch인지 나타내는 번호(1~4)이다.
# xb (type: torch.Tensor, shape: (batch 크기, 2), dtype: float32): 이번 batch의 입력 X 묶음이다.
# yb (type: torch.Tensor, shape: (batch 크기,), dtype: int64): 이번 batch의 정답 y 묶음이다.
# DataLoader는 (X[i], y[i]) 튜플 여러 개를 X끼리, y끼리 쌓아서 (xb, yb) 한 쌍으로 돌려준다.
for iteration, (xb, yb) in enumerate(loader, start=1):
    # tuple(xb.shape): torch.Size를 보기 쉬운 일반 튜플로 바꿔 출력한다. 예: (6, 2)
    # 예상 출력: iteration 1~3은 X: (6, 2) y: (6,), iteration 4는 X: (2, 2) y: (2,)이다.
    # 마지막 batch가 2개인 이유: 6×3 = 18개를 꺼내고 남은 20 - 18 = 2개가 마지막 batch가 되기 때문이다.
    print("iteration:", iteration, "X:", tuple(xb.shape), "y:", tuple(yb.shape))

# -----------------------------------------------------------------------------
# 한 epoch의 iteration 수 확인하기
# -----------------------------------------------------------------------------
# len(loader) (type: int): 한 epoch 동안 꺼내는 batch 개수, 즉 iteration 수이다.
# drop_last=False일 때 계산식은 올림(20 / 6) = 올림(3.33...) = 4이다.
# 예상 출력: 한 epoch의 iteration: 4
# 만약 drop_last=True였다면 마지막 2개를 버려 3이 되고, 그 2개 sample은 이 epoch에서 학습에 쓰이지 않는다.
print("한 epoch의 iteration:", len(loader))
