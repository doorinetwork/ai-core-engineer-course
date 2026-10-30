# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [외부 라이브러리] pandas: 표 형태(DataFrame) 데이터를 다루는 라이브러리이다.
import pandas as pd
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산·자동미분 라이브러리이다.
import torch
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
from torch import nn

# -----------------------------------------------------------------------------
# 실행 환경 고정
# -----------------------------------------------------------------------------
# CPU 연산 스레드를 1개로 고정한다.
# 여러 스레드가 덧셈 순서를 바꾸면 소수점 끝자리가 달라질 수 있어, 결과를 매번 똑같이 재현하려고 쓴다.
torch.set_num_threads(1)
# base (type: Path): 이 Python 파일이 들어 있는 폴더의 절대 경로이다.
base = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 데이터 읽기와 train/valid 분핧
# -----------------------------------------------------------------------------
# df (type: DataFrame): x1, x2, label, split 열을 가진 점 데이터 전체이다.
df = pd.read_csv(base / "points.csv")
# train (type: DataFrame): split 값이 "train"인 행만 고른 학습용 자료이다. 가중치를 학습하는 데 쓴다.
train = df[df["split"] == "train"]
# valid (type: DataFrame): split 값이 "valid"인 행만 고른 검증용 자료이다.
valid = df[df["split"] == "valid"]

# -----------------------------------------------------------------------------
# 텐서 만들기
# -----------------------------------------------------------------------------
# X_train (type: torch.Tensor, shape: (train 건수, 2), dtype: float32): 학습 입력 특성 x1, x2이다.
# dtype=torch.float32인 이유: nn.Linear의 가중치가 float32이므로 입력도 같은 float32여야 행렬곱이 된다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# y_train (type: torch.Tensor, shape: (train 건수,), dtype: int64): 학습 정답 class 번호(0 또는 1)이다.
# dtype=torch.long인 이유: CrossEntropyLoss는 정답을 class 번호(정수 int64)로 받도록 정해져 있기 때문이다.
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# X_valid (type: torch.Tensor, shape: (valid 건수, 2), dtype: float32): 검증 입력 특성이다.
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
# y_valid (type: torch.Tensor, shape: (valid 건수,), dtype: int64): 검증 정답 class 번호이다.
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않는다.
# 이유: 지금은 학습률 후보를 "고르는" 단계라 valid로만 비교한다.
# test를 여기서 보면 test 점수에 맞춰 lr을 고르게 되어 최종 평가가 공정하지 않게 된다.

# -----------------------------------------------------------------------------
# 학습률 후보 정하기
# -----------------------------------------------------------------------------
# learning_rates (type: list[float]): 비교할 학습률 후보 목록이다.
#   - 0.001: AdamW의 기본 lr 값이다. 한 번에 조금씩 움직여 안정적이지만 loss가 천천히 줄어든다.
#   - 0.01: 0.001의 10배 보폭이다. loss가 더 빨리 줄 수 있지만 너무 크면 최적점을 지나쳐 흔들린다.
# 10배 간격으로 고른 이유: 학습률은 크기의 "자릿수"에 따라 결과가 크게 달라지므로
# 후보를 로그 간격(0.001, 0.01, ...)으로 두고 비교하는 것이 기본이다.
# 이 실험에서 바꾸는 것은 이 리스트의 값 하나뿐이고, 나머지 조건은 모두 같게 둔다.
learning_rates = [0.001, 0.01]

# -----------------------------------------------------------------------------
# 손실 함수 정하기
# -----------------------------------------------------------------------------
# loss_fn (type: nn.CrossEntropyLoss): 분류용 손실 함수이다.
# 모델 출력(logit 2개)에 내부적으로 softmax를 적용하고 정답 확률의 -log를 평균한다.
# 그래서 모델 끝에 softmax를 따로 넣지 않는다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# 학습률 후보마다 모델 만들기 → Optimizer 만들기 → 학습하며 epoch별 valid loss 기록
# -----------------------------------------------------------------------------
# lr (type: float): 이번 반복에서 쓸 학습률이다. 0.001 → 0.01 순서로 같은 절차를 반복한다.
for lr in learning_rates:
    # 후보마다 같은 초기 가중치에서 다시 시작한다. 앞 후보의 모델을 이어 학습하지 않는다.
    # 전역 난수 시드를 42로 고정한다. 바로 아래 nn.Linear의 가중치 초기값이 이 난수로 정해지므로,
    # 두 후보가 똑같은 출발점을 갖게 되어 결과 차이가 오직 lr 때문임을 보장한다.
    torch.manual_seed(42)
    # model (type: nn.Sequential): 입력 2개 → 은닉 8개 → ReLU → 출력 2개(class 0, 1의 logit)인 작은 신경망이다.
    # 반복마다 새로 만들어 앞 후보의 학습 결과를 물려받지 않는다.
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    # optimizer (type: torch.optim.AdamW): 새로 만든 model의 가중치를 갱신할 AdamW이다.
    #   - model.parameters(): 이 반복에서 새로 만든 model의 가중치를 넘긴다. 그래야 이 모델이 학습된다.
    #   - lr=lr: 이번 후보의 학습률을 넣는다. 반복마다 달라지는 유일한 값이다.
    #   - weight_decay=0.0: AdamW의 기본값은 0.01이라 0.0을 적어 감쇠를 끈다.
    #     감쇠가 켜져 있으면 lr과 감쇠 효과가 섞여, 차이가 lr 때문인지 가려낼 수 없다.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    # losses (type: list[float]): epoch마다 잰 valid loss를 차례대로 담는 빈 리스트이다.
    # 마지막 한 점만이 아니라 학습곡선 전체(감소 속도·흔들림)를 보려고 epoch마다 기록한다.
    losses = []
    for epoch in range(20):
        # 학습 모드로 바꾼다. 이 모델에는 Dropout·BatchNorm이 없어 동작 차이는 없지만,
        # 아래에서 eval()로 바꾸므로 매 epoch 학습 전에 다시 학습 모드로 돌려놓는 코드이다.
        model.train()
        # 지난 갱신에서 쌓인 gradient를 0으로 지운다.
        # PyTorch는 gradient를 더해 쌓으므로 지우지 않으면 이전 epoch의 값이 섞인다.
        optimizer.zero_grad()
        # train 전체로 순전파(logit 계산) → 평균 손실 계산 → 역전파를 한 줄로 수행한다.
        # backward()가 손실에 대한 각 가중치의 gradient를 계산해 param.grad에 저장한다.
        loss_fn(model(X_train), y_train).backward()
        # 저장된 gradient와 lr로 가중치를 한 번 갱신한다. lr이 클수록 이 한 번의 이동 폭이 커진다.
        optimizer.step()
        # 갱신이 끝난 모델의 valid loss를 기록한다. valid로 backward하지 않는다.
        # 평가 모드로 바꾼다. 방금 갱신된 가중치로 valid 손실을 잰다.
        model.eval()
        # 평가에는 gradient가 필요 없으므로 기록을 끈다.
        # 메모리와 계산을 아끼고, valid 데이터가 가중치 갱신에 섞이지 않게 한다.
        with torch.no_grad():
            # valid 평균 손실을 계산하고 .item()으로 파이썬 float로 꺼내 losses 끝에 붙인다.
            losses.append(loss_fn(model(X_valid), y_valid).item())

       # lr 값과 처음 5개 epoch의 valid loss를 소수점 6자리로 반올림해 출력한다.
    
    # losses[:5]: 앞에서 5개(epoch 0~4)를 자른다. 초반에 loss가 얼마나 빨리 줄어드는지 비교하는 데 쓴다.
    print("lr:", lr, "처음 5개:", [round(x, 6) for x in losses[:5]])
    # losses[-1]: 리스트의 마지막 값, 즉 20번째 epoch 뒤 valid loss이다. 최종적으로 어디까지 내려갔는지 본다.
    print("마지막 valid loss:", round(losses[-1], 6))
