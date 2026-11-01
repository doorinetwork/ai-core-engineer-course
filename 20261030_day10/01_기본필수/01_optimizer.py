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
# 손실 함수 정하기
# -----------------------------------------------------------------------------
# loss_fn (type: nn.CrossEntropyLoss): 분류용 손실 함수이다.
# 모델 출력(logit 2개)에 내부적으로 softmax를 적용하고 정답 확률의 -log를 평균한다.
# 그래서 모델 끝에 softmax를 따로 넣지 않는다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# Optimizer 후보마다 모델 만들기 → Optimizer 만들기 → 학습 → valid 평가
# -----------------------------------------------------------------------------
# name (type: str): 비교할 Optimizer 이름("SGD" 또는 "AdamW")이다. 두 후보를 같은 절차로 차례대로 돌린다.
for name in ["SGD", "AdamW"]:
    # 후보마다 같은 초기 가중치에서 다시 시작한다. 앞 모델을 이어 학습하지 않는다.
    # 전역 난수 시드를 42로 고정한다. 바로 아래 nn.Linear의 가중치 초기값이 이 난수로 정해지므로,
    # 두 후보가 똑같은 출발점을 갖게 되어 결과 차이가 오직 Optimizer 때문임을 보장한다.
    torch.manual_seed(42)
    # model (type: nn.Sequential): 입력 2개 → 은닉 8개 → ReLU → 출력 2개(class 0, 1의 logit)인 작은 신경망이다.
    # 반복마다 새로 만들어 앞 후보의 학습 결과를 물려받지 않는다.
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    if name == "SGD":
        # SGD의 학습할 파라미터는 model.parameters()이다. lr=0.01, momentum=0.0으로 만든다.
        # 학습률은 기본값에 기대지 않고 명시하며, 두 Optimizer 모두 weight_decay=0.0으로 비교한다.
        # optimizer (type: torch.optim.SGD): 새로 만든 model의 가중치를 갱신할 SGD이다.
        # 갱신 규칙은 "가중치 -= lr × gradient"이다.
        #   - model.parameters(): 이 반복에서 새로 만든 model의 가중치를 넘긴다. 그래야 이 모델이 학습된다.
        #   - lr=0.01: 한 번에 움직이는 보폭이다. SGD 기본값은 0.001이라 반드시 명시한다.
        #   - momentum=0.0: 이전 이동 방향을 섞지 않는 순수 SGD라는 뜻이다.
        #   - weight_decay=0.0: 가중치를 0 쪽으로 끌어당기는 감쇠를 끈다.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    else:
        # AdamW를 같은 lr=0.01로 만든다. weight_decay=0.0을 반드시 적는다.
        # 이번에는 감쇠 효과가 아닌 갱신 규칙을 비교한다. 이 공통 lr이 두 알고리즘의 최적 lr이라는 뜻은 아니다.
        # optimizer (type: torch.optim.AdamW): 새로 만든 model의 가중치를 갱신할 AdamW이다.
        # gradient의 이동 평균(방향)과 제곱 이동 평균(크기)을 써서 파라미터마다 보폭을 자동으로 조절한다.
        #   - lr=0.01: SGD와 같은 값으로 맞춰 조건을 같게 한다.
        #   - weight_decay=0.0: AdamW의 기본값은 0.01이라, 0.0을 적어야 SGD와 감쇠 조건이 같아진다.
        #   - betas, eps는 기본값을 쓴다.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    for epoch in range(20):
        # 학습 모드로 바꾼다. 이 모델에는 Dropout·BatchNorm이 없어 동작 차이는 없지만,
        # 학습과 평가 모드를 구분하는 습관을 지키는 코드이다.
        model.train()
        # 지난 갱신에서 쌓인 gradient를 0으로 지운다.
        # PyTorch는 gradient를 더해 쌓으므로 지우지 않으면 이전 값이 섞인다.
        optimizer.zero_grad()
        # loss (type: torch.Tensor, 스칼라): train 전체의 순전파 결과(logit)와 정답으로 계산한 평균 손실이다.
        loss = loss_fn(model(X_train), y_train)
        # 역전파로 손실에 대한 각 가중치의 gradient를 계산해 저장한다.
        loss.backward()
        # 저장된 gradient로 가중치를 한 번 갱신한다. 이 줄에서 SGD와 AdamW의 갱신 규칙 차이가 나타난다.
        optimizer.step()
    # 평가 모드로 바꾼다. 학습이 끝났으니 valid 손실을 잰다.
    model.eval()
    # 평가에는 gradient가 필요 없으므로 기록을 끈다.
    with torch.no_grad():
        # valid_loss (type: float): valid 데이터의 평균 손실이다.
        valid_loss = loss_fn(model(X_valid), y_valid).item()
    print(name, "20 epochs / valid loss:", round(valid_loss, 6))
