# -----------------------------------------------------------------------------
# 라이브러리 불러오기와 실행 환경 고정
# -----------------------------------------------------------------------------
# 표준 라이브러리. 목적: 데이터·출력 폴더 위치 찾기. 기능: 운영체제와 상관없이 파일 경로를 객체로 다룬다.
from pathlib import Path
# 외부 라이브러리. 목적: CSV 읽기, epoch 기록 표 만들기, 결과 CSV 저장. 기능: 표(DataFrame)를 다룬다.
import pandas as pd
# 외부 라이브러리(PyTorch). 목적: Tensor 생성·Optimizer·난수 고정. 기능: GPU/CPU 텐서 연산과 자동 미분을 제공한다.
import torch
# 외부 라이브러리(PyTorch의 신경망 모듈). 목적: 층(Linear, ReLU, Dropout, BatchNorm1d)과 손실 함수 만들기. 기능: 신경망 구성 요소를 모아 둔 모듈이다.
from torch import nn

# -----------------------------------------------------------------------------
# 실행 환경 고정
# -----------------------------------------------------------------------------
# CPU 연산 스레드를 1개로 고정한다. 여러 스레드가 덧셈 순서를 바꾸면 소수점 끝자리가 달라질 수 있어, 두 후보의 결과를 매번 똑같이 재현하려고 쓴다.
torch.set_num_threads(1)
# base (type: Path) 이 .py 파일이 있는 폴더의 절대 경로다. 어디서 실행하든 같은 폴더 기준으로 data/와 outputs/를 찾게 한다.
base = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 추가 라이브러리
# -----------------------------------------------------------------------------
# 외부 라이브러리. Python 시각화 라이브러리.
import matplotlib
# 창을 띄우지 않고 PNG에 저장한다. "Agg"는 화면 없이 이미지 파일만 만드는 backend라 서버·터미널 환경에서도 오류 없이 동작한다. pyplot을 불러오기 전에 정해야 적용된다.
matplotlib.use("Agg")  
# 외부 라이브러리. 목적: loss 곡선 그림 그리기. 기능: plot, axvline, savefig 같은 그리기 함수를 제공한다.
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 데이터 읽기 및 train/valid 분할
# -----------------------------------------------------------------------------
# df (type: DataFrame) x1, x2, label, split 열을 가진 점 데이터 전체를 읽는다.
df = pd.read_csv(base / "points.csv")
# train (type: DataFrame) split 값이 "train"인 행만 고른다. 가중치를 학습하는 데 쓰는 데이터다.
train = df[df["split"] == "train"]
# valid (type: DataFrame) split 값이 "valid"인 행만 고른다. 학습에 쓰지 않고 두 momentum 설정을 비교하는 데만 쓴다.
valid = df[df["split"] == "valid"]
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
# 이유: 지금은 lr 설정과 best epoch를 "고르는" 단계라 valid로만 비교한다.
# test를 여기서 보면 test 점수에 맞춰 설정을 고르게 되어 최종 평가가 공정하지 않게 된다.

# -----------------------------------------------------------------------------
# 손실 함수
# -----------------------------------------------------------------------------
# 실습 07~08: 기준 lr과 비교 lr만 다르다. 더 나쁘게 나와도 그대로 보고한다.
# loss_fn (type: CrossEntropyLoss) 분류용 손실 함수다. 
# 모델 출력(logit 2개)에 내부적으로 softmax를 적용하고 정답 확률의 -log를 평균한다. 그래서 모델 끝에 softmax를 따로 넣지 않는다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# 설정마다: 모델 새로 만들기 → Optimizer 만들기 → 20 epochs 학습하며 history 기록
# -----------------------------------------------------------------------------
# summary (type: list[list]): 설정마다 [이름, lr, best epoch, best valid loss] 한 줄씩 모으는 빈 리스트이다.
summary = []
# name (type: str): 설정 이름이다. 결과 파일 이름(history_baseline.csv 등)과 비교표의 setting 열에 쓴다.
# lr (type: float): 이번 설정의 학습률이다.
#   - baseline 0.001: AdamW의 기본 lr이다. 한 번에 조금씩 움직여 안정적이지만 loss가 천천히 줄어든다.
#   - candidate 0.01: baseline의 10배 보폭이다. loss가 더 빨리 줄 수 있지만 너무 크면 최적점을 지나쳐 흔들린다.
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
    # 전역 난수 시드를 42로 고정한다. 바로 아래 nn.Linear의 가중치 초기값이 이 난수로 정해지므로,
    # 두 설정이 똑같은 출발점을 갖게 되어 결과 차이가 오직 lr 때문임을 보장한다.
    torch.manual_seed(42)
    # model (type: nn.Sequential): 입력 2개 → 은닉 8개 → ReLU → 출력 2개(class 0, 1의 logit)인 작은 신경망이다.
    # 설정마다 처음부터 새로 만들어, 앞 설정에서 학습된 가중치를 물려받지 않는다.
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    # optimizer (type: torch.optim.AdamW): 새로 만든 model의 가중치를 갱신할 AdamW이다.
    #   - model.parameters(): 이번 반복에서 새로 만든 model의 가중치를 넘긴다. 그래야 이 모델이 학습된다.
    #   - lr=lr: 이번 설정의 학습률을 넣는다. 두 설정 사이에서 달라지는 유일한 값이다.
    #   - weight_decay=0.0: AdamW의 기본값은 0.01이라 0.0을 적어 감쇠를 끈다.
    #     감쇠가 켜져 있으면 lr과 감쇠 효과가 섞여, 차이가 lr 때문인지 가려낼 수 없다.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    # history (type: list[list]): epoch마다 [epoch 번호, train loss, valid loss]를 한 줄씩 담는 빈 리스트이다.
    # 마지막 한 점만이 아니라 학습곡선 전체를 보고 best epoch를 찾기 위해 epoch마다 기록한다.
    history = []
    for epoch in range(1, 21):
        # 학습 모드로 바꾼다. 이 모델에는 Dropout·BatchNorm이 없어 동작 차이는 없지만,
        # 아래에서 eval()로 바꾸므로 매 epoch 학습 전에 다시 학습 모드로 돌려놓는 코드이다.
        model.train()
        # 지난 갱신에서 쌓인 gradient를 0으로 지운다.
        # PyTorch는 gradient를 더해 쌓으므로 지우지 않으면 이전 epoch의 값이 섞인다.
        optimizer.zero_grad()
        # loss (type: torch.Tensor, shape: (), 0차원 스칼라): train 전체로 순전파해 계산한 평균 손실이다.
        # 아직 gradient 기록이 켜져 있으므로, 이 loss에서 역전파해 각 가중치의 gradient를 구할 수 있다.
        loss = loss_fn(model(X_train), y_train)
        # 이미 zero_grad와 loss 계산이 끝났다. loss.backward() 다음 optimizer.step()을 호출한다.
        # 학습 데이터만 사용한다. 비교 후보마다 모델을 처음부터 새로 만든다.
        # 역전파: train loss에 대한 각 가중치의 gradient를 계산해 param.grad에 저장한다.
        # loss가 X_train으로만 계산됐으므로 gradient에는 학습 데이터 정보만 들어간다.
        loss.backward()
        # 저장된 gradient와 lr로 가중치를 한 번 갱신한다.
        # 반드시 backward 다음에 호출한다. 순서가 바뀌면 gradient가 아직 0이라 가중치가 움직이지 않는다.
        # lr이 클수록(candidate 0.01) 이 한 번의 이동 폭이 커진다.
        optimizer.step()

        # 평가 모드로 바꾼다. 방금 갱신된 가중치로 train/valid 손실을 잰다.
        model.eval()
        # 평가에는 gradient가 필요 없으므로 기록을 끈다.
        with torch.no_grad():
            # train_loss (type: float): 갱신 "후" 가중치로 eval 모드에서 다시 잰 train 평균 손실이다.
            # 위의 loss는 갱신 "전" 가중치의 값이라, valid와 같은 시점·같은 모드로 비교하려고 다시 잰다.
            # .item()은 0차원 Tensor를 파이썬 float로 꺼낸다.
            train_loss = loss_fn(model(X_train), y_train).item()
            # valid_loss (type: float): 갱신 후 가중치로 잰 valid 평균 손실이다. best epoch를 고르는 기준이다.
            valid_loss = loss_fn(model(X_valid), y_valid).item()
        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록한다.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가한다.
        # 실제로 잰 valid_loss를 넣는다.
        # 순서 [epoch, train_loss, valid_loss]는 아래 DataFrame의 열 이름 순서와 정확히 맞아야 한다.
        history.append([epoch, train_loss, valid_loss])

    # --- # 설정 하나의 학습이 끝난 뒤: history CSV 저장 → best epoch 찾기 → 요약 추가 ---
    # table (type: DataFrame, 20행 × 3열): history를 표로 바꾼 것이다.
    # train_loss_eval이라는 이름은 "eval 모드로 갱신 후 다시 잰 train loss"라는 뜻을 열 이름에 남긴 것이다.
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
    # history_baseline.csv, history_candidate.csv로 저장한다.
    # index=False: DataFrame의 행 번호(0~19)는 epoch 열과 겹치는 불필요한 정보라 파일에 쓰지 않는다.
    table.to_csv(base / f"history_{name}.csv", index=False)
    # best (type: Series): valid_loss가 가장 작은 행 한 줄이다.
    # idxmin()은 최솟값이 있는 행의 인덱스를 돌려주고, loc[]로 그 행 전체(epoch, 두 loss)를 꺼낸다.
    # 마지막 epoch가 아니라 valid 성적이 가장 좋았던 지점을 고르는 것이 best epoch의 뜻이다.
    best = table.loc[table["valid_loss"].idxmin()]
    # 이 설정의 요약 한 줄 [이름, lr, best epoch, best valid loss]를 summary에 붙인다.
    # int(): DataFrame에 float와 섞여 저장되며 float(예: 11.0)가 된 epoch 번호를 정수로 되돌린다.
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])

    # --- 학습곡선을 그리고 PNG 저장 ---
    # 새 이미지를 만든다. figsize=(8, 5)는 가로 8인치, 세로 5인치이다.
    # 설정마다 새 figure를 만들어 앞 설정의 선이 겹쳐 그려지지 않게 한다.
    plt.figure(figsize=(8, 5))
    # x축 epoch, y축 train loss(eval 모드)인 곡선을 그린다. label은 범례에 표시될 이름이다.
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    # x축 epoch, y축 valid loss인 곡선을 그린다. train 선과 벌어지기 시작하면 과적합 신호로 읽는다.
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # best epoch 위치에 점선(linestyle="--") 세로선을 그어 가장 좋은 지점을 눈으로 바로 찾게 한다.
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    # x축 이름을 "Epoch"로 단다.
    plt.xlabel("Epoch")
    # y축 이름을 "Loss"로 단다.
    plt.ylabel("Loss")
    # 위에서 정한 label들로 범례 상자를 그린다.
    plt.legend()
    # 축 이름·범례가 이미지 밖으로 잘리지 않게 여백을 자동으로 맞춘다.
    plt.tight_layout()
    # curves_baseline.png, curves_candidate.png로 저장한다.
    # dpi=150: 1인치당 150픽셀로, 화면 기본(100)보다 선명하게 저장한다.
    plt.savefig(base / f"curves_{name}.png", dpi=150)
    # 다 그린 figure를 닫아 메모리를 돌려준다. 닫지 않으면 반복마다 figure가 쌓인다.
    plt.close()

# -----------------------------------------------------------------------------
# 두 설정의 비교표 저장과 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame, 2행 × 4열): baseline·candidate의 lr, best epoch, best valid loss를 나란히 둔 비교표이다.
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
# experiment_table.csv로 저장한다. index=False로 행 번호는 쓰지 않는다.
result.to_csv(base / "experiment_table.csv", index=False)
# 비교표를 소수점 6자리로 반올림해 출력한다. to_string(index=False)는 행 번호 없이 표 모양 그대로 문자열로 만든다.
print(result.round(6).to_string(index=False))
# 이 실험의 조건을 한 줄로 밝힌다.
# test는 쓰지 않았고, best epoch는 기록한 20개 중 valid loss가 최소였던 지점일 뿐이며,
# 기본필수 단계에서는 그 시점의 모델을 저장·복원하지 않는다(마지막 epoch 모델이 메모리에 남는다).
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")
