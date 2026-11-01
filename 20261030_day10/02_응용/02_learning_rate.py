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
# 추가 라이브러리
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch.utils.data의 TensorDataset, DataLoader: 데이터를 mini-batch로 공급하는 도구이다.
from torch.utils.data import TensorDataset, DataLoader
# [표준 라이브러리] copy: 객체를 복사하는 모듈이다.
import copy
# [외부 라이브러리] matplotlib: Python 시각화 라이브러리이다.
import matplotlib
# 창을 띄우지 않고 PNG에 저장한다. "Agg"는 화면 없이 이미지 파일만 만드는 backend라 서버·터미널 환경에서도 오류 없이 동작한다. pyplot을 불러오기 전에 정해야 적용된다.
matplotlib.use("Agg") 
# 외부 라이브러리. 목적: loss 곡선 그림 그리기. 기능: plot, axvline, savefig 같은 그리기 함수를 제공한다.
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 출력 경로와 device 정하기
# -----------------------------------------------------------------------------
# out (type: Path) 결과 CSV와 PNG를 저장할 폴더 경로다.
out = base / "outputs"
# outputs 폴더를 만든다. exist_ok=True는 이미 있으면 오류 없이 넘어가라는 뜻이라 여러 번 실행해도 멈추지 않는다.
out.mkdir(exist_ok=True)
# 재현용 기본값은 CPU다. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꾼다.
# device (type: torch.device) 모델과 Tensor를 올릴 장치다. CPU로 고정해 누구의 컴퓨터에서든 같은 결과가 나오게 한다.
device = torch.device("cpu")

# -----------------------------------------------------------------------------
# 데이터 읽기 및 train/valid 분할
# -----------------------------------------------------------------------------
# df (type: DataFrame) x1, x2, label, split 열을 가진 점 데이터 전체를 읽는다.
df = pd.read_csv(base / "data" / "points.csv")
# train (type: DataFrame) split 값이 "train"인 행만 고른다. 가중치를 학습하는 데 쓰는 데이터다.
train = df[df["split"] == "train"]
# valid (type: DataFrame) split 값이 "valid"인 행만 고른다. 학습에 쓰지 않고 두 momentum 설정을 비교하는 데만 쓴다.
valid = df[df["split"] == "valid"]
# X_train (type: Tensor, shape [train 행 수, 2]) 입력 특성 2개다. nn.Linear의 가중치가 float32이므로 같은 float32로 만든다. 
# DataLoader가 CPU에서 섞어야 하므로 device로 옮기지 않고 batch마다 옮긴다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
# y_train (type: Tensor, shape [train 행 수]) 정답 클래스 번호(0 또는 1)다. 
# CrossEntropyLoss는 정답을 정수(long) 클래스 번호로 받으므로 long으로 만든다.
y_train = torch.tensor(train["label"].values, dtype=torch.long)
# X_valid (type: Tensor, shape [valid 행 수, 2]) 비교용 입력 특성이다. 매 epoch 평가에 그대로 쓰므로 미리 device로 옮겨 둔다.
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
# y_valid (type: Tensor, shape [valid 행 수]) 비교용 정답 클래스 번호다. 모델 출력과 같은 장치에 있어야 손실을 계산할 수 있어 device로 옮긴다.
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

# -----------------------------------------------------------------------------
# 손실 함수
# -----------------------------------------------------------------------------
# loss_fn (type: CrossEntropyLoss) 분류용 손실 함수다. 
# 모델 출력(logit 2개)에 내부적으로 softmax를 적용하고 정답 확률의 -log를 평균한다. 그래서 모델 끝에 softmax를 따로 넣지 않는다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# 비교할 설정 후보 정하기
# -----------------------------------------------------------------------------
# AdamW의 lr 후보 0.001, 0.01, 0.1을 세 설정으로 만든다.
# 실행 순서·seed·모델 폭·데이터는 바꾸지 않는다.
# configs (type: list[dict]): 비교할 설정 후보 3개의 목록이다. dict 하나가 실험 한 번이다.
#   - 'name' (type: str): 출력 파일 이름(02_lr0.001_history.csv 등)과 print에 쓰이는 후보 이름이다.
#   - 'lr' (type: float): 이 후보에서 바꾸는 유일한 값인 학습률이다.
#     0.001: AdamW의 기본 lr 값이다. 한 번에 조금씩 움직여 안정적이지만 loss가 천천히 줄어든다.
#     0.01: 0.001의 10배 보폭이다. loss가 더 빨리 줄 수 있다.
#     0.1: 0.001의 100배 보폭이다. 너무 크면 최적점을 지나쳐 loss가 흔들리거나 다시 커진다.
#   - 10배 간격으로 고른 이유: 학습률은 크기의 "자릿수"에 따라 결과가 크게 달라지므로
#     후보를 로그 간격(0.001, 0.01, 0.1)으로 두고 비교하는 것이 기본이다.
#   - 'optimizer' 키를 적지 않은 이유: 아래 config.get("optimizer", "AdamW")의 기본값이 "AdamW"라
#     키가 없으면 자동으로 AdamW가 만들어진다.
#   - width·bn·dropout 키도 적지 않아 세 후보 모두 기본값(32, False, 0.0)이 똑같이 적용된다.
#     그래서 세 후보 사이에 다른 값은 lr 하나뿐이다.
#   - 리스트 순서(0.001 → 0.01 → 0.1)가 곧 실행 순서이다.
configs = [{'name': 'lr0.001', 'lr': 0.001},
           {'name': 'lr0.01', 'lr': 0.01},
           {'name': 'lr0.1', 'lr':0.1}]
# summary (type: list[dict]): 후보마다 best epoch, best valid loss 등 요약 한 줄을 모아 둘 목록이다.
summary = []

# -----------------------------------------------------------------------------
# 후보마다 모델 만들기 → DataLoader 만들기 → Optimizer 만들기
# -----------------------------------------------------------------------------
# config (type: dict): 이번에 실험할 설정 한 개이다. 세 후보를 같은 절차로 차례대로 돌린다.
for config in configs:
    # 전역 난수 시드를 42로 고정한다.
    torch.manual_seed(42)
    # width (type: int): 은닉층 뉴런 수이다. config에 "width"가 없으면 기본값 32를 쓴다.
    width = config.get("width", 32)
    # layers (type: list[nn.Module]): 층을 순서대로 담을 목록이다.
    # 첫 층은 입력 2개(x1, x2)를 width개로 바꾸는 선형층이다.
    layers = [nn.Linear(2, width)]
    # config의 "bn"이 True일 때만 BatchNorm을 넣는다. 이번 세 후보는 "bn"이 없어 False이므로 넣지 않는다.
    if config.get("bn", False):
        # 첫 은닉층 출력을 batch 기준으로 평균 0·분산 1 근처로 정규화하는 층을 붙인다.
        layers.append(nn.BatchNorm1d(width))
    # ReLU(음수는 0으로 바꾸는 활성화) 
    # → Dropout(학습 중 일부 뉴런을 확률적으로 끄는 층) 
    # → 두 번째 선형층(width→width)을 이어 붙인다.
    # 이번에는 "dropout" 키가 없어 p=0.0이므로 Dropout은 아무 뉴런도 끄지 않는다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # 두 번째 은닉층 뒤에도 같은 조건으로 BatchNorm을 넣는다. 이번 세 후보는 넣지 않는다.
    if config.get("bn", False):
        # 두 번째 은닉층 출력을 정규화하는 층을 붙인다.
        layers.append(nn.BatchNorm1d(width))
    # ReLU → Dropout(p=0.0) → 출력층(width→2, class 0과 1의 logit)을 이어 붙인다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # model (type: nn.Sequential): 목록의 층을 순서대로 연결한 신경망을 만들고 device로 옮긴다.
    # *layers는 목록을 풀어서 인자 여러 개로 넘긴다는 뜻이다.
    # 반복마다 새로 만들어 앞 후보의 학습 결과를 물려받지 않는다.
    model = nn.Sequential(*layers).to(device)
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만든다.
    # loader (type: DataLoader): train 데이터를 32개씩 섞어서 꺼내 준다.
    #   - batch_size=32: 가중치를 한 번 갱신할 때 쓰는 데이터 개수이다.
    #   - shuffle=True: epoch마다 데이터 순서를 섞는다.
    #   - generator=torch.Generator().manual_seed(43): seed 43으로 만든 전용 난수 생성기를 후보마다 새로 넘긴다.
    #     그래서 세 후보가 똑같은 batch 순서를 보게 되고, 순서 차이가 결과에 섞이지 않는다.
    #   - num_workers=0: 별도 프로세스 없이 메인 프로세스에서 읽어 재현성을 지킨다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # config의 optimizer가 "SGD"이면 SGD를 만든다. 키가 없으면 "AdamW"로 본다.
    # 이번 세 후보는 "optimizer" 키가 없어 이 분기로 들어가지 않는다.
    if config.get("optimizer", "AdamW") == "SGD":
        # optimizer (type: torch.optim.SGD): 이번 실험에서는 실행되지 않는 분기이다.
        # momentum 키가 없으면 0.0을 쓰고, weight_decay=0.0으로 가중치 감쇠를 끈다.
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    # optimizer가 "SGD"가 아니면 AdamW를 만든다. 이번 세 후보는 모두 이 분기로 들어간다.
    else:
        # optimizer (type: torch.optim.AdamW): 새로 만든 model의 가중치를 갱신할 AdamW이다.
        #   - model.parameters(): 이 반복에서 새로 만든 model의 가중치를 넘긴다. 그래야 이 모델이 학습된다.
        #   - lr=config["lr"]: 이번 후보의 학습률(0.001, 0.01, 0.1 중 하나)을 넣는다.
        #     후보마다 달라지는 유일한 값이다.
        #   - weight_decay=0.0: AdamW의 기본값은 0.01이라 0.0을 적어 감쇠를 끈다.
        #     감쇠가 켜져 있으면 lr과 감쇠 효과가 섞여, 차이가 lr 때문인지 가려낼 수 없다.
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # --- 20 epoch 학습과 epoch 말 평가 ---
    # history (type: list[list]): epoch마다 [epoch, train_loss, valid_loss, valid_acc]를 한 줄씩 기록한다.
    history = []
    for epoch in range(1, 20 + 1):
        # 학습 모드로 바꾼다. Dropout·BatchNorm이 학습용으로 동작하게 하는 스위치이다.
        # 아래에서 eval()로 바꾸므로 매 epoch 학습 전에 다시 학습 모드로 돌려놓는다.
        model.train()
        # xb (type: torch.Tensor, shape: (최대 32, 2)), yb (type: torch.Tensor, shape: (최대 32,)):
        # 섞인 mini-batch를 하나씩 꺼낸다. batch 하나마다 가중치를 한 번 갱신한다.
        for xb, yb in loader:
            # batch를 모델과 같은 장치로 옮긴다.
            xb, yb = xb.to(device), yb.to(device)
            # 지난 batch에서 쌓인 gradient를 0으로 지운다.
            # PyTorch는 gradient를 더해 쌓으므로 지우지 않으면 이전 값이 섞인다.
            optimizer.zero_grad()
            # loss (type: torch.Tensor, 스칼라): 이 batch의 순전파 결과(logit)와 정답으로 계산한 평균 손실이다.
            loss = loss_fn(model(xb), yb)
            # 역전파로 손실에 대한 각 가중치의 gradient를 계산해 param.grad에 저장한다.
            loss.backward()
            # 저장된 gradient로 가중치를 갱신한다.
            # AdamW는 gradient의 이동 평균과 제곱 이동 평균으로 방향과 크기를 다듬은 뒤 lr을 곱해 움직이므로,
            # lr이 클수록 이 한 번의 이동 폭이 커진다.
            optimizer.step()
        # 같은 epoch 말 모델을 eval 모드로 비교한다. 학습 중 Dropout이 켜진 상태의 loss와 구분한다.
        # 평가 모드로 바꾼다. Dropout을 끄고 BatchNorm은 저장된 통계를 쓰게 해, 같은 입력에 같은 출력이 나오게 한다.
        model.eval()
        # 평가에는 gradient가 필요 없으므로 기록을 끈다.
        # 메모리와 계산을 아끼고, valid 데이터가 가중치 갱신에 섞이지 않게 한다.
        with torch.no_grad():
            # train_loss (type: float): epoch 말 모델로 train 전체를 eval 모드에서 다시 계산한 손실이다.
            # 학습 중 batch 손실 평균과 달리 valid와 같은 조건에서 잰 값이라 둘을 공정하게 비교할 수 있다.
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # logits (type: torch.Tensor, shape: (valid 건수, 2)): valid 입력에 대한 class별 점수(logit)이다.
            logits = model(X_valid)
            # valid_loss (type: float): valid 평균 손실이다. 값이 작을수록 정답 class에 높은 확률을 준 것이다.
            valid_loss = loss_fn(logits, y_valid).item()
            # valid_acc (type: float): valid 정확도이다.
            # argmax(1)로 행마다 점수가 큰 class를 예측으로 고르고, 정답과 같은지(True/False)를 1.0/0.0으로 바꿔 평균한다.
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        # 이번 epoch의 결과 한 줄을 기록에 추가한다.
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # --- 후보별 기록 저장과 best epoch 찾기 ---
    # table (type: DataFrame): 20 epoch 기록을 열 이름이 붙은 표로 만든다.
    # "train_loss_eval"은 eval 모드로 잰 train 손실이라는 뜻을 이름에 담은 것이다.
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    # best (type: Series): valid_loss가 가장 작은 행이다.
    # idxmin()은 최솟값의 행 번호를 돌려주고(같은 값이 여럿이면 가장 앞 epoch), loc로 그 행을 꺼낸다.
    # train이 아닌 valid 기준으로 골라야 과적합 전 가장 일반화가 잘 된 시점을 찾는다.
    best = table.loc[table["valid_loss"].idxmin()]
    # name (type: str): 후보 이름("lr0.001", "lr0.01", "lr0.1" 중 하나)이다. 파일 이름과 출력에 쓴다.
    name = config["name"]
    # 후보별 epoch 기록을 CSV로 저장한다. 예: outputs/02_lr0.01_history.csv
    # index=False는 0, 1, 2… 행 번호 열을 파일에 쓰지 않는다는 뜻이다.
    table.to_csv(out / f"02_{name}_history.csv", index=False)
    # 설정값과 결과 요약을 한 dict로 합쳐 summary에 추가한다.
    #   - **config: name·lr을 풀어 넣어, 비교표에서 어떤 lr의 결과인지 바로 보이게 한다.
    #   - best 행의 값은 numpy 숫자라 int/float로 바꿔 파이썬 숫자로 저장한다.
    #   - last_* 값은 반복문이 끝난 뒤 남은 20 epoch째 값이다.
    #     best와 last를 함께 두면 best 이후 loss가 다시 커졌는지(불안정·과적합)를 판단할 수 있다.
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})

    # --- 후보별 loss 곡선 저장 ---
    # # 가로 8인치·세로 5인치짜리 새 그림을 연다. 후보마다 새로 열어 그래프가 겹치지 않게 한다.
    plt.figure(figsize=(8, 5))
    # epoch별 train 손실(eval 모드) 곡선을 그린다.
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    # epoch별 valid 손실 곡선을 그린다. 
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    # best epoch 위치에 세로 점선을 긋는다.
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    # x축 이름을 붙인다.
    plt.xlabel("Epoch")
    # y축 이름을 붙인다.
    plt.ylabel("Loss")
    # 각 곡선의 label을 범례로 표시한다.
    plt.legend()
    # 축 이름과 범례가 잘리지 않게 여백을 자동 조정한다.
    plt.tight_layout()
    # 이미지릃 PNG 파일로 저장한다. dpi=150은 인치당 점 수로, 화면용보다 선명하게 저장한다.
    plt.savefig(out / f"02_{name}_curves.png", dpi=150)
    # 이미지를 닫아 메모리를 비운다. 닫지 않으면 후보가 많을 때 이미지가 계속 쌓인다.
    plt.close()
    # 후보 이름, best epoch, best valid loss(소수점 6자리 반올림)를 출력한다.
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# -----------------------------------------------------------------------------
# 비교표 저장
# -----------------------------------------------------------------------------
# result (type: DataFrame) 후보별 요약 dict 목록을 표로 만든다. 한 행이 한 후보다.
result = pd.DataFrame(summary)
# 세 lr 설정의 비교표를 CSV로 저장한다.
# best_epoch가 작을수록 "빠른 감소", best와 last valid loss의 차이가 작을수록 "안정성"이 좋다고 따로 판단할 수 있다.
result.to_csv(out / "02_comparison.csv", index=False)
# 생성된 파일 종류와, 설정 선택 단계라 test를 쓰지 않았음을 출력한다.
print("생성: outputs/02_*.csv, *.png / test 미사용")
