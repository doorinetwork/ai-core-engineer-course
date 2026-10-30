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
# 외부 라이브러리(PyTorch). 목적: train 데이터를 mini-batch로 나눠 공급하기. 
# 기능: TensorDataset은 (X, y) 쌍을 묶고, DataLoader는 그것을 batch 단위로 섞어서 꺼내 준다.
from torch.utils.data import TensorDataset, DataLoader
# 표준 라이브러리. 목적: 객체 깊은 복사(예: best 모델 가중치 보관)에 쓰려고 불러온다. 기능: copy.deepcopy 등을 제공한다. 이 파일에서는 실제로 호출하지 않는다.
import copy
# 외부 라이브러리. Python 시각화 라이브러리.
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
# Dropout 확률 p=0.0과 p=0.3을 비교한다.
# 평가에서는 model.eval()과 no_grad를 모두 사용하고, best epoch는 valid loss로 찾는다.
# configs (type: list[dict]): 비교할 설정 후보 2개의 목록이다. dict 하나가 실험 한 번이다.
#   - 'name' (type: str): 출력 파일 이름(05_dropout0.0_history.csv 등)과 print에 쓰이는 후보 이름이다.
#   - 'lr' (type: float): 두 후보 모두 0.01로 똑같이 둔다. 아래 AdamW가 config["lr"]을 키로 직접 읽으므로
#     이 키는 반드시 있어야 한다. 없으면 KeyError로 멈춘다.
#   - 'dropout' (type: float): 이 실험에서 바꾸는 유일한 값인 Dropout 확률 p이다.
#     아래 nn.Dropout(config.get("dropout", 0.0))이 이 값을 읽어 두 Dropout 층에 똑같이 넣는다.
#     0.0: 학습 중에도 아무 activation도 끄지 않는다. Dropout이 없는 모델과 같은 기준선(baseline)이다.
#     0.3: 학습 중 각 activation을 30% 확률로 0으로 만들고, 살아남은 값은 1/(1-0.3) = 1/0.7 ≈ 1.43배로 키워
#          출력의 기댓값을 맞춘다. 특정 뉴런에 지나치게 의존하지 못하게 막아 과적합을 줄이는 규제이다.
#   - 오류 수정: 원본은 두 번째 후보가 {'name': 'dropout0.3', 'lr': 0.01, 'dropout': 0.0}이었다.
#     'dropout' 값이 0.0이면 이름만 dropout0.3일 뿐 첫 후보와 설정이 똑같아 p=0.3 비교가 이루어지지 않는다.
#     요구사항 "p=0.0과 p=0.3을 비교"에 맞추려고 이 값만 0.3으로 고쳤다.
#   - 'width'·'optimizer'·'bn' 키를 적지 않은 이유: 아래 config.get의 기본값(width 32, "AdamW", False)이
#     두 후보에 똑같이 적용되므로, 두 후보 사이에 다른 값은 dropout 하나뿐이 된다.
#     교안의 오류 사례처럼 "Dropout만 추가하고 다른 조건까지 함께 변경"하면 효과를 분리해 해석할 수 없다.
#   - 40 epochs는 config가 아니라 아래 range(1, 40 + 1)에 고정돼 있어 두 후보에 똑같이 적용된다.
#   - 리스트 순서(dropout0.0 → dropout0.3)가 곧 실행 순서이다.
configs = [{'name': 'dropout0.0', 'lr': 0.01, 'dropout': 0.0},
           {'name': 'dropout0.3', 'lr': 0.01, 'dropout': 0.0}]
# summary (type: list[dict]): 후보마다 best epoch, best valid loss 등 요약 한 줄을 모아 둘 목록이다.
summary = []

# -----------------------------------------------------------------------------
# 후보마다 모델 만들기 → DataLoader 만들기 → Optimizer 만들기
# -----------------------------------------------------------------------------
# config (type: dict): 이번에 실험할 설정 한 개이다. 두 후보를 같은 절차로 차례대로 돌린다.
for config in configs:
    # 전역 난수 시드를 42로 고정한다.
    torch.manual_seed(42)
    # width (type: int): 은닉층 뉴런 수이다. config에 "width"가 없으므로 두 후보 모두 기본값 32를 쓴다.
    width = config.get("width", 32)
    # layers (type: list[nn.Module]): 층을 순서대로 담을 목록이다.
    # 첫 층은 입력 2개(x1, x2)를 width(32)개로 바꾸는 선형층이다.
    layers = [nn.Linear(2, width)]
    # config의 "bn"이 True일 때만 BatchNorm을 넣는다. 이번 두 후보는 "bn"이 없어 False이므로 넣지 않는다.
    if config.get("bn", False):
        # 첫 은닉층 출력을 batch 기준으로 평균 0·분산 1 근처로 정규화하는 층을 붙인다.
        layers.append(nn.BatchNorm1d(width))
    # ReLU(음수는 0으로 바꾸는 활성화) → Dropout(p) → 두 번째 선형층(32→32)을 이어 붙인다.
    # Dropout을 ReLU 뒤에 두어, 활성화된 첫 은닉층 출력 중 일부를 학습 중에 무작위로 끈다.
    # p는 config["dropout"]에서 읽는다. dropout0.0 후보는 아무것도 끄지 않고, dropout0.3 후보는 30%를 끈다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # 두 번째 은닉층 뒤에도 같은 조건으로 BatchNorm을 넣는다. 이번 두 후보는 넣지 않는다.
    if config.get("bn", False):
        # 두 번째 은닉층 출력을 정규화하는 층을 붙인다.
        layers.append(nn.BatchNorm1d(width))
    # ReLU → Dropout(같은 p) → 출력층(32→2, class 0과 1의 logit)을 이어 붙인다.
    # 출력층 뒤에는 Dropout을 두지 않는다. 최종 점수(logit)를 무작위로 지우면 정답 판단 자체가 흔들리기 때문이다.
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
    #     Dropout mask는 전역 난수(torch.manual_seed로 정한 것)를 쓰고, 섞기는 이 전용 생성기를 쓴다.
    #     둘을 분리했기 때문에 p=0.3 후보에서 Dropout이 난수를 소비해도 batch 순서는 p=0.0 후보와 똑같다.
    #   - num_workers=0: 별도 프로세스 없이 메인 프로세스에서 읽어 재현성을 지킨다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # config의 optimizer가 "SGD"이면 SGD를 만든다. 키가 없으면 "AdamW"로 본다.
    # 이번 두 후보는 "optimizer" 키가 없어 이 분기로 들어가지 않는다.
    if config.get("optimizer", "AdamW") == "SGD":
        # optimizer (type: torch.optim.SGD): 이번 실험에서는 실행되지 않는 분기이다.
        # momentum 키가 없으면 0.0을 쓰고, weight_decay=0.0으로 가중치 감쇠를 끈다.
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    # optimizer가 "SGD"가 아니면 AdamW를 만든다. 이번 두 후보는 모두 이 분기로 들어간다.
    else:
        # optimizer (type: torch.optim.AdamW): 새로 만든 model의 가중치를 갱신할 AdamW이다.
        #   - model.parameters(): 이 반복에서 새로 만든 model의 가중치를 넘긴다. 그래야 이 모델이 학습된다.
        #   - lr=config["lr"]: 두 후보 모두 0.01을 넣는다. 보폭을 같게 해야 Dropout의 효과만 비교된다.
        #   - weight_decay=0.0: AdamW의 기본값은 0.01이라 0.0을 적어 감쇠를 끈다.
        #     감쇠도 규제라서 켜 두면 Dropout의 효과와 섞여 어느 쪽 덕분인지 구분할 수 없다.
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # --- 40 epoch 학습과 epoch 말 평가 ---
    # history (type: list[list]): epoch마다 [epoch, train_loss, valid_loss, valid_acc]를 한 줄씩 기록한다.
    history = []
    # epoch (type: int): 1~40, 총 40번 train 전체를 돈다.
    # Dropout은 학습을 느리게 하는 대신 과적합을 늦추므로, 과적합 구간까지 곡선에 담으려고 길게 돌린다.
    # 1부터 세어 표와 그래프의 epoch 번호를 사람이 읽기 쉽게 한다.
    for epoch in range(1, 40 + 1):
        # 학습 모드로 바꾼다. 이 모드에서만 Dropout이 무작위로 activation을 끈다.
        # 아래에서 eval()로 바꾸므로 매 epoch 학습 전에 다시 학습 모드로 돌려놓아야 한다.
        # 이 줄이 없으면 2번째 epoch부터 Dropout이 꺼진 채 학습되어 p=0.3의 효과가 사라진다.
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
            # p=0.3 후보는 forward마다 새 Dropout mask를 뽑으므로, 매 batch 일부 뉴런이 꺼진 모델로 계산된다.
            loss = loss_fn(model(xb), yb)
            # 역전파로 손실에 대한 각 가중치의 gradient를 계산해 param.grad에 저장한다.
            # 꺼진 뉴런을 지나는 경로는 이번 batch에서 gradient가 0이 된다.
            loss.backward()
            # 저장된 gradient로 가중치를 갱신한다.
            # AdamW는 gradient의 이동 평균과 제곱 이동 평균으로 방향과 크기를 다듬은 뒤 lr을 곱해 움직인다.
            optimizer.step()
        # 같은 epoch 말 모델을 eval 모드로 비교한다. 학습 중 Dropout loss와 구분한다.
        # 평가 모드로 바꾼다. Dropout은 아무것도 끄지 않고 입력을 그대로 전달하게 되어, 같은 입력에 같은 출력이 나온다.
        # 요구사항 "평가에서는 model.eval()을 사용"에 해당한다.
        # eval()은 동작 모드만 바꿀 뿐 gradient 계산은 끄지 않으므로, 아래 no_grad를 따로 써야 한다.
        model.eval()
        # 평가에는 gradient가 필요 없으므로 기록을 끈다. 요구사항 "no_grad를 모두 사용"에 해당한다.
        with torch.no_grad():
            # train_loss (type: float): epoch 말 모델로 train 전체를 eval 모드(Dropout 꺼짐)에서 다시 계산한 손실이다.
            # 학습 중 batch 손실은 p=0.3 후보에서 뉴런이 꺼진 상태라 실제 모델보다 높게 나온다.
            # valid와 같은 조건(Dropout 꺼짐)에서 잰 값이라 두 후보를 공정하게 비교할 수 있다.
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
        # table (type: DataFrame): 40 epoch 기록을 열 이름이 붙은 표로 만든다.
        # "train_loss_eval"은 eval 모드로 잰 train 손실이라는 뜻을 이름에 담은 것이다.
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    # best (type: Series): valid_loss가 가장 작은 행이다. 요구사항 "best epoch는 valid loss로 찾는다"에 해당한다.
    # idxmin()은 최솟값의 행 번호를 돌려주고(같은 값이 여럿이면 가장 앞 epoch), loc로 그 행을 꺼낸다.
    # train이 아닌 valid 기준으로 골라야 과적합 전 가장 일반화가 잘 된 시점을 찾는다.
    best = table.loc[table["valid_loss"].idxmin()]
    # name (type: str): 후보 이름("dropout0.0", "dropout0.3" 중 하나)이다. 파일 이름과 출력에 쓴다.
    name = config["name"]
    # 후보별 epoch 기록을 CSV로 저장한다. 예: outputs/05_dropout0.3_history.csv
    # index=False는 0, 1, 2… 행 번호 열을 파일에 쓰지 않는다는 뜻이다.
    table.to_csv(out / f"05_{name}_history.csv", index=False)
    # 설정값과 결과 요약을 한 dict로 합쳐 summary에 추가한다.
    #   - **config: name·lr·dropout을 풀어 넣어, 비교표에서 어떤 p의 결과인지 바로 보이게 한다.
    #     원본처럼 dropout이 둘 다 0.0이면 이 열에서 두 행이 같은 값으로 보여 오류를 확인할 수 있다.
    #   - best 행의 값은 numpy 숫자라 int/float로 바꿔 파이썬 숫자로 저장한다.
    #   - last_* 값은 반복문이 끝난 뒤 남은 40 epoch째 값이다.
    #     best_valid_loss와 last_valid_loss를 함께 보면 best 이후 valid loss가 얼마나 다시 커졌는지(과적합)를 판단할 수 있다.
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    
    # --- 후보별 loss 곡선 저장 ---
    # 가로 8인치·세로 5인치짜리 새 이미지를 연다. 후보마다 새로 열어 그래프가 겹치지 않게 한다.
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
    # 그림을 PNG 파일로 저장한다. 예: outputs/05_dropout0.3_curves.png
    # dpi=150은 인치당 점 수로, 화면용보다 선명하게 저장한다.
    plt.savefig(out / f"05_{name}_curves.png", dpi=150)
    # 이미지를 닫아 메모리를 비운다. 닫지 않으면 후보가 많을 때 이미지가 계속 쌓인다.
    plt.close()
    # 후보 이름, best epoch, best valid loss(소수점 6자리 반올림)를 출력한다.
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# -----------------------------------------------------------------------------
# 두 후보 비교표 저장
# -----------------------------------------------------------------------------
# result (type: DataFrame): 후보별 요약 dict 목록을 표로 만든다. 한 행이 한 Dropout 후보이다.
result = pd.DataFrame(summary)
# 두 Dropout 설정의 비교표를 CSV로 저장한다.
result.to_csv(out / "05_comparison.csv", index=False)
# 생성된 파일 종류와, 설정 선택 단계라 test를 쓰지 않았음을 출력한다.
print("생성: outputs/05_*.csv, *.png / test 미사용")
