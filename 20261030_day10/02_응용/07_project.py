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
# device = torch.device("cpu")
# GPU로 device를 변경해본다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
# configs (type: list[dict]): 비교할 설정 후보 3개의 목록이다. dict 하나가 실험 한 번이다.
#   - sgd: optimizer를 "SGD"로 지정한 후보이다. momentum 키가 없어 0.0, dropout 키가 없어 0.0이 적용된다.
#   - adamw: optimizer 키가 없어 기본값 "AdamW"가 적용된다. sgd와 다른 값은 optimizer 하나뿐이다.
#   - adamw_dropout: adamw에 dropout 0.3만 더한 후보이다. adamw와 다른 값은 dropout 하나뿐이다.
#   세 후보 모두 lr=0.01, width 32(기본값), bn 없음(기본값 False)으로 같다.
#   그래서 sgd↔adamw를 비교하면 Optimizer의 효과를, adamw↔adamw_dropout을 비교하면 Dropout의 효과를 단정할 수 있다.
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01}, 
           {'name': 'adamw', 'lr': 0.01}, 
           {'name': 'adamw_dropout', 'lr': 0.01, 'dropout': 0.3}]
# summary (type: list[dict]): 후보마다 best epoch, best valid loss, 복원 확인 결과 등 요약 한 줄을 모아 둘 목록이다.
summary = []

# -----------------------------------------------------------------------------
# 후보마다 모델 만들기 → DataLoader 만들기 → Optimizer 만들기
# -----------------------------------------------------------------------------
# config (type: dict): 이번에 실험할 설정 한 개이다. 두 후보를 같은 절차로 차례대로 돌린다.
for config in configs:
    # 전역 난수 시드를 42로 고정한다.
    torch.manual_seed(42)
    # width (type: int): 은닉층 뉴런 수이다. 세 후보 모두 "width" 키가 없으므로 기본값 32를 쓴다.
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
    # p는 config["dropout"]에서 읽는다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    # 두 번째 은닉층 뒤에도 같은 조건으로 BatchNorm을 넣는다. 이번 실험에서는 세 후보 모두 건너뛴다.
    if config.get("bn", False):
        # 두 번째 은닉층 출력(32개 특성)을 정규화하는 층을 붙인다.
        layers.append(nn.BatchNorm1d(width))
    # ReLU → Dropout → 출력층(32→2, class 0과 1의 logit)을 이어 붙인다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    # model (type: nn.Sequential): 목록의 층을 순서대로 연결한 신경망을 만들고 device로 옮긴다.
    # *layers는 목록을 풀어서 인자 여러 개로 넘긴다는 뜻이다.
    # 세 후보 모두 Linear·ReLU·Dropout·Linear·ReLU·Dropout·Linear 7층이다.
    # 반복마다 새로 만들어 앞 후보의 학습 결과를 물려받지 않는다.
    model = nn.Sequential(*layers).to(device)
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만든다.
    # loader (type: DataLoader): train 데이터를 32개씩 섞어서 꺼내 준다.
    #   - batch_size=32: 가중치를 한 번 갱신할 때 쓰는 데이터 개수이다.
    #     drop_last 기본값이 False라 240건은 32건씩 7번과 마지막 16건 1번, 모두 8번의 batch로 나뉜다.
    #   - shuffle=True: epoch마다 데이터 순서를 섞는다.
    #   - generator=torch.Generator().manual_seed(43): seed 43으로 만든 전용 난수 생성기를 후보마다 새로 넘긴다.
    #     그래서 세 후보가 epoch마다 똑같은 순서의 batch를 받는다.
    #     전역 seed(42)와 따로 두었기 때문에, Dropout이 전역 난수를 써도 batch 순서는 바뀌지 않는다.
    #   - num_workers=0: 별도 프로세스 없이 메인 프로세스에서 읽어 재현성을 지킨다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # config의 optimizer가 "SGD"이면 SGD를 만든다. 키가 없으면 "AdamW"로 본다.
    # 세 후보 중 sgd만 이 분기로 들어간다.
    if config.get("optimizer", "AdamW") == "SGD":
        # optimizer (type: torch.optim.SGD): gradient 방향으로 lr만큼 그대로 움직이는 기본 경사하강법이다.
        #   - lr=config["lr"]: 0.01을 넣는다. AdamW 후보와 같은 값이라 Optimizer 종류만 다르게 된다.
        #   - momentum=config.get("momentum", 0.0): momentum 키가 없어 0.0이다. 이전 이동 방향을 이어받지 않는다.
        #   - weight_decay=0.0: 가중치 감쇠를 끈다. 규제 효과가 섞이지 않게 한다.
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    # optimizer가 "SGD"가 아니면 AdamW를 만든다. adamw와 adamw_dropout이 이 분기로 들어간다.
    else:
        # optimizer (type: torch.optim.AdamW): gradient의 이동 평균과 제곱 이동 평균으로 보폭을 다듬는 Optimizer이다.
        #   - lr=config["lr"]: 0.01을 넣는다.
        #   - weight_decay=0.0: AdamW의 기본값은 0.01이라 0.0을 적어 감쇠를 끈다.
        #     감쇠도 규제라서 켜 두면 Dropout의 효과와 섞여 어느 쪽 덕분인지 구분할 수 없다.
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # --- best 기록용 변수 준비 ---
    # history (type: list[list]): epoch마다 [epoch, train_loss, valid_loss, valid_acc]를 한 줄씩 기록한다.
    history = []
    # best_loss (type: float): 지금까지 관측한 가장 작은 valid loss이다.
    # 양의 무한대(inf)로 시작해야 첫 epoch의 valid loss가 무엇이든 반드시 더 작아 첫 best로 저장된다.
    best_loss = float("inf")
    # best_state (type: None → dict[str, torch.Tensor]): best epoch의 가중치 복사본이다. 처음에는 아직 없어 None이다.
    best_state = None
    # best_epoch (type: int): best_loss가 나온 epoch 번호이다. 아직 관측 전이라 0으로 둔다.
    best_epoch = 0

    # --- 40 epoch 학습 ---
    for epoch in range(1, 40 + 1):
        # 학습 모드로 바꾼다. 이 모드에서 Dropout(p>0)은 값을 무작위로 끈다.
        # 아래 A08에서 eval()로 바꾸므로 매 epoch 학습 전에 다시 학습 모드로 돌려놓아야 한다.
        # 이 줄이 없으면 2번째 epoch부터 adamw_dropout의 Dropout이 꺼진 채 학습해 Dropout 비교가 무의미해진다.
        model.train()
        # xb (type: torch.Tensor, shape: (32 또는 마지막 16, 2)), yb (type: torch.Tensor, shape: (32 또는 16,)):
        # 섞인 mini-batch를 하나씩 꺼낸다. epoch마다 8번 꺼내고, batch 하나마다 가중치를 한 번 갱신한다.
        for xb, yb in loader:
            # batch를 모델과 같은 장치로 옮긴다.
            xb, yb = xb.to(device), yb.to(device)
            # 지난 batch에서 쌓인 gradient를 0으로 지운다.
            # PyTorch는 gradient를 더해 쌓으므로 지우지 않으면 이전 값이 섞인다.
            optimizer.zero_grad()
            # loss (type: torch.Tensor, 스칼라): 이 batch의 순전파 결과(logit)와 정답으로 계산한 평균 손실이다.
            loss = loss_fn(model(xb), yb)
            # loss.backward() 다음 optimizer.step()을 호출한다.
            # 같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분한다.
            # 역전파로 손실에 대한 각 가중치의 gradient를 계산해 param.grad에 저장한다.
            # 이 호출이 먼저 있어야 아래 step()이 쓸 gradient가 생긴다.
            loss.backward()
            # 저장된 gradient로 가중치를 갱신한다.
            # 같은 코드라도 위에서 만든 optimizer 종류에 따라 갱신 방식이 다르다.
            # sgd는 "가중치 -= 0.01 × gradient"로 그대로 움직이고,
            # adamw는 gradient의 이동 평균과 제곱 이동 평균으로 방향과 크기를 다듬은 뒤 0.01을 곱해 움직인다.
            # 순서를 바꿔 step()을 먼저 부르면 아직 계산되지 않은(0으로 지워진) gradient로 갱신하게 된다.
            optimizer.step()

        # --- epoch 말 평가 ---
        # 같은 epoch 말 모델을 eval 모드로 비교한다. 학습 중 Dropout loss와 구분한다.
        # epoch 종료 후 model.eval()을 호출한다.
        # 그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않는다.
        # 평가 모드로 바꾼다. 이 모드에서 Dropout은 아무 값도 끄지 않고 입력을 그대로 통과시킨다.
        # 그래서 같은 입력에는 항상 같은 출력이 나오고, 세 후보를 같은 조건(Dropout 꺼짐)에서 비교할 수 있다.
        # 이 줄이 없으면 adamw_dropout의 평가 loss에 무작위로 30%가 꺼진 출력이 섞여 실제 실력보다 나쁘게 잰다.
        model.eval()
        # 평가에는 gradient가 필요 없으므로 기록을 끈다.
        with torch.no_grad():
            # train_loss (type: float): epoch 말 모델로 train 전체를 eval 모드에서 다시 계산한 손실이다.
            # 학습 중 batch 손실은 batch마다 가중치가 다르고 Dropout이 섞여 있으므로,
            # valid와 같은 조건(eval 모드, 같은 가중치)에서 다시 재야 두 값을 공정하게 비교할 수 있다.
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            # logits (type: torch.Tensor, shape: (valid 건수, 2)): valid 입력에 대한 class별 점수(logit)이다.
            logits = model(X_valid)
            # valid_loss (type: float): valid 평균 손실이다. 값이 작을수록 정답 class에 높은 확률을 준 것이다.
            valid_loss = loss_fn(logits, y_valid).item()
            # valid_acc (type: float): valid 정확도이다.
            # argmax(1)로 행마다 점수가 큰 class를 예측으로 고르고, 정답과 같은지(True/False)를 1.0/0.0으로 바꿔 평균한다.
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
            # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관한다.
            #   model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 된다.
            #   이번 valid loss가 지금까지의 최소값보다 "엄격하게" 작을 때만 best를 바꾼다.
            #   <=가 아니라 <를 쓰는 이유: 같은 값(동률)이 나중에 나오면 조건이 거짓이 되어 먼저 발견한 epoch가 남는다.
            #   아래 table의 idxmin()도 같은 값이면 가장 앞 행을 고르므로, 두 방식의 best epoch가 일치한다.
            if valid_loss < best_loss:
                # best_loss (type: float): 새 최소 valid loss로 바꾼다.
                best_loss = valid_loss
                # best_epoch (type: int): 그 값이 나온 epoch 번호를 함께 기록한다.
                best_epoch = epoch
                # best_state (type: dict[str, torch.Tensor]): 지금 가중치의 독립된 복사본이다.
                # model.state_dict()는 모델 안의 가중치 Tensor를 "가리키는" dict라서,
                # 그대로 대입하면 다음 optimizer.step()이 그 Tensor 값을 직접 바꿔 best_state도 함께 바뀐다.
                # 결국 마지막 epoch 가중치가 남는다. copy.deepcopy로 새 메모리에 값을 복사해야 이 시점의 값이 고정된다.
                best_state = copy.deepcopy(model.state_dict())
        # 이번 epoch의 결과 한 줄을 기록에 추가한다.
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # --- 후보별 기록 저장과 best epoch 찾기 ---
    # table (type: DataFrame): 40 epoch 기록을 열 이름이 붙은 표로 만든다.
    # "train_loss_eval"은 eval 모드로 잰 train 손실이라는 뜻을 이름에 담은 것이다.
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    # best (type: Series): valid_loss가 가장 작은 행이다.
    # idxmin()은 최솟값의 행 번호를 돌려주고(같은 값이 여럿이면 가장 앞 epoch), loc로 그 행을 꺼낸다.
    # train이 아닌 valid 기준으로 골라야 과적합 전 가장 일반화가 잘 된 시점을 찾는다.
    best = table.loc[table["valid_loss"].idxmin()]
    # name (type: str): 후보 이름("sgd", "adamw", "adamw_dropout" 중 하나)이다. 파일 이름과 출력에 쓴다.
    name = config["name"]
    # 후보별 epoch 기록을 CSV로 저장한다. 예: outputs/07_adamw_history.csv
    # index=False는 0, 1, 2… 행 번호 열을 파일에 쓰지 않는다는 뜻이다.
    table.to_csv(out / f"07_{name}_history.csv", index=False)
    # 설정값과 결과 요약을 한 dict로 합쳐 summary에 추가한다.
    #   - **config: name·optimizer·lr·dropout을 풀어 넣는다.
    #     후보마다 가진 키가 달라, 비교표에서 없는 칸(adamw의 optimizer, sgd·adamw의 dropout)은 빈칸(NaN)으로 저장된다.
    #   - best 행의 값은 numpy 숫자라 int/float로 바꿔 파이썬 숫자로 저장한다.
    #   - last_* 값은 반복문이 끝난 뒤 남은 40 epoch째 값이다.
    #     best_valid_loss와 last_valid_loss를 함께 보면 best 이후 valid loss가 얼마나 다시 커졌는지(과적합)를 판단할 수 있다.
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})

    # --- best 모델 복원과 재평가 ---
    # 제공 코드: 학습이 끝난 뒤 최적 상태로 복원하고 그 모델로 다시 평가한다.
    # 지금 model에는 40 epoch째 가중치가 들어 있다. A09에서 보관한 best_state의 값을 모델에 다시 써 넣는다.
    # 마지막 모델이 아니라 best 모델을 쓰는 이유: 40 epoch째는 과적합으로 valid 성능이 떨어졌을 수 있기 때문이다.
    model.load_state_dict(best_state)
    # 평가 모드로 바꾼다. 마지막 epoch에서 이미 eval()이었지만, 예측 전에 Dropout이 꺼져 있음을 확실히 한다.
    model.eval()
    # 예측에는 gradient가 필요 없으므로 기록을 끈다.
    with torch.no_grad():
        # best_pred (type: torch.Tensor, shape: (valid 건수,), dtype: int64): best 모델의 valid 예측 class이다.
        # argmax(1)로 행마다 점수가 큰 class 번호를 고르고, .cpu()로 CPU에 옮겨 리스트·CSV로 바꿀 수 있게 한다.
        best_pred = model(X_valid).argmax(1).cpu()
        # restored_loss (type: float): 복원한 모델로 다시 잰 valid loss이다.
        restored_loss = loss_fn(model(X_valid), y_valid).item()
    # labels (type: torch.Tensor, shape: (valid 건수,), dtype: int64): valid 실제 정답을 CPU로 옮긴 것이다.
    # best_pred와 같은 장치(CPU)에 있어야 서로 비교할 수 있다.
    labels = y_valid.cpu()

    # --- 클래스별 support·correct·Recall 계산 ---
    # class_rows (type: list[dict]): class 0과 1의 지표를 한 행씩 담을 목록이다.
    class_rows = []
    # label (type: int): 이번에 계산할 실제 정답 class 번호(0, 1)이다.
    for label in (0, 1):
        # 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
        # 전체 accuracy나 '예측 label의 수'로 나누지 않는다. 0건이면 recall=None.

        # mask (type: torch.Tensor, shape: (valid 건수,), dtype: bool): 실제 정답이 label인 행이면 True이다.
        # Recall은 "실제로 이 class인 것 중 몇 개를 찾아냈나"이므로, 예측이 아니라 실제 정답으로 먼저 고른다.
        mask = labels == label
        # support (type: int): 실제 정답이 label인 valid 행의 개수이다.
        # True를 1, False를 0으로 더하고, .item()으로 파이썬 숫자로 꺼낸 뒤 int로 바꾼다.
        support = int(mask.sum().item())
        # correct (type: int): 그 행들 중 예측도 label인 개수, 곧 맞힌 개수이다.
        # best_pred[mask]로 실제 정답이 label인 행의 예측만 남긴 뒤, label과 같은 것을 센다.
        correct = int((best_pred[mask] == label).sum().item())
        # recall (type: float 또는 None): 맞힌 개수 ÷ 실제 개수이다.
        # support가 0이면 0으로 나눌 수 없고 "찾아야 할 대상이 없음"을 0점으로 기록하면 틀린 정보가 되므로 None을 넣는다.
        # None은 CSV에 빈칸으로 저장된다.
        recall = correct / support if support > 0 else None
        # 이 class의 결과를 한 행(dict)으로 추가한다. 키가 CSV의 열 이름이 된다.
        class_rows.append({"label": label, "support": support, "correct": correct, "recall": recall})

    # 클래스별 지표를 CSV로 저장한다. 예: outputs/07_adamw_class_metrics.csv (2행: class 0, class 1)
    # 전체 accuracy는 한쪽 class가 많으면 다른 class를 놓쳐도 높게 나오므로, class마다 따로 확인한다.
    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    # 요청(id)별 실제 정답과 best 모델 예측을 표로 만들어 CSV로 저장한다. 예: outputs/07_adamw_predictions.csv
    # valid["id"]·labels·best_pred는 모두 valid 행 순서 그대로라 같은 위치끼리 같은 요청이다.
    # .tolist()로 Tensor·Series를 파이썬 리스트로 바꿔 DataFrame 열로 넣는다.
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
    # 방금 추가한 이 후보의 요약 dict(summary[-1])에 복원 후 valid loss를 더한다.
    summary[-1]["restored_valid_loss"] = restored_loss
    # 복원 모델의 loss와, best_loss 사이의 차이가 1e-6(0.000001)보다 작으면 True를 기록한다.
    # 소수 계산의 아주 작은 오차는 허용하기 위해 == 대신 차이의 절댓값으로 비교한다.
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6

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
    # 이미지를 PNG 파일로 저장한다. 예: outputs/07_adamw_curves.png
    # dpi=150은 인치당 점 수로, 화면용보다 선명하게 저장한다.
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    # 이미지를 닫아 메모리를 비운다. 닫지 않으면 후보가 많을 때 이미지가 계속 쌓인다.
    plt.close()
    # 후보 이름, best epoch, best valid loss(소수점 6자리 반올림)를 출력한다.
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# -----------------------------------------------------------------------------
# 후보 비교표 저장
# -----------------------------------------------------------------------------
# result (type: DataFrame): 후보별 요약 dict 목록을 표로 만든다. 한 행이 한 후보(sgd, adamw, adamw_dropout)이다.
result = pd.DataFrame(summary)
# 세 설정의 비교표를 CSV로 저장한다.
result.to_csv(out / "07_comparison.csv", index=False)
# 생성된 파일 종류와, 설정 선택 단계라 test를 쓰지 않았음을 출력한다.
print("생성: outputs/07_*.csv, *.png / test 미사용")
