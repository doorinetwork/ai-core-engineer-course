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
# batch_size 후보 정하기
# -----------------------------------------------------------------------------
# batch_size 후보를 [8, 32]로 둔다. 총 train 행은 240건이다.
# drop_last=False이므로 32인 경우 마지막 배치는 16건이다.
# batch_sizes (type: list[int]): 비교할 batch 크기 후보 2개이다.
#   - 8: 작은 batch이다. 한 epoch에 가중치를 자주(30번) 갱신하지만, batch마다 gradient가 더 흔들린다.
#   - 32: 큰 batch이다. 한 epoch에 갱신 횟수는 적지만(8번), batch 평균 gradient가 더 안정적이다.
batch_sizes = [8, 32]

# -----------------------------------------------------------------------------
# batch_size별로 batch를 나눠 보고 기록하기
# -----------------------------------------------------------------------------
# rows (type: list): batch_size 하나당 [batch_size, iterations, last_batch, samples] 한 줄씩 모을 빈 리스트이다.
rows = []
for size in batch_sizes:
    # loader (type: DataLoader): train 240건을 size개씩 꺼내 주는 반복 가능한(iterable) 객체이다.
    #   - TensorDataset(X_train, y_train): X와 y를 행 번호끼리 짝지은 dataset이다.
    #   - batch_size=size: 한 iteration에 꺼내는 sample 개수이다.
    #   - shuffle=False: 순서를 섞지 않고 0번 행부터 차례대로 꺼낸다.
    #     이 실습은 batch 개수와 크기만 세는 것이 목적이므로, 섞든 섞지 않든 개수는 같지만
    #     순서를 고정해 두면 어느 행이 어느 batch에 들어가는지 예측할 수 있다.
    # drop_last를 지정하지 않았으므로 기본값 False가 적용되어, size개를 못 채운 마지막 묶음도 버리지 않는다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=size, shuffle=False)
    # counts (type: list[int]): 각 batch에 들어 있는 sample 수를 차례대로 담은 리스트이다.
    # 리스트 컴프리헨션으로 loader를 한 바퀴(1 epoch) 돌면서, batch마다 정답 묶음 yb의 길이를 센다.
    # xb (type: Tensor, shape [batch 크기, 2]): 이번 batch의 입력 묶음이다. 여기서는 개수만 세므로 쓰지 않는다.
    # yb (type: Tensor, shape [batch 크기]): 이번 batch의 정답 묶음이다. len(yb)가 곧 batch 크기이다.
    # 예: size=8이면 [8, 8, ..., 8] (30개), size=32이면 [32, 32, 32, 32, 32, 32, 32, 16] (8개)이다.
    counts = [len(yb) for xb, yb in loader]
    # 이번 batch_size의 결과 한 줄을 rows에 추가한다.
    #   - size: batch 크기이다.
    #   - len(counts): batch 개수, 즉 한 epoch의 iteration 수(= 가중치 갱신 횟수)이다.
    #   - counts[-1]: 마지막 batch의 크기이다. 인덱스 -1은 리스트의 맨 끝 원소를 뜻한다.
    #   - sum(counts): 한 epoch에 처리한 sample 총수이다. drop_last=False이므로 항상 240이어야 한다.
    rows.append([size, len(counts), counts[-1], sum(counts)])
    # 이번 batch_size의 결과를 콘솔에 출력한다.
    # 예상 출력(240건 기준 손계산):
    #   batch: 8 iterations: 30 마지막: 8 처리 수: 240
    #     → 240 ÷ 8 = 30으로 나누어떨어지므로 30개 batch가 모두 8건이다.
    #   batch: 32 iterations: 8 마지막: 16 처리 수: 240
    #     → 240 ÷ 32 = 7.5이므로 꽉 찬 batch 7개(32 × 7 = 224건) 뒤에 남은 240 - 224 = 16건이 마지막 batch가 된다.
    #       그래서 iteration 수는 올림(7.5) = 8이다.
    print("batch:", size, "iterations:", len(counts), "마지막:", counts[-1], "처리 수:", sum(counts))

# -----------------------------------------------------------------------------
# 결과 CSV 저장
# -----------------------------------------------------------------------------
# rows를 DataFrame(type: DataFrame, shape [2, 4])으로 바꿔 outputs/03_batches.csv에 저장한다.
#   - columns: 각 열의 이름이다. rows 한 줄의 순서 [size, len(counts), counts[-1], sum(counts)]와 같은 순서로 붙인다.
#   - index=False: DataFrame의 행 번호(0, 1)는 의미 없는 값이라 CSV에 쓰지 않는다.
# 예상 파일 내용:
#   batch_size,iterations,last_batch,samples
#   8,30,8,240
#   32,8,16,240
pd.DataFrame(rows, columns=["batch_size", "iterations", "last_batch", "samples"]).to_csv(
    out / "03_batches.csv", index=False)
