"""실습 07~08: 앞 실습을 한 파일로 연결하고 환경·학습·평가 결과를 저장합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [표준 라이브러리] csv: CSV 파일을 읽고 쓰는 모듈이다.
import csv
# [표준 라이브러리] json: JSON 형식 데이터를 읽고 쓰는 모듈이다.
import json
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산·자동미분 라이브러리이다.
import torch
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
#   - 목적: 예시 손실 계산에 쓸 CrossEntropyLoss를 가져오기 위해 쓴다.
from torch import nn
# [외부 라이브러리] torch.utils.data: 학습 데이터를 묶고 공급하는 도구 모음이다.
#   - 목적: X와 y를 한 쌍으로 묶고(TensorDataset), 정해진 크기로 잘라 차례로 꺼내기(DataLoader) 위해 쓴다.
#   - 기능: TensorDataset은 같은 길이의 텐서들을 인덱스별 (x, y) 쌍으로 묶고,
#           DataLoader는 그 Dataset을 batch 단위로 반복(for문)할 수 있게 만든다.
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
# base (type: Path): 이 Python 파일이 들어 있는 폴더의 절대 경로이다.
base = Path(__file__).resolve().parent
# out (type: Path): 결과 파일을 저장할 outputs 폴더 경로이다.
out = base / "outputs"
# outputs 폴더를 만든다. exist_ok=True: 폴더가 이미 있어도 오류를 내지 않고 그대로 쓴다.
out.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# data/points.csv를 연다. encoding="utf-8"은 한글 등 문자를 올바르게 읽기 위한 인코딩이고,
# newline=""은 csv 모듈이 줄바꿈을 직접 처리하도록 맡기라는 csv 공식 권장 설정이다.
# f (type: TextIOWrapper): 열린 파일 객체이다. with 블록이 끝나면 자동으로 닫힌다.
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    # rows (type: list[dict[str, str]]): CSV의 모든 행이다. 
    rows = list(csv.DictReader(f))
# train_rows (type: list[dict]): split 열이 "train"인 행만 고른 학습용 자료이다. 
train_rows = [r for r in rows if r["split"] == "train"]
# valid_rows (type: list[dict]): split 열이 "valid"인 행만 고른 검증용 자료이다. 학습에는 쓰지 않고 성능 측정에만 쓴다.
valid_rows = [r for r in rows if r["split"] == "valid"]

# -----------------------------------------------------------------------------
# 텐서 만들기
# -----------------------------------------------------------------------------
# X_train (type: torch.Tensor, shape: (train 건수, 2), dtype: float32): 학습 입력이다. 문자열을 float로 바꿔 [x1, x2] 한 행씩 쌓는다.
# float로 바꾸는 이유: nn.Linear의 가중치가 float32이므로 입력도 실수여야 행렬곱이 된다.
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
# y_train (type: torch.Tensor, shape: (train 건수,), dtype: int64): 학습 정답 class 번호(0 또는 1)이다.
# dtype=torch.long인 이유: CrossEntropyLoss는 정답을 class 번호(정수 int64)로 받도록 정해져 있기 때문이다.
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# X_valid (type: torch.Tensor, shape: (valid 건수, 2), dtype: float32): 검증 입력이다.
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# y_valid (type: torch.Tensor, shape: (valid 건수,), dtype: int64): 검증 정답 class 번호이다.
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않는다. 순서를 바꾸거나 다시 분할하지 않는다.
# (이유: test는 모든 선택이 끝난 뒤 한 번만 쓰는 최종 시험이다. 지금 보면 그 결과에 맞춰 설정을 고치게 되어 최종 평가가 공정하지 않게 된다.
#  또 정해진 분할을 그대로 써야 같은 결과를 다시 재현할 수 있다.)

# -----------------------------------------------------------------------------
# 재현성 고정, 장치 선택, DataLoader 만들기
# -----------------------------------------------------------------------------
# 전역 난수 시드를 42로 고정한다. 바로 아래 nn.Linear의 가중치 초기값이 이 난수로 정해지므로,
# 시드를 고정해야 실행할 때마다 같은 출발점에서 학습해 같은 기록이 나온다.
torch.manual_seed(42)

# device (type: torch.device): 계산할 장치이다. GPU(CUDA)를 쓸 수 있으면 첫 번째 GPU "cuda:0", 없으면 "cpu"를 고른다.
# 이렇게 쓰면 같은 코드가 GPU가 있는 컴퓨터와 없는 컴퓨터 모두에서 실행된다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loader (type: DataLoader): 학습 자료를 32건씩 묶어서 꺼내 주는 반복 가능한 객체이다.
#   - TensorDataset(X_train, y_train): 같은 행 번호의 입력과 정답을 (x, y) 쌍으로 묶는다.
#   - batch_size=32: 한 번의 갱신에 32건을 쓴다. 전체를 한 번에 쓰는 full-batch보다 한 epoch에 여러 번 갱신한다.
#     train 건수가 32로 나누어떨어지지 않으면 마지막 batch는 32건보다 작다. drop_last 기본값이 False라서 그 batch도 버리지 않는다.
#   - shuffle=True: epoch마다 샘플 순서를 섞는다. 매번 같은 순서로 배우면 순서의 고정 패턴에 영향을 받기 때문이다.
#   - generator=torch.Generator().manual_seed(43): 섞는 순서만 담당하는 전용 난수 생성기를 시드 43으로 고정한다.
#     전역 시드(42)와 분리해 두면 모델 초기화와 섞는 순서가 서로 영향을 주지 않고, 섞는 순서가 매 실행 똑같이 재현된다.
#   - num_workers=0: 별도 프로세스 없이 메인 프로세스에서 자료를 꺼낸다. 작은 자료에서 가장 단순하고 결과가 재현된다.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)

# -----------------------------------------------------------------------------
# 모델·손실 함수·최적화기 만들기
# -----------------------------------------------------------------------------
# model (type: nn.Sequential): 입력 2개 → 은닉 뉴런 8개 → ReLU → 출력 2개(class 0 점수, class 1 점수)인 2층 신경망이다.
# .to(device): 모델의 가중치를 선택한 장치로 옮긴다. 모델과 입력이 같은 장치에 있어야 계산할 수 있다.
# 마지막에 softmax가 없는 이유: CrossEntropyLoss가 내부에서 log-softmax를 직접 계산하기 때문이다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
# loss_fn (type: nn.CrossEntropyLoss): 분류 손실 함수이다. logit (N, 2)와 정답 번호 (N,)를 받아
# 정답 class에 준 확률이 낮을수록 커지는 값을 N건 평균(reduction 기본값 "mean")으로 돌려준다.
loss_fn = nn.CrossEntropyLoss()
# optimizer (type: torch.optim.SGD): 확률적 경사하강법 최적화기이다.
#   - model.parameters(): 갱신 대상(두 Linear 층의 가중치와 편향)을 넘긴다. 여기에 넘긴 파라미터만 step()에서 바뀐다.
#   - lr=0.1: 학습률이다. 한 번 갱신할 때 "파라미터 ← 파라미터 - 0.1 × 기울기"만큼 움직인다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# Xv, yv (type: torch.Tensor): 검증 입력·정답을 모델과 같은 장치로 옮긴 것이다. 평가할 때마다 옮기지 않도록 미리 한 번 옮긴다.
Xv, yv = X_valid.to(device), y_valid.to(device)
# Xt, yt (type: torch.Tensor): 학습 입력·정답 전체를 장치로 옮긴 것이다. 학습용이 아니라 "현재 모델의 train loss 평가"에 쓴다.
Xt, yt = X_train.to(device), y_train.to(device)
# history (type: list[list]): 기록 시점마다 [epoch, train_loss, valid_loss, accuracy]를 한 행씩 쌓을 목록이다.
history = []

# -----------------------------------------------------------------------------
# 학습 루프 (zero_grad → forward → loss → backward → step) + 10 epoch마다 기록
# -----------------------------------------------------------------------------
# epoch (type: int): 0부터 80까지 81개 값을 돈다. range(81)인 이유: epoch 0은 "학습 전" 기록용이고,
# 1~80이 실제 학습 80 epoch이다. 그래서 기록 시점은 0, 10, 20, ..., 80의 9개가 된다.
for epoch in range(81):
    # epoch 0에서는 학습을 건너뛰고 아래 기록만 한다. 학습 전 출발점을 남겨야 학습 효과를 비교할 수 있다.
    if epoch > 0:
        # 학습 모드로 둔다. 이 모델에는 Dropout·BatchNorm이 없어 동작 차이는 없지만,
                # 직전 기록 단계에서 eval()로 바꿨으므로 학습 전에 다시 train()으로 되돌리는 것이 올바른 순서이다.
                # train()은 gradient 기록을 켜는 함수가 아니다(기록 여부는 no_grad가 정한다).
        model.train()
        # xb (type: torch.Tensor, shape: (≤32, 2)), yb (type: torch.Tensor, shape: (≤32,)): 이번 batch의 입력과 정답이다.
        for xb, yb in loader:
            # batch를 모델과 같은 장치로 옮긴다. DataLoader는 CPU 텐서를 주므로 GPU 학습이면 여기서 옮겨야 한다.
            xb, yb = xb.to(device), yb.to(device)
            # 이전 batch에서 .grad에 쌓인 기울기를 지운다. backward()는 기울기를 덮어쓰지 않고 더하기(누적) 때문이다.
            optimizer.zero_grad()
            # 순전파(model(xb))로 batch의 class 점수를 내고, 정답 yb와 비교해 평균 손실을 계산한다.
            # loss (type: torch.Tensor, shape: (), 스칼라): 이번 batch의 평균 손실이며 계산 과정이 그래프로 기록되어 있다.
            loss = loss_fn(model(xb), yb)
            # 현재 batch의 loss로 역전파하고 optimizer를 한 번 갱신한다.
            # 역전파를 실행한다. loss에서 출발해 기록된 계산 그래프를 거꾸로 따라가며
            # 각 파라미터의 기울기(∂loss/∂파라미터)를 구해 그 파라미터의 .grad에 저장한다.
            # 이 줄은 고칠 방향만 계산하고 가중치 값은 바꾸지 않는다.
            loss.backward()
            # .grad에 들어 있는 기울기로 파라미터를 실제로 한 번 갱신한다. SGD 규칙 "파라미터 ← 파라미터 - lr × .grad"를 적용한다.
            # 반드시 backward() 다음에 호출해야 이번 batch의 기울기로 갱신된다.
            # 이 두 줄이 batch마다 실행되므로, 한 epoch 동안 batch 개수만큼 갱신이 일어난다.
            optimizer.step()
    # epoch=0은 학습 전이다. 로그는 갱신이 끝난 동일 모델로 측정한다.
        # (이유: 학습 중 batch마다 나온 loss는 서로 다른 시점의 모델이 낸 값이 섞인 것이다.
        #  epoch이 끝난 뒤 가중치를 고정한 한 모델로 train과 valid를 모두 재야 두 값을 공정하게 비교할 수 있다.)
        # epoch % 10 == 0: epoch을 10으로 나눈 나머지가 0일 때, 곧 0, 10, 20, ..., 80일 때만 기록한다.
    if epoch % 10 == 0:
        # 평가 모드로 바꾼다. Dropout·BatchNorm 같은 층이 평가용으로 동작하게 하는 스위치이다.
        model.eval()
        # 계산 그래프 기록을 끈다. 평가에는 역전파가 필요 없으므로 메모리와 시간을 아끼고, 평가 계산이 학습 기울기에 섞이지 않게 한다.
        # eval()만으로는 gradient 기록이 꺼지지 않으므로 no_grad를 함께 쓴다.
        with torch.no_grad():
            # train_loss (type: float): 현재 모델로 train 전체를 한 번에 다시 넣어 잰 평균 손실이다.
            # 학습 중 batch loss가 아니라 "고정된 모델의 평가 loss"이므로 CSV 열 이름이 train_loss_eval이다.
            # .item(): 원소 1개짜리 텐서를 파이썬 float로 꺼낸다.
            train_loss = loss_fn(model(Xt), yt).item()
            # logits (type: torch.Tensor, shape: (valid 건수, 2)): 같은 모델이 valid 각 건에 낸 class별 점수이다.
            logits = model(Xv)
            # valid_loss (type: float): valid 전체의 평균 손실이다. 처음 보는 자료에서의 손실이므로 일반화 정도를 보여 준다.
            valid_loss = loss_fn(logits, yv).item()
            # accuracy (type: float, 0~1): valid 정확도이다.
            #   - logits.argmax(dim=1): 각 행에서 점수가 더 큰 class 번호를 예측값으로 고른다. dim=1은 class 축이다.
            #   - (... == yv): 예측과 정답이 같으면 True인 bool 텐서를 만든다.
            #   - .float().mean(): True=1.0, False=0.0으로 바꿔 평균하면 맞힌 비율이 된다.
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        # 이번 기록 시점의 한 행을 history에 추가한다.
        history.append([epoch, train_loss, valid_loss, accuracy])
        # 화면에 epoch, train loss(소수 6자리), valid loss(소수 6자리), valid 정확도(소수 4자리)를 출력한다.
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

# -----------------------------------------------------------------------------
# 학습 기록 저장
# -----------------------------------------------------------------------------
# outputs/training_history.csv를 쓰기 모드("w")로 연다.
# encoding="utf-8-sig": 파일 맨 앞에 BOM을 넣어 엑셀에서 열어도 글자가 깨지지 않게 한다.
# newline="": csv 모듈이 줄바꿈을 직접 처리하게 해서 빈 줄이 끼지 않게 한다.
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: csv.writer 객체): 리스트를 쉼표로 구분된 한 행으로 기록한다.
    writer = csv.writer(f)
    # 첫 행에 열 이름을 쓴다. train_loss_eval은 "학습 중 loss가 아니라 고정 모델로 다시 잰 train loss"라는 뜻을 이름에 담은 것이다.
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    
    writer.writerows(history)

# -----------------------------------------------------------------------------
# 최종 모델로 valid 예측 만들기
# -----------------------------------------------------------------------------
# 80 epoch 학습이 끝난 최종 모델을 평가 모드로 둔다.
model.eval()
# 예측에는 역전파가 필요 없으므로 계산 그래프 기록을 끈다.
with torch.no_grad():
    # logits (type: torch.Tensor, shape: (40, 2), device: 선택한 device): 최종 모델이 valid 40건에 낸 class별 점수이다.
    logits = model(Xv)
    # logits의 클래스 축에서 가장 큰 점수의 인덱스를 구한다.
    # CSV로 저장하기 전에 CPU로 옮긴다. y_valid의 순서는 그대로 유지한다.
    # predictions (type: torch.Tensor, shape: (40,), dtype: int64, device: cpu): valid 각 행의 예측 class 번호(0 또는 1)이다.
    #   - logits.argmax(dim=1): dim=1(class 축)을 따라 점수가 가장 큰 위치의 번호를 고른다.
    #     dim=0으로 쓰면 40건 중 어느 행이 가장 큰지를 class별로 고르게 되어 shape (2,)의 엉뚱한 값이 나온다.
    #     argmax는 정수 위치를 돌려주므로 결과 dtype은 int64(long)이고, 정답 y_valid와 같은 dtype이 된다.
    #   - .cpu(): GPU에서 계산했다면 결과를 CPU 메모리로 옮긴다. 이미 CPU이면 같은 텐서를 그대로 돌려준다.
    #     CPU로 옮기는 이유: 아래 저장 단계에서 CPU 텐서인 y_valid와 함께 int()로 하나씩 꺼내 CSV에 쓰기 때문이다.
    # argmax는 행 순서를 바꾸지 않으므로 predictions[i]는 valid_rows[i], y_valid[i]와 같은 요청의 예측이다.
    predictions = logits.argmax(dim=1).cpu()

# -----------------------------------------------------------------------------
# 최종 지표·예측·환경 정보 저장
# -----------------------------------------------------------------------------
# metrics (type: dict): final_metrics.json에 담을 요약 지표이다.
#   - "device" (type: str): 실제로 학습에 쓴 장치 이름(예: "cuda:0" 또는 "cpu")이다.
#   - "torch" (type: str): 설치된 PyTorch 버전이다. 같은 결과를 재현하려면 버전도 같아야 하므로 기록한다.
#   - "epochs" (type: int): 실제 학습 epoch 수 80이다(epoch 0은 학습 전 기록이라 세지 않는다).
#   - "train_rows", "valid_rows" (type: int): 학습·검증에 쓴 표본 수(120, 40)이다. 정확도는 표본 수와 함께 봐야 의미가 정확해진다.
#   - "test_used" (type: bool): test 자료를 쓰지 않았다는 사실을 False로 명시한다.
#   - "initial_train_loss" (type: float): history 첫 행(epoch 0, 학습 전)의 train loss이다.
#   - "final_train_loss" (type: float): history 마지막 행(epoch 80)의 train loss이다. 처음 값과 비교해 학습으로 손실이 줄었는지 본다.
#   - "valid_loss", "valid_accuracy" (type: float): epoch 80 최종 모델의 valid 손실과 정확도이다.
metrics = {"device": str(device), "torch": str(torch.__version__), "epochs": 80,
           "train_rows": len(train_rows), "valid_rows": len(valid_rows), "test_used": False,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_loss": history[-1][2], "valid_accuracy": history[-1][3]}

# metrics를 JSON 문자열로 바꿔 outputs/final_metrics.json에 쓴다.
#   - ensure_ascii=False: 한글 같은 비영어 문자를 \uXXXX로 바꾸지 않고 그대로 쓴다.
#   - indent=2: 키마다 줄을 바꾸고 2칸 들여 써서 사람이 읽기 쉽게 한다.
(out / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# outputs/validation_predictions.csv를 쓰기 모드로 연다(BOM 포함 UTF-8, 줄바꿈은 csv 모듈이 처리).
with (out / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: csv.writer 객체)를 만들고 바로 헤더 행 [id, actual, predicted]를 쓴다. 세미콜론(;)은 두 문장을 한 줄에 쓰는 문법이다.
    writer = csv.writer(f); writer.writerow(["id", "actual", "predicted"])
    # zip(valid_rows, y_valid, predictions): 같은 위치의 (행 딕셔너리, 정답 텐서 원소, 예측 텐서 원소)를 하나씩 묶는다.
    #   - r["id"] (type: str): 요청 ID이다.
    #   - int(y), int(p) (type: int): 원소 1개짜리 텐서를 파이썬 정수로 바꾼 정답과 예측이다.
    # 세 목록이 모두 valid_rows 순서이므로 각 행의 ID·정답·예측이 서로 같은 요청을 가리킨다.
    writer.writerows((r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions))
# outputs/gpu_check.txt에 환경 정보 4줄을 쓴다.
#   - torch.__version__: PyTorch 버전이다.
#   - torch.version.cuda: 이 PyTorch가 어떤 CUDA 버전으로 빌드되었는지이다. CPU 전용 빌드이면 None이다.
#   - torch.cuda.is_available(): 지금 이 컴퓨터에서 GPU를 실제로 쓸 수 있는지(True/False)이다.
#     CUDA 빌드여도 GPU나 드라이버가 없으면 False이므로 cuda_build와 따로 기록한다.
#   - device: 위 판단으로 실제 선택한 장치이다.
(out / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device={device}\n", encoding="utf-8")
print("생성: outputs의 환경·학습·검증 결과")

# -----------------------------------------------------------------------------
# 추가 A10: valid 오답 목록 만들기
# -----------------------------------------------------------------------------
# 추가 A10: valid의 실제값과 예측을 비교해 오답 목록을 만든다.
# 입력: valid_rows, y_valid, predictions는 동일한 40개 ID 순서이다.
# 반환 형식: errors=[(요청ID, 실제정답int, 예측int), ...].
# 조건: 실제값과 예측이 다른 행만 포함하고 원래 valid 순서를 유지한다.
# 금지: test 자료 사용, 정답 변경, 오답이 없을 때 가상의 오답 추가.
# 완료: 오류 행 수 = 40 - 맞힌 개수, 저장 파일은 헤더를 포함한다.

# 실제 오답 행만 ID와 함께 수집한다.
# errors (type: list[tuple[str, int, int]]): (요청 ID, 실제 정답, 예측) 묶음 중 실제 정답과 예측이 다른 것만 모은 목록이다.
#   - zip(valid_rows, y_valid, predictions): 세 목록을 같은 위치끼리 묶어 valid 원래 순서대로 하나씩 꺼낸다.
#     zip은 순서를 바꾸지 않으므로 errors도 valid 원래 순서를 그대로 따른다.
#   - int(y), int(p): 텐서 원소를 파이썬 정수로 바꾼다. 저장 형식이 정수이고, 정수끼리 비교해야 결과가 True/False 한 값으로 나온다.
#   - if int(y) != int(p): 정답과 예측이 다른 행만 남긴다. 맞힌 행은 넣지 않으므로 len(errors) = 40 - 맞힌 개수가 된다.
# 오답이 하나도 없으면 errors는 빈 리스트 []로 남고, 저장 파일에는 헤더만 들어간다. 가상의 오답을 만들어 넣지 않는다.
# 쉽게 말해, 모의고사 40문제를 채점해서 틀린 문제만 문제 번호와 함께 오답 노트에 순서대로 옮겨 적는 일이다.
errors = [(r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions) if int(y) != int(p)]

# outputs/validation_errors.csv를 쓰기 모드로 연다(BOM 포함 UTF-8, 줄바꿈은 csv 모듈이 처리).
with (out / "validation_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: csv.writer 객체): 리스트·튜플을 한 행으로 기록한다.
    writer = csv.writer(f)
    # 헤더 행을 쓴다. 오답이 0건이어도 이 헤더는 항상 들어간다.
    writer.writerow(["id", "actual", "predicted"])
    # 오답 행을 valid 원래 순서대로 모두 기록한다.
    writer.writerows(errors)
# 오답 개수를 출력한다. 이 값은 40 - (valid_accuracy × 40)과 같아야 한다.
print("valid 오답 수:", len(errors))
