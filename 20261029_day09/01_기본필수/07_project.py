"""실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""
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

# -----------------------------------------------------------------------------
# 자료 읽기와 train/valid 분리
# -----------------------------------------------------------------------------
# base (type: Path): 이 스크립트 파일이 있는 폴더의 절대 경로이다.
#   - __file__: 지금 실행 중인 스크립트 파일의 경로이다.
#   - .resolve(): 상대 경로를 절대 경로로 바꾼다. .parent: 파일이 들어 있는 폴더를 고른다.
# 상대 경로 "points.csv"만 쓰면 터미널의 현재 위치를 기준으로 찾기 때문에, 다른 폴더에서 실행하면 파일을 못 찾는다. 그래서 base를 기준으로 삼는다.
base = Path(__file__).resolve().parent
# points.csv를 연다. f (type: TextIOWrapper): 열린 파일 객체이다.
#   - encoding="utf-8": 한글 등이 깨지지 않도록 UTF-8로 읽는다.
#   - newline="": csv 모듈이 줄바꿈을 직접 처리하도록 파이썬의 줄바꿈 변환을 끈다. csv 공식 문서가 요구하는 방식이다.
# with 블록이 끝나면 파일이 자동으로 닫힌다.
with (base / "points.csv").open(encoding="utf-8", newline="") as f:
    # rows (type: list[dict[str, str]]): 20개 행 각각이 {"id": ..., "x1": ..., "x2": ..., "label": ..., "split": ...} 사전이다.
    # CSV에서 읽은 값은 모두 문자열(str)이므로, 아래에서 float·int로 바꿔야 계산에 쓸 수 있다.
    rows = list(csv.DictReader(f))

# train_rows (type: list[dict]): split 열이 "train"인 행만 고른 리스트이다(12건). 모델이 직접 보고 배우는 자료이다.
train_rows = [r for r in rows if r["split"] == "train"]
# valid_rows (type: list[dict]): split 열이 "valid"인 행만 고른 리스트이다(4건). 학습에 쓰지 않고 성능 확인에만 쓴다.
# test 행(4건)은 여기서 아예 고르지 않는다. 최종 평가 전에 test를 보면 그 점수가 더 이상 "처음 보는 자료" 점수가 아니게 되기 때문이다.
valid_rows = [r for r in rows if r["split"] == "valid"]

# X_train (type: torch.Tensor, shape: (12, 2), dtype: float32): train 입력 특성이다. 행 하나가 점 하나, 열 2개가 x1·x2이다.
# float()로 바꾸는 이유: CSV 값은 문자열이고, nn.Linear의 가중치가 float32이므로 입력도 실수여야 곱셈이 된다.
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
# y_train (type: torch.Tensor, shape: (12,), dtype: int64): train 정답 class 번호(0 또는 1)이다.
# dtype=torch.long으로 지정하는 이유: CrossEntropyLoss는 정답을 정수 class 번호(int64)로 받도록 정해져 있어 실수형이면 오류가 난다.
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# X_valid (type: torch.Tensor, shape: (4, 2), dtype: float32): valid 입력 특성이다. 만드는 방식은 X_train과 같다.
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# y_valid (type: torch.Tensor, shape: (4,), dtype: int64): valid 정답 class 번호이다. 예측과 비교해 정확도를 잰다.
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)

# -----------------------------------------------------------------------------
# 모델·손실 함수·최적화기 만들기
# -----------------------------------------------------------------------------
# 기본필수는 CPU 고정, full-batch 80회이다. test는 Tensor로 만들지 않는다.
# 난수 시드를 42로 고정한다. 
torch.manual_seed(42)
# model (type: nn.Sequential): 입력 2개 → 은닉 뉴런 8개 → ReLU → 출력 2개로 이어지는 2층 신경망이다.
#   - nn.Linear(2, 8): 특성 2개(x1, x2)를 받아 은닉값 8개를 만든다(가중치 2×8=16개 + 편향 8개).
#   - nn.ReLU(): 음수는 0으로, 양수는 그대로 둔다. 이 비선형 단계가 있어야 층을 쌓아 곡선 경계도 배울 수 있다.
#   - nn.Linear(8, 2): 은닉값 8개를 class 수(2개)만큼의 점수(logit)로 바꾼다(가중치 8×2=16개 + 편향 2개).
#   - 학습 가능한 파라미터는 모두 16+8+16+2 = 42개이다.
# 마지막에 softmax가 없는 이유: CrossEntropyLoss가 내부에서 log-softmax를 직접 계산하기 때문이다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
# loss_fn (type: nn.CrossEntropyLoss): 분류 손실 함수이다. logit (N, 2)와 정답 번호 (N,)를 받아,
# 정답 class에 준 확률이 낮을수록 커지는 값을 N건 평균(reduction 기본값 "mean")으로 돌려준다.
loss_fn = nn.CrossEntropyLoss()
# optimizer (type: torch.optim.SGD): 경사하강법 최적화기이다.
#   - model.parameters(): 갱신 대상(두 Linear 층의 가중치·편향 42개)을 넘긴다. 여기에 넘긴 파라미터만 step()에서 바뀐다.
#   - lr=0.1: 학습률이다. 한 번 갱신할 때 "파라미터 -= 0.1 × 기울기"만큼 움직인다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# history (type: list[list[int, float]]): [epoch, train loss] 기록 목록이다. 첫 원소는 학습 전(epoch 0) 상태이다.
# 학습 전 loss를 기준점으로 남겨야, 마지막 loss와 비교해 학습이 실제로 loss를 줄였는지 판단할 수 있다.
# .item(): 원소 1개짜리 텐서를 파이썬 float로 꺼낸다. CSV·JSON에 저장하려면 텐서가 아니라 파이썬 숫자여야 한다.
history = [[0, loss_fn(model(X_train), y_train).item()]]

# -----------------------------------------------------------------------------
# 학습 루프 (zero_grad → forward → loss → backward → step)
# -----------------------------------------------------------------------------
# 모델을 학습 모드로 둔다. 이 모델에는 Dropout·BatchNorm이 없어 동작 차이는 없지만,
# "지금부터 학습한다"는 상태를 명시하는 것이 올바른 관례이다. 학습 모드가 gradient 기록을 켜는 것은 아니다.
model.train()
# epoch (type: int): 1부터 80까지 바뀌는 반복 번호이다. range(1, 81)로 쓰는 이유: 0번은 위에서 "학습 전"으로 이미 기록했기 때문이다.
# full-batch이므로 1 epoch = train 12건 전체로 한 번 갱신이다. 총 80번 갱신한다.
for epoch in range(1, 81):
    # 이전 반복에서 .grad에 쌓인 기울기를 지운다. PyTorch는 backward() 때 기울기를 덮어쓰지 않고 더하므로(누적),
    # 지우지 않으면 지난 epoch의 기울기가 섞여 엉뚱한 방향으로 갱신된다.
    optimizer.zero_grad()
    # forward와 loss를 한 줄에 계산한다. model(X_train)이 순전파(점수 계산), loss_fn(..., y_train)이 채점이다.
    # loss (type: torch.Tensor, shape: (), 스칼라): 이번 epoch의 평균 손실이며, 계산 과정이 그래프로 기록되어 있다.
    loss = loss_fn(model(X_train), y_train)
    # 역전파를 실행한다. loss에서 출발해 기록된 계산 그래프를 거꾸로 따라가며,
    # 각 파라미터가 loss에 미치는 영향(∂loss/∂파라미터)을 구해 그 파라미터의 .grad에 저장한다.
    # 이 줄은 "어느 방향으로 고쳐야 하는지"만 계산할 뿐, 가중치 값 자체는 바꾸지 않는다.
    loss.backward()
    # .grad에 들어 있는 기울기로 파라미터를 실제로 갱신한다. SGD 규칙에 따라 "파라미터 ← 파라미터 - 0.1 × .grad"를 계산한다.
    # 반드시 backward() 다음에 호출해야 한다. 순서가 바뀌면 zero_grad 뒤의 빈 기울기로 갱신하게 되어 아무것도 배우지 못한다.
    # 쉽게 말해, backward가 "틀린 곳 표시하기"라면 step은 "표시한 곳을 실제로 고치기"이다.
    optimizer.step()
    # 갱신 후 같은 train 자료의 loss를 기록한다. 마지막 batch loss와 구분한다.
    # 위의 loss 변수는 갱신 "전" 가중치로 잰 값이므로, 갱신 "후" 상태를 기록하려면 다시 계산해야 한다.
    # no_grad 안에서 계산하는 이유: 기록용 계산은 역전파에 쓰지 않으므로 계산 그래프를 만들 필요가 없고, 메모리를 아낀다.
    with torch.no_grad():
        # [epoch 번호, 갱신 후 train loss(type: float)]를 history에 추가한다. 루프가 끝나면 history는 81개 행이 된다(0~80).
        history.append([epoch, loss_fn(model(X_train), y_train).item()])

# -----------------------------------------------------------------------------
# valid 평가
# -----------------------------------------------------------------------------
# 모델을 평가 모드로 바꾼다. Dropout·BatchNorm 같은 층이 평가용으로 동작하게 하는 스위치이다.
# 평가 모드는 gradient 기록을 끄지 않으므로, 아래에서 torch.no_grad()를 따로 함께 쓴다.
model.eval()
# no_grad 블록 안에서는 계산 그래프를 기록하지 않는다. 평가에는 역전파가 필요 없기 때문이다.
with torch.no_grad():
    # logits (type: torch.Tensor, shape: (4, 2), dtype: float32): 학습을 마친 모델이 valid 4건에 낸 class별 점수이다.
    # 열 0은 class 0 점수, 열 1은 class 1 점수이다.
    logits = model(X_valid)
    # prediction (type: torch.Tensor, shape: (4,), dtype: int64): 각 점에서 점수가 더 큰 class 번호(0 또는 1)이다.
    #   - dim=1: 한 행 안에서 열(class 축) 방향으로 최댓값 위치를 찾는다. 그래서 점마다 번호 하나가 나와 shape가 (4,)가 된다.
    #   - dim=0으로 쓰면 4건 중 어느 행이 가장 큰지를 class별로 찾게 되어 shape가 (2,)가 되고, 예측이 아닌 엉뚱한 값이 된다.
    #   - softmax를 거치지 않아도 되는 이유: softmax는 크기 순서를 바꾸지 않으므로 logit의 최댓값 위치와 확률의 최댓값 위치가 같다.
    #   - argmax의 결과는 int64이므로 y_valid(int64)와 바로 비교할 수 있다.
    # 쉽게 말해, 모델이 두 답안에 매긴 점수 중 더 높은 쪽을 최종 답으로 고르는 줄이다.
    prediction = logits.argmax(dim=1)
    # accuracy (type: float, 0~1): valid 정확도이다.
    #   - (prediction == y_valid): 예측과 정답이 같으면 True인 bool 텐서 (4,)이다.
    #   - .float().mean(): True=1.0, False=0.0으로 바꿔 평균을 내면 맞힌 비율이 된다.
    # valid가 4건뿐이므로 정확도는 0, 0.25, 0.5, 0.75, 1.0 중 하나만 나온다. 한 건 차이로 25%p가 움직이므로 표본 수와 함께 해석해야 한다.
    accuracy = (prediction == y_valid).float().mean().item()

# -----------------------------------------------------------------------------
# 결과 파일 저장 (training_history.csv, final_metrics.json, gpu_check.txt)
# -----------------------------------------------------------------------------
# training_history.csv를 쓰기 모드("w")로 연다.
#   - encoding="utf-8-sig": 파일 앞에 BOM을 붙인 UTF-8이다. 엑셀에서 열어도 한글이 깨지지 않게 하려는 설정이다.
#   - newline="": csv.writer가 줄바꿈을 직접 넣으므로, 파이썬이 줄바꿈을 한 번 더 바꿔 빈 줄이 생기는 것을 막는다.
with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: csv.writer): 리스트를 CSV 한 줄로 바꿔 쓰는 객체이다.
    writer = csv.writer(f)
    # 머리글 행을 쓴다. train_loss_eval이라는 이름은 "갱신 후 train 자료로 다시 잰 loss"라는 뜻이다.
    writer.writerow(["epoch", "train_loss_eval"])
    # history의 81개 행(epoch 0~80)을 한 번에 쓴다.
    writer.writerows(history)
# metrics (type: dict): final_metrics.json에 저장할 최종 지표 사전이다.
#   - "device": 이번 실행은 CPU에 고정했으므로 "cpu"이다.
#   - "epochs": 갱신 횟수 80이다. "train_rows"·"valid_rows": 표본 수 12·4이다. 정확도는 표본 수와 함께 봐야 의미가 있다.
#   - "test_used": False로 test를 쓰지 않았음을 기록으로 남긴다.
#   - "valid_accuracy": 위에서 구한 valid 정확도(type: float)이다.
#   - "initial_train_loss"·"final_train_loss": history 첫 값(학습 전)과 마지막 값(80회 후)이다. 둘을 비교하면 학습 효과가 보인다.
#   - "valid_ids": valid 4건의 id 목록(type: list[str])이다. 어떤 점이 맞고 틀렸는지 원본과 대조할 수 있게 한다.
#   - "actual"·"predicted": 정답과 예측을 .tolist()로 파이썬 리스트(type: list[int])로 바꾼 값이다. JSON은 텐서를 저장하지 못한다.
metrics = {"device": "cpu", "epochs": 80, "train_rows": 12, "valid_rows": 4,
           "test_used": False, "valid_accuracy": accuracy,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_ids": [r["id"] for r in valid_rows], "actual": y_valid.tolist(),
           "predicted": prediction.tolist()}
# metrics를 JSON 문자열로 바꿔 final_metrics.json에 쓴다.
#   - ensure_ascii=False: 한글 등 비ASCII 문자를 \uXXXX로 바꾸지 않고 그대로 저장한다.
#   - indent=2: 들여쓰기 2칸으로 사람이 읽기 쉽게 줄을 나눈다.
(base / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# gpu_check.txt에 실행 환경을 기록한다. 같은 결과를 재현하려면 어떤 환경에서 돌렸는지가 남아 있어야 한다.
#   - torch.__version__ (type: str): 설치된 PyTorch 버전이다.
#   - torch.version.cuda (type: str 또는 None): PyTorch가 빌드될 때 쓴 CUDA 버전이다. CPU 전용 빌드면 None이다.
#   - torch.cuda.is_available() (type: bool): 지금 이 컴퓨터에서 GPU를 실제로 쓸 수 있는지 여부이다.
#   - selected_device=cpu: GPU가 있더라도 기본필수는 CPU로 고정했다는 사실을 따로 적는다. "쓸 수 있음"과 "실제로 씀"은 다른 정보이다.
(base / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")

# -----------------------------------------------------------------------------
# 결과 요약 출력
# -----------------------------------------------------------------------------
# 분할 구성을 출력한다. test 4건은 쓰지 않고 보관했음을 화면에서도 확인한다.
print("분할: train 12 / valid 4 / test 4(보관)")
# 학습 전 → 80회 후 train loss를 소수 6자리로 출력한다. 줄어들었다면 학습 루프가 올바르게 연결된 것이다.
print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
# valid 정확도를 소수 4자리로 출력한다. 학습에 쓰지 않은 자료의 성능이므로 train loss보다 일반화 성능에 가깝다.
print("valid 정확도:", round(accuracy, 4))
# valid 4건의 예측 class 번호 목록(type: list[int])을 출력한다.
print("valid 예측:", prediction.tolist())
# 생성한 결과 파일 3개의 이름을 출력한다.
print("생성: gpu_check.txt, training_history.csv, final_metrics.json")
