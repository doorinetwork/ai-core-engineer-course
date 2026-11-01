"""실습 07: 120건 학습, 40건 검증. test 40건은 다음 독립 평가용으로 보관합니다."""
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

# [프로젝트 내부 모듈] check_gpu.choose_device: 같은 폴더의 check_gpu.py에 있는 장치 선택 함수이다.
#   - 목적: CUDA GPU가 있으면 cuda:0, 없으면 cpu를 고르고 환경 정보를 기록하기 위해 쓴다.
from check_gpu import choose_device
# [프로젝트 내부 모듈] model.SimpleANN: 같은 폴더의 model.py에 있는 2→8→2 신경망 클래스이다.
#   - 목적: 학습할 모델 구조를 만들기 위해 쓴다. restore.py도 같은 클래스를 써서 구조가 똑같이 맞춰진다.
#   - 기능: 입력 2개 → 은닉 8개(ReLU) → 출력 2개(logits)를 계산한다.
from model import SimpleANN
# [프로젝트 내부 모듈] train.train_one_epoch, train.evaluate: 같은 폴더의 train.py에 있는 학습·평가 함수이다.
#   - 목적: 한 epoch 학습과 가중치를 바꾸지 않는 평가를 분리해서 부르기 위해 쓴다.
#   - 기능: train_one_epoch는 표본 가중 평균 train loss를, evaluate는 (loss, 정확도, 예측)을 돌려준다.
from train import train_one_epoch, evaluate

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

# -----------------------------------------------------------------------------
# 재현성 고정, 장치·모델·DataLoader·손실·최적화기 준비
# -----------------------------------------------------------------------------
# 전역 난수 시드를 42로 고정한다. 바로 아래 nn.Linear의 가중치 초기값이 이 난수로 정해지므로,
# 시드를 고정해야 실행할 때마다 같은 출발점에서 학습해 같은 기록이 나온다.
torch.manual_seed(42)
# device (type: torch.device): 계산 장치(cuda:0 또는 cpu)이다.
# info (type: dict): torch 버전, CUDA 빌드, CUDA 사용 가능 여부, 선택한 장치 등 환경 기록이다. 뒤에서 gpu_check.txt로 저장한다.
device, info = choose_device()
# model (type: SimpleANN): 2→8→2 신경망이다. .to(device)로 가중치를 선택한 장치로 옮긴다.
model = SimpleANN().to(device)
# loader (type: DataLoader): train 120건을 32건씩 꺼내는 반복자이다. 한 epoch는 32·32·32·24의 4개 batch이다.
#   - batch_size=32: 한 번 갱신할 때 쓰는 표본 수이다.
#   - shuffle=True: epoch마다 순서를 섞어, 매번 같은 순서로 배워 생기는 치우침을 막는다.
#   - generator=torch.Generator().manual_seed(43): 섞는 순서 전용 난수 생성기를 seed 43으로 고정한다.
#     전역 seed(42)와 분리하는 이유: 모델 초기화와 데이터 섞기가 서로의 난수 소비에 영향을 주지 않게 해서 재현성을 지키기 위해서이다.
#   - num_workers=0: 별도 작업 프로세스 없이 메인 프로세스에서 batch를 만든다. 작은 자료에서 가장 단순하고 결과가 재현된다.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
# loss_fn (type: nn.CrossEntropyLoss): 2클래스 분류 손실이다. 기본 reduction="mean"이므로 batch 안의 평균 loss를 돌려준다.
loss_fn = nn.CrossEntropyLoss()
# optimizer (type: torch.optim.SGD): 모델의 모든 파라미터를 w ← w − 0.1 × grad로 갱신하는 최적화기이다. lr=0.1은 README의 고정 학습률이다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# initial (type: float): 학습 전 train 120건 전체의 평균 loss이다. evaluate가 돌려주는 튜플의 0번(loss)만 꺼낸다.
# 학습 전 값을 재 두는 이유: 학습 후 final_train_loss와 비교해 실제로 loss가 줄었는지 확인하기 위해서이다.
initial = evaluate(model, X_train, y_train, loss_fn, device)[0]

# -----------------------------------------------------------------------------
# 80 epoch 학습 루프
# -----------------------------------------------------------------------------
# history (type: list[list]): epoch마다 [epoch, train_loss_online, valid_loss, valid_accuracy]를 쌓는 기록이다.
history = []
# epoch (type: int): 1부터 80까지의 epoch 번호이다. range(1, 81)은 끝값 81을 포함하지 않으므로 정확히 80번 돈다.
for epoch in range(1, 81):
    # online_loss (type: float): 이번 epoch 학습 중 잰 train loss(batch loss×행 수 합 ÷ 120)이다.
    online_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
    # valid_loss (type: float), accuracy (type: float), prediction (type: torch.Tensor, shape: (40,)): 이번 epoch 직후 valid 평가 결과이다.
    # 루프가 끝나면 마지막(80 epoch) 값이 남아 아래 저장에 쓰인다.
    valid_loss, accuracy, prediction = evaluate(model, X_valid, y_valid, loss_fn, device)
    # 이번 epoch의 네 값을 한 행으로 기록한다.
    history.append([epoch, online_loss, valid_loss, accuracy])
    # 10 epoch마다 진행 상황을 출력한다. 80줄을 전부 찍으면 화면이 길어지므로 10번째마다 8줄만 보인다.
    if epoch % 10 == 0:
        # loss는 소수 6자리, 정확도는 소수 4자리로 반올림해 출력한다(expected/main_console.txt와 같은 형식).
        print(epoch, round(online_loss, 6), round(valid_loss, 6), round(accuracy, 4))

# -----------------------------------------------------------------------------
# 파라미터 사전 저장
# -----------------------------------------------------------------------------
# 아래는 저장 과정이다. 전체 Module 대신 직접 학습한 파라미터 사전만 저장한다.
# 배포 파일에는 미리 학습한 모델이 없다. 실행한 본인이 만든 파일만 복원한다.
# model.state_dict() (type: OrderedDict[str, torch.Tensor]): "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias" 이름별 학습된 텐서 사전이다.
# torch.save로 이 사전만 outputs/model_state.pt에 저장한다.
# 모델 객체 전체(model)가 아니라 state_dict만 저장하는 이유: 텐서만 담긴 파일은 restore.py에서 weights_only=True로 안전하게 읽을 수 있고,
# 클래스 코드(model.py)와 숫자(가중치)를 분리해 두면 같은 구조만 만들면 어디서든 복원할 수 있기 때문이다.
torch.save(model.state_dict(), out / "model_state.pt")

# -----------------------------------------------------------------------------
# 결과 파일 저장
# -----------------------------------------------------------------------------
# training_history.csv를 쓰기 모드("w")로 연다. utf-8-sig는 맨 앞에 BOM을 붙여 엑셀에서 열어도 글자가 깨지지 않게 하는 인코딩이다.
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # w (type: csv.writer): 리스트를 CSV 한 행으로 쓰는 객체이다. 머리글 행을 먼저 쓰고, history 80행을 한 번에 쓴다.
    w=csv.writer(f);w.writerow(["epoch","train_loss_online","valid_loss","valid_accuracy"]);w.writerows(history)
# metrics (type: dict): 최종 요약 지표이다.
#   - epochs·train_rows·valid_rows: 고정 조건 기록이다. test_used=False는 test 40건을 쓰지 않았음을 명시한다.
#   - initial_train_loss: 학습 전 train loss, final_train_loss: 학습이 끝난 모델로 train 120건 전체를 다시 평가한 loss이다.
#     final_train_loss는 학습 중 잰 online loss(80 epoch 값)와 달리, 가중치가 고정된 최종 모델 한 개로 잰 값이다.
#   - valid_loss·valid_accuracy: 루프 마지막(80 epoch) valid 결과이다.
metrics = {"epochs":80,"device":str(device),"train_rows":120,"valid_rows":40,"test_used":False,
           "initial_train_loss":initial,"final_train_loss":evaluate(model,X_train,y_train,loss_fn,device)[0],
           "valid_loss":valid_loss,"valid_accuracy":accuracy}
# metrics를 JSON 문자열로 바꿔 final_metrics.json에 쓴다. 
# ensure_ascii=False는 한글을 \uXXXX로 바꾸지 않고 그대로 쓰라는 뜻이고, indent=2는 보기 좋게 2칸 들여쓰기한다.
(out / "final_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
# info 사전을 "키: 값" 줄로 바꿔 gpu_check.txt에 쓴다. 실제로 어떤 환경·장치에서 학습했는지 남기는 기록이다.
(out / "gpu_check.txt").write_text("\n".join(f"{k}: {v}" for k,v in info.items()),encoding="utf-8")
# validation_predictions.csv를 쓰기 모드로 연다. restore.py가 이 파일을 다시 읽어 복원한 모델의 예측과 비교한다.
with (out / "validation_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
    # 머리글 행(id, actual, predicted)을 쓴다.
    w=csv.writer(f);w.writerow(["id","actual","predicted"])
    # valid_rows·y_valid·prediction을 같은 순서로 묶어(zip) 한 행씩 쓴다.
    # r (type: dict): valid 한 행, y (type: Tensor 0차원): 실제값, p (type: Tensor 0차원): 예측값이다. 
    # int()로 텐서를 파이썬 정수로 바꿔 저장한다.
    w.writerows((r["id"],int(y),int(p)) for r,y,p in zip(valid_rows,y_valid,prediction))
# 저장이 끝났음을 알리고, 다음에 새 프로세스로 restore.py를 실행하라고 안내한다.
print("저장 완료. 다음 순서: python restore.py")
