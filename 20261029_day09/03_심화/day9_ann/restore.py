"""실습 08: main.py가 만든 state_dict만 로드합니다. 새 프로세스 CPU에서 복원합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [표준 라이브러리] csv: CSV 파일을 읽고 쓰는 모듈이다.
import csv
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산·자동미분 라이브러리이다.
import torch

# [프로젝트 내부 모듈] model.SimpleANN: 같은 폴더의 model.py에 있는 2→8→2 신경망 클래스이다.
#   - 목적: 학습할 모델 구조를 만들기 위해 쓴다. restore.py도 같은 클래스를 써서 구조가 똑같이 맞춰진다.
#   - 기능: 입력 2개 → 은닉 8개(ReLU) → 출력 2개(logits)를 계산한다.
from model import SimpleANN

# -----------------------------------------------------------------------------
# valid 입력 읽기
# -----------------------------------------------------------------------------
# base (type: Path): 이 Python 파일이 들어 있는 폴더(day9_ann)의 절대 경로이다.
base = Path(__file__).resolve().parent
# data/points.csv를 연다. f (type: TextIOWrapper): with 블록이 끝나면 자동으로 닫히는 파일 객체이다.
with (base / "data" / "points.csv").open(encoding="utf-8",newline="") as f:
    # valid_rows (type: list[dict[str, str]]): split이 "valid"인 40행이다. main.py와 같은 조건·같은 파일 순서로 고르므로 행 순서가 같다.
    valid_rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]
# X (type: torch.Tensor, shape: (40, 2), dtype: float32): valid 입력 x1, x2이다. 정답(label)은 비교에 쓰지 않으므로 만들지 않는다.
X = torch.tensor([[float(r["x1"]),float(r["x2"])] for r in valid_rows])

# -----------------------------------------------------------------------------
# 같은 구조의 모델에 파라미터 사전 복원
# -----------------------------------------------------------------------------
# model (type: SimpleANN): 학습 전 상태의 새 모델이다. 지금은 무작위 초기 가중치를 갖고 있으며 CPU에 있다.
model = SimpleANN()
# 직접 저장한 model_state.pt를 map_location="cpu", weights_only=True로 읽고 모델에 넣는다.
# 먼저 같은 모델 구조를 만들었다. 인터넷에서 받은 출처 불명 파일은 읽지 않는다.
# >>> H10
# state (type: dict[str, torch.Tensor]): main.py가 저장한 파라미터 사전("net.0.weight" 등 4개 텐서)이다.
#   - map_location="cpu": 저장할 때 텐서가 cuda:0에 있었더라도 CPU로 옮겨 읽는다. GPU가 없는 컴퓨터에서도 복원되게 하기 위해서이다.
#   - weights_only=True: 텐서·기본 자료형만 읽고 임의의 파이썬 객체는 만들지 않는다.
#     pickle 형식 파일은 읽는 순간 숨겨진 코드를 실행할 수 있으므로, 이 옵션으로 그 위험을 막는다.
state = torch.load(base / "outputs" / "model_state.pt", map_location="cpu", weights_only=True)
# 읽은 파라미터를 모델에 넣는다. 기본값 strict=True이므로 키 이름이나 텐서 모양이 하나라도 다르면 오류를 내서, 구조가 다른 파일을 잘못 넣는 일을 막는다.
model.load_state_dict(state)

# -----------------------------------------------------------------------------
# 복원 모델로 다시 예측하고 저장값과 비교
# -----------------------------------------------------------------------------
# 평가 모드로 바꾼다. 학습과 평가에서 다르게 동작하는 층을 평가용으로 맞추는 스위치이다.
model.eval()
# gradient 기록 없이 예측한다. 예측만 하므로 계산 그래프가 필요 없다.
with torch.no_grad():
    # predictions (type: list[int]): valid 40건 각각의 예측 클래스 번호이다.
    # model(X)로 logits(shape: (40, 2))를 구하고, argmax(dim=1)로 행마다 점수가 큰 클래스를 고르고, tolist()로 파이썬 리스트로 바꾼다.
    predictions = model(X).argmax(dim=1).tolist()
# main.py가 저장한 validation_predictions.csv를 연다. 저장할 때 utf-8-sig로 썼으므로 BOM을 떼고 읽도록 같은 인코딩을 쓴다.
with (base / "outputs" / "validation_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
    # saved (type: list[dict[str, str]]): 저장된 40행({"id", "actual", "predicted"})이다.
    saved = list(csv.DictReader(f))
# 지금 읽은 valid ID 순서와 저장 파일의 ID 순서가 같은지 출력한다(True여야 한다). 순서가 다르면 아래 예측 비교가 의미 없어진다.
print("ID 순서 동일:", [r["id"] for r in valid_rows] == [r["id"] for r in saved])
# 복원 모델의 예측과 저장된 예측이 40건 모두 같은지 출력한다. 저장 값은 문자열이므로 int()로 바꿔 비교한다.
print("예측 동일:", predictions == [int(r["predicted"]) for r in saved])
# 이 검사의 한계를 밝힌다: 예측 클래스가 같은지만 본 것이고, 다른 CPU/GPU에서 확률의 마지막 비트까지 같다는 보장은 아니다.
print("서로 다른 CPU/GPU의 마지막 비트까지 동일함을 보장하는 검사는 아닙니다.")
