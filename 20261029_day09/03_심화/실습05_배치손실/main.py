"""실습 05: 배치 평균들의 단순 평균이 아닌 샘플 수 가중 평균을 계산합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산 라이브러리이다.
import torch
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소(층·손실 함수) 모음이다.
#   - 목적: 분류 손실 함수 CrossEntropyLoss를 가져오기 위해 쓴다.
from torch import nn
# [외부 라이브러리] torch.utils.data: 학습 데이터를 묶고 batch 단위로 공급하는 도구 모음이다.
#   - 목적: logits와 y를 한 쌍으로 묶고(TensorDataset), 4건씩 잘라 차례로 꺼내기(DataLoader) 위해 쓴다.
#   - 기능: TensorDataset은 같은 길이의 텐서들을 행 번호별 (x, y) 쌍으로 묶고,
#           DataLoader는 그 Dataset을 batch 단위로 for문에서 반복할 수 있게 만든다.
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# 데이터 준비: logits, 정답 y, DataLoader, 손실 함수
# -----------------------------------------------------------------------------
# logits (type: torch.Tensor, shape (10, 2), dtype float32): 샘플 10개에 대한 클래스 2개의 점수(softmax 전 값)이다.
#   [[3., 0.]] * 8은 "클래스 0에 3점, 클래스 1에 0점"인 행을 8개 만든다. → 클래스 0을 강하게 맞히는 샘플 8개이다.
#   [[0., 3.]] * 2는 "클래스 1에 3점"인 행을 2개 만든다. → 정답이 0인데 1로 강하게 틀리는 샘플 2개이다.
#   리스트 두 개를 +로 이어 붙여 10행을 만든 뒤 torch.tensor로 텐서로 바꾼다. 3. 처럼 소수점을 붙여 float32가 된다.
#   틀리는 2개를 일부러 맨 뒤에 두어 마지막 2행짜리 batch에 몰리게 만든 것이 이 실습의 핵심 장치이다.
logits = torch.tensor([[3.,0.]]*8 + [[0.,3.]]*2)
# y (type: torch.Tensor, shape (10,), dtype int64): 10개 샘플의 정답 클래스 번호로, 모두 0이다.
#   dtype=torch.long인 이유: CrossEntropyLoss는 정답을 클래스 인덱스로 받을 때 정수형(int64)을 요구한다.
y = torch.zeros(10, dtype=torch.long)
# loader (type: DataLoader): (logits, y) 쌍을 4건씩 잘라 순서대로 꺼내 주는 반복 가능한 객체이다.
#   batch_size=4: 10건을 4건, 4건, 2건으로 나눈다. drop_last의 기본값이 False라서 마지막 2건도 버리지 않는다.
#   shuffle=False: 섞지 않고 0번 행부터 순서대로 꺼낸다. 그래서 틀리는 2행이 반드시 세 번째 batch에 들어간다.
loader = DataLoader(TensorDataset(logits, y), batch_size=4, shuffle=False)
# loss_fn (type: nn.CrossEntropyLoss): 다중 클래스 분류 손실 함수이다.
#   내부에서 softmax로 확률을 만든 뒤 정답 클래스 확률에 -log를 취하고, 기본 설정(reduction='mean')으로
#   batch 안 샘플들의 손실을 "평균" 내어 하나의 숫자로 돌려준다. 이 "평균"이 바로 가중 집계가 필요한 이유이다.
loss_fn = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# batch별 손실을 집계하기
# -----------------------------------------------------------------------------
# total_loss (type: float): "batch 평균 × batch 행 수"를 계속 더해 전체 샘플 손실의 합계를 만드는 변수이다. 0.0에서 시작한다.
# count (type: int): 지금까지 처리한 샘플 수를 세는 변수이다. 0에서 시작해 마지막에 10이 된다.
total_loss, count = 0.0, 0
# batch_means (type: list): batch마다 나온 평균 손실을 그대로 모아 두는 리스트이다. 잘못된 단순 평균을 보여 주기 위해 쓴다.
batch_means = []

# loader에서 batch를 하나씩 꺼낸다. 이 데이터에서는 3번 반복한다.
for xb, yb in loader:
    # loss (type: torch.Tensor, 0차원 스칼라): 이번 batch 샘플들의 손실을 "평균"한 값이다.
    loss = loss_fn(xb, yb)
    # loss.item() (type: float): 0차원 텐서에서 파이썬 숫자 하나를 꺼낸다. 계산 그래프와 분리된 순수 숫자라서 누적해도 메모리가 쌓이지 않는다.
    # len(yb) (type: int): 이번 batch의 실제 행 수이다. 4, 4, 2 순서로 나온다.
    # 평균 × 행 수 = 이번 batch 샘플 손실의 "합계"이다. 평균으로 줄어든 값을 원래 합계로 되돌려야 batch 크기 차이가 정확히 반영된다.
    # 쉽게 말해, 반 평균 점수에 그 반의 실제 학생 수를 곱해야 반 총점이 되는 것과 같다.
    total_loss += loss.item() * len(yb)
    # 이번 batch의 실제 행 수를 처리한 샘플 수에 더한다. 반복이 끝나면 4 + 4 + 2 = 10이 된다.
    count += len(yb)
    # 비교용으로 batch 평균 손실을 가중치 없이 그대로 리스트에 추가한다.
    batch_means.append(loss.item())

# -----------------------------------------------------------------------------
# 세 가지 평균을 비교해 출력하기
# -----------------------------------------------------------------------------
# batch 구성을 안내하는 문장을 출력한다.
print("배치 크기: 4, 4, 2")
# 잘못된 단순 평균: batch 평균 3개를 더해 3으로 나눈다. 2행짜리 batch가 4행짜리와 같은 1/3 비중을 받아 틀린 값이 나온다.
print("잘못된 단순 평균:", round(sum(batch_means)/len(batch_means), 6))
# 샘플 가중 평균: 전체 샘플 손실 합계를 전체 샘플 수 10으로 나눈다.
print("샘플 가중 평균:", round(total_loss/count, 6))
# 전체를 한 번에 계산: batch로 나누지 않고 10행 전체에 손실 함수를 적용한 기준값이다.
print("전체를 한 번에 계산:", round(loss_fn(logits, y).item(), 6))
