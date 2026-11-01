"""실습 05: 마지막 작은 batch를 버리지 않고 자료를 모두 읽습니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [표준 라이브러리] csv: CSV 파일을 읽고 쓰는 모듈이다.
import csv
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 요소 모음이다.
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
# 데이터 준비: 입력 X, 정답 y, Dataset
# -----------------------------------------------------------------------------
# X (type: torch.Tensor, shape (10, 2), dtype float32): 샘플 10개, 특성 2개인 입력이다.
#   torch.arange(20)은 0~19 정수 20개를 만들고, reshape(10, 2)로 10행 2열로 바꾼 뒤 10으로 나눈다.
#   그래서 값은 [[0.0, 0.1], [0.2, 0.3], ..., [1.8, 1.9]]이다.
#   dtype=torch.float32로 지정하는 이유: 신경망의 가중치가 float32이므로 입력도 같은 실수형이어야 계산이 맞는다.
X = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 10
# y (type: torch.Tensor, shape (10,), dtype int64): 각 샘플의 정답 클래스 번호이다.
#   앞 5개는 클래스 0, 뒤 5개는 클래스 1이다.
#   dtype=torch.long인 이유: CrossEntropyLoss는 정답을 클래스 인덱스로 받을 때 정수형(int64)을 요구한다.
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
# dataset (type: TensorDataset): X와 y를 같은 행 번호끼리 짝지은 데이터 묶음이다.
#   dataset[i]는 (X[i], y[i]) 튜플을 돌려주고, len(dataset)은 10이다.
#   X와 y의 첫 번째 차원 길이가 같아야 묶을 수 있다(둘 다 10).
dataset = TensorDataset(X, y)

# -----------------------------------------------------------------------------
# DataLoader 만들기
# -----------------------------------------------------------------------------
# loader (type: DataLoader): dataset을 4건씩 잘라 순서대로 꺼내 주는 반복 가능한 객체이다.
#   batch_size=4: 한 번에 4건씩 묶는다. 원본의 5로 두면 5·5로 나뉘어 요구한 4·4·2가 나오지 않는다.
#   shuffle=False: 섞지 않고 0번 샘플부터 순서대로 꺼낸다. 순서가 고정되어야 batch 구성을 손으로 확인할 수 있다.
#   drop_last=False: 10을 4로 나누면 나머지 2건이 남는데, 이 마지막 작은 batch를 버리지 않고 그대로 공급한다.
#     True로 두면 마지막 2건이 학습·평가에서 빠져 합계가 8건이 된다. 그래서 반드시 False로 둔다.
#     (기본값도 False이지만 요구사항이 "버리지 않는다"이므로 의도를 드러내기 위해 명시한다.)
#   num_workers=0: 별도 작업 프로세스 없이 메인 프로세스에서 데이터를 읽는다.
#     데이터가 10건뿐이라 병렬 읽기가 필요 없고, Windows·노트북 환경에서 멀티프로세스 오류를 피할 수 있다.
loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

# -----------------------------------------------------------------------------
# batch를 하나씩 꺼내 크기 확인하기
# -----------------------------------------------------------------------------
# sizes (type: list): [batch 번호, 그 batch의 표본 수] 쌍을 모으는 빈 리스트이다. 
sizes = []
# enumerate(loader, 1): batch를 꺼낼 때마다 1부터 시작하는 번호를 함께 붙인다(0이 아니라 1부터 세기 위해 1을 준다).
# index (type: int): batch 번호(1, 2, 3)이다.
# xb (type: torch.Tensor, shape (batch 크기, 2)): 이번 batch의 입력이다.
# yb (type: torch.Tensor, shape (batch 크기,)): 이번 batch의 정답이다.
#   DataLoader가 (X[i], y[i]) 쌍 여러 개를 첫 번째 차원으로 쌓아 xb, yb 두 텐서로 돌려준다.
for index, (xb, yb) in enumerate(loader, 1):
    # len(yb) (type: int): 이번 batch의 실제 표본 수이다(4, 4, 2). 
    #                      batch_size(4)가 아니라 실제 개수를 기록해야 마지막 batch가 2건이라는 사실이 정확히 남는다.
    sizes.append([index, len(yb)])
    # batch 번호와 xb, yb의 shape을 출력한다. tuple()로 바꿔 torch.Size 대신 보기 쉬운 튜플로 찍는다.
    # 예상 출력:
    #   1 (4, 2) (4,)
    #   2 (4, 2) (4,)
    #   3 (2, 2) (2,)
    print(index, tuple(xb.shape), tuple(yb.shape))
# sizes에서 표본 수(n)만 꺼내 모두 더한다. _는 쓰지 않는 batch 번호 자리이다.
# 4 + 4 + 2 = 10이므로 원본 10건을 빠짐없이 읽었음을 확인한다. 예상 출력: 합계: 10
print("합계:", sum(n for _, n in sizes))

# -----------------------------------------------------------------------------
# 예시 교차 엔트로피 손실 계산
# -----------------------------------------------------------------------------
# logits (type: torch.Tensor, shape (2, 2), dtype float32): 샘플 2개의 클래스별 원점수이다.
#   1번째 샘플은 클래스 0 점수가, 2번째 샘플은 클래스 1 점수가 2.0으로 더 높다.
logits = torch.tensor([[2., 0.], [0., 2.]])
# nn.CrossEntropyLoss()(logits, 정답): 내부에서 log_softmax를 계산하고 정답 위치 값의 음수를 평균 낸다.
#   정답 torch.tensor([0, 1])은 1번째 샘플이 클래스 0, 2번째 샘플이 클래스 1이라는 뜻이다.
# .item() (type: float)으로 파이썬 숫자를 꺼내 소수점 6자리로 반올림한다.
print("예시 CE:", round(nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).item(), 6))

# -----------------------------------------------------------------------------
# batch 크기 기록을 CSV로 저장하기
# -----------------------------------------------------------------------------
# out (type: Path): 이 Python 파일이 있는 폴더 아래의 outputs 폴더 경로이다.
out = Path(__file__).resolve().parent / "outputs"
# outputs 폴더를 만든다. exist_ok=True는 폴더가 이미 있어도 오류를 내지 않고 넘어가게 한다.
out.mkdir(exist_ok=True)
# batch_sizes.csv를 쓰기 모드("w")로 연다.
#   encoding="utf-8-sig": 파일 맨 앞에 BOM을 넣어 엑셀에서 열어도 한글이 깨지지 않게 한다.
#   newline="": csv 모듈이 줄바꿈을 직접 처리하므로, 윈도우에서 빈 줄이 끼어드는 것을 막는다.
# f (type: TextIOWrapper): 열린 파일 객체이다. with 블록이 끝나면 자동으로 닫힌다.
with (out / "batch_sizes.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: csv.writer): f에 CSV 행을 기록하는 객체이다.
    # 첫 줄에 헤더 ["batch", "samples"]를 쓰고, 이어서 sizes의 각 [번호, 표본 수]를 한 줄씩 쓴다.
    # 저장 내용: batch,samples / 1,4 / 2,4 / 3,2
    writer = csv.writer(f); writer.writerow(["batch", "samples"]); writer.writerows(sizes)

# -----------------------------------------------------------------------------
# 배치 크기가 다를 때 평균 loss를 합치는 방법
# -----------------------------------------------------------------------------
# loss_means (type: list[float]): batch별 평균 손실이다. CrossEntropyLoss의 기본 reduction="mean"이
#   batch 안에서 이미 평균을 낸 값이라고 보면 된다.
loss_means = [0.1, 0.2, 0.9]
# sample_counts (type: list[int]): 각 batch의 실제 표본 수이다. 위 DataLoader 결과(4, 4, 2)와 같다.
sample_counts = [4, 4, 2]

# zip(loss_means, sample_counts): 같은 위치의 (평균 손실, 표본 수)를 짝지어 (0.1, 4), (0.2, 4), (0.9, 2)를 꺼낸다.
# loss * count: batch 평균에 표본 수를 곱해 그 batch의 "손실 총합"으로 되돌린다.
#   0.1 × 4 = 0.4, 0.2 × 4 = 0.8, 0.9 × 2 = 1.8 이고, 합은 3.0 이다.
# sum(sample_counts) (type: int): 전체 표본 수 4 + 4 + 2 = 10 이다.
#   마지막 batch를 batch_size인 4로 세지 않고 실제 개수 2로 세야 분모가 12가 아닌 10으로 정확해진다.
# weighted_loss (type: float): 전체 10건 기준 샘플 하나당 평균 손실이다. 3.0 / 10 = 0.3 이다.
# 단순 평균 (0.1 + 0.2 + 0.9) / 3 = 0.4 가 틀린 이유: 2건뿐인 마지막 batch를 4건짜리 batch와 똑같은 비중(1/3)으로 세어
#   손실이 큰 마지막 batch(0.9)의 영향이 실제(2/10 = 1/5)보다 부풀려진다. 그래서 0.3보다 큰 0.4가 나온다.
# 쉽게 말해, 반 평균을 낼 때 학생 4명 반과 2명 반의 평균을 그냥 더해 나누면 안 되고,
#   반마다 학생 수만큼 점수를 곱해 전체 학생 수로 나눠야 진짜 전체 평균이 된다는 뜻이다.
weighted_loss = sum(loss * count for loss, count in zip(loss_means, sample_counts)) / sum(sample_counts)
# 샘플 가중 평균을 소수점 6자리로 반올림해 출력한다.
# 예상 출력: 샘플 가중 loss: 0.3
print("샘플 가중 loss:", round(weighted_loss, 6))
# 비교용으로 batch 평균 세 개를 단순 평균한 잘못된 값을 출력한다. 예상 출력: 배치 단순 평균: 0.4
print("배치 단순 평균:", round(sum(loss_means) / len(loss_means), 6))
