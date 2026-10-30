# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# 외부 라이브러리. Python 시각화 라이브러리.
import matplotlib
# 창을 띄우지 않고 PNG에 저장한다.
matplotlib.use("Agg")
# 외부 라이브러리. 목적: loss 곡선 그림 그리기. 기능: plot, axvline, savefig 같은 그리기 함수를 제공한다.
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 학습곡선 데이터 준비
# -----------------------------------------------------------------------------
# 아래 20개 값은 학습곡선 읽기용으로 제공한 설명용 값이다.
# 실제 모델 실행 결과라고 보고하지 않는다. 07번은 직접 학습한 곡선을 그린다.
# train_losses (type: list of float): epoch 1~20의 train loss 20개이다.
# 0.90에서 0.09까지 계속 줄어든다. 모델이 train 데이터를 점점 더 잘 맞힌다는 뜻이다.
train_losses = [0.90,0.76,0.65,0.56,0.49,0.43,0.38,0.33,0.29,0.25,
                0.22,0.20,0.18,0.16,0.14,0.13,0.12,0.11,0.10,0.09]
# valid_losses (type: list of float): epoch 1~20의 valid loss 20개이다.
# 0.94에서 0.32(11번째 값)까지 줄다가 그 뒤로 다시 0.52까지 커진다.
# train loss는 계속 줄어드는데 valid loss가 다시 커지는 구간이 과적합(overfitting)이 시작된 구간이다.
valid_losses = [0.94,0.81,0.71,0.61,0.53,0.47,0.42,0.38,0.35,0.33,
                0.32,0.33,0.35,0.37,0.40,0.43,0.46,0.48,0.50,0.52]

# -----------------------------------------------------------------------------
# best epoch 찾기
# -----------------------------------------------------------------------------
# valid_losses에서 가장 작은 값의 위치를 구하고 1을 더해 best_epoch에 넣는다.
# 리스트 위치는 0부터, 여기서 출력할 epoch는 1부터 시작한다.
# best_epoch (type: int): valid loss가 가장 낮았던 epoch 번호(1부터 셈)이다. 이 데이터에서는 11이다.
#   min(valid_losses)는 가장 작은 값 0.32를 돌려주고, .index(0.32)는 그 값이 처음 나오는 위치 10을 돌려준다.
#   리스트 위치는 0부터 세지만 epoch는 1부터 세므로 1을 더해야 그림의 x축(range(1, 21))과 번호가 맞는다.
#   1을 더하지 않으면 실제 최솟값보다 한 칸 왼쪽(epoch 10)에 점선이 그려진다.
#   best epoch를 train loss가 아니라 valid loss로 고르는 이유는, 
#   train loss는 외울수록 계속 줄어들어 처음 보는 데이터에 대한 실력을 알려 주지 못하고, valid loss만이 그 실력을 보여 주기 때문이다.
best_epoch = valid_losses.index(min(valid_losses)) + 1
# best epoch 번호와 그때의 최소 valid loss를 출력해 결과를 눈으로 확인한다.
print("best epoch:", best_epoch, "최소 valid loss:", min(valid_losses))

# -----------------------------------------------------------------------------
# 학습곡선 그리기 및 저장
# -----------------------------------------------------------------------------
# 가로 8인치, 세로 5인치 크기의 새 이미지를 만든다.
# figsize는 (가로, 세로) 인치 단위이며, epoch 20개가 옆으로 넉넉히 펼쳐지도록 가로를 더 길게 잡는다.
plt.figure(figsize=(8, 5))
# x축을 epoch 1~20, y축을 train loss로 하는 곡선을 그린다.
#   range(1, 21)은 1부터 20까지(21은 포함하지 않음)라서 loss 값 20개와 개수가 정확히 맞는다.
#   label="train"은 범례에 표시할 이름이다.
plt.plot(range(1, 21), train_losses, label="train")
# 같은 x축(epoch 1~20)에 valid loss 곡선을 겹쳐 그린다.
# 두 선을 한 그림에 겹쳐야 두 곡선이 벌어지기 시작하는 지점(과적합 시작)을 한눈에 볼 수 있다.
plt.plot(range(1, 21), valid_losses, label="valid")
# best_epoch 위치에 세로 점선을 그어 가장 좋은 epoch를 표시한다.
#   linestyle="--"는 점선이라는 뜻으로, 데이터 곡선과 구분되는 기준선임을 보여 준다.
plt.axvline(best_epoch, linestyle="--", label="best epoch")
# x축 이름을 "Epoch"로 붙인다. 데이터 전체를 한 번 다 학습한 횟수를 뜻한다.
plt.xlabel("Epoch")
# y축 이름을 "Cross entropy loss"로 붙인다. 분류 모델의 오차 크기이며 작을수록 좋다.
plt.ylabel("Cross entropy loss")
# 각 선의 label("train", "valid", "best epoch")을 모아 범례 상자를 표시한다.
plt.legend()
# 축 이름이나 범례가 그림 밖으로 잘리지 않도록 여백을 자동으로 조정한다.
plt.tight_layout()
# 그림을 이 .py 파일과 같은 폴더에 practice04_curves.png로 저장한다.
#   Path(__file__).resolve().parent는 이 파일이 든 폴더의 절대 경로라서, 어디서 실행하든 같은 위치에 저장된다.
#   dpi=150은 1인치당 점 150개라는 뜻으로, 8x5인치 그림이 1200x750 픽셀의 선명한 이미지가 된다.
plt.savefig(Path(__file__).resolve().parent / "practice04_curves.png", dpi=150)
# 다 그린 이미지를 닫아 메모리를 돌려준다.
# 닫지 않으면 이미지를 여러 장 그릴 때 계속 쌓여 메모리를 차지한다.
plt.close()
# 이미지가 만들어졌음을 알리는 메시지를 출력한다.
print("생성: practice04_curves.png")
