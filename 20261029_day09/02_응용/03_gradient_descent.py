"""실습 03: 매번 새로운 그래프를 만들고 gradient를 비웁니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [표준 라이브러리] csv: CSV 파일을 읽고 쓰는 모듈이다.
import csv
# 외부 라이브러리. 목적: 딥러닝용 Tensor를 다루기 위해 불러온다.
import torch

# -----------------------------------------------------------------------------
# 학습할 파라미터와 설정 준비
# -----------------------------------------------------------------------------
# w (type: torch.Tensor, 0차원 float32 스칼라): 경사하강법으로 찾아갈 파라미터이다. 시작값은 0.0이다.
#   requires_grad=True: w로 만든 계산을 그래프로 기록해 두었다가 backward() 때 w에 대한 미분값을 w.grad에 채우라는 뜻이다.
#   0.0처럼 소수로 써야 float 텐서가 된다. 정수 텐서는 requires_grad=True를 쓸 수 없다.
w = torch.tensor(0.0, requires_grad=True)
# lr (type: float): 학습률(learning rate)이다. 기울기에 곱해 한 번에 얼마나 움직일지 정하는 보폭이다.
#   이 문제에서 lr = 0.1이면 매 step마다 w와 목표값 4 사이의 거리가 정확히 0.8배로 줄어든다(아래 TODO 설명 참고).
lr = 0.1

# -----------------------------------------------------------------------------
# 경사하강법 반복 (10 step)
# -----------------------------------------------------------------------------
# history (type: list): step마다 [step, 갱신 후 w, 갱신 후 loss] 한 줄(list)씩 쌓는 기록 목록이다.
history = []
for step in range(1, 11):
    # loss (type: torch.Tensor, 0차원): 손실 함수 (w - 4)^2 의 현재 값이다. w = 4일 때 0으로 가장 작다.
    #   반복문 안에서 매번 새로 계산하므로 반복마다 "현재 w"에 대한 새 계산 그래프가 만들어진다.
    #   backward()가 끝나면 그 그래프는 버려지므로, 다음 step에서 반드시 loss를 다시 계산해야 다시 backward()를 할 수 있다.
    loss = (w - 4.0) ** 2
    # backward(): 그래프를 거꾸로 따라가며 d(loss)/dw = 2(w - 4) 를 계산해 w.grad에 "더한다"(덮어쓰지 않는다).
    #   그래서 아래에서 매 step 끝에 w.grad.zero_()로 비워야 이번 step의 기울기만 남는다.
    loss.backward()
    # torch.no_grad(): 이 블록 안의 연산은 계산 그래프에 기록하지 않는다.
    #   w 갱신은 학습 규칙이지 모델 계산이 아니므로 미분 대상이 되면 안 된다.
    #   또한 requires_grad=True인 잎(leaf) 텐서 w를 그래프 기록 중에 제자리(in-place) 수정하면 RuntimeError가 나므로 이 블록 안에서 바꾼다.
    with torch.no_grad():
        # 경사하강법 갱신 규칙: w ← w − lr × (d loss / d w).
        #   기울기의 반대 방향으로 움직여야 loss가 줄어든다. 시작점 w = 0에서 기울기는 2(0 - 4) = -8(음수)이므로 w는 커진다.
        #   식을 풀면 w_new = w − 0.1 × 2(w − 4) = 0.8w + 0.8 이고, 4 − w_new = 0.8 × (4 − w) 이다. 즉 4까지 남은 거리가 매 step 0.8배가 된다.
        # w -= ...: 제자리(in-place) 연산이다. 새 텐서를 만들지 않고 w라는 같은 객체의 값만 바꾼다.
        #   w = w - lr * w.grad 처럼 쓰면 이름 w가 새 텐서를 가리키게 되고, 그 새 텐서는 no_grad 안에서 만들어져 requires_grad=False가 된다.
        #   그러면 다음 step에서 loss.backward()가 w.grad를 채우지 못해 에러가 나므로, 요구사항대로 반드시 제자리 갱신을 써야 한다.
        # lr * w.grad (type: torch.Tensor, 0차원): 이번 step에 움직일 양이다. step 1에서는 0.1 × (-8) = -0.8이므로 w는 0.8이 된다.
        w -= lr * w.grad
    # w.grad.zero_(): w.grad에 쌓인 기울기를 0으로 비운다. 이름 끝의 _ 는 제자리 연산이라는 PyTorch 표기이다.
    #   비우지 않으면 다음 step의 backward()가 새 기울기를 이전 기울기에 더해 버려, 실제보다 큰 잘못된 기울기로 갱신하게 된다.
    #   w.grad 자체는 그래프를 추적하지 않는 텐서이므로 no_grad 블록 밖에서 호출해도 문제가 없다.
    w.grad.zero_()
    # 아래 loss는 갱신한 w에서 다시 계산한 값이다.
    # w.item() (type: float): 갱신 후 w 값을 파이썬 숫자로 꺼낸다. 텐서 그대로 넣으면 w가 바뀔 때 기록도 같이 바뀌어 보이므로 숫자로 저장한다.
    # w.detach() (type: torch.Tensor): w와 값은 같지만 그래프 추적이 끊긴 텐서이다.
    #   기록용 계산이 계산 그래프에 섞이지 않게 하려고 detach()한 뒤 (w - 4)^2 를 계산한다. .square()는 제곱이다.
    # (w.detach() - 4.0).square().item() (type: float): 갱신 후 w에서 다시 잰 loss 값이다. 위 loss는 갱신 전 w의 값이므로 서로 다르다.
    history.append([step, w.item(), (w.detach() - 4.0).square().item()])

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# step (type: int), value (type: float), loss_after (type: float): history의 한 행을 세 변수로 풀어 받는다.
for step, value, loss_after in history:
    # round(…, 6): 소수 6자리까지 반올림해 출력한다.
    #   논리적으로 기대되는 출력은 w = 4 × (1 − 0.8^step) 이다. 예: step 1 → 0.8 / 10.24, step 2 → 1.44 / 6.5536, step 10 → 약 3.570503 / 0.184467.
    #   w가 4에 가까워지고 loss가 계속 줄어들면 갱신 방향과 크기가 올바르다는 뜻이다.
    print(step, round(value, 6), round(loss_after, 6))

# -----------------------------------------------------------------------------
# 결과 저장
# -----------------------------------------------------------------------------
# out (type: pathlib.Path): 이 스크립트 파일이 있는 폴더 아래의 outputs 폴더 경로이다.
#   Path(__file__).resolve().parent를 쓰는 이유: 어느 폴더에서 실행하든 항상 스크립트 옆 outputs에 저장되게 하기 위해서이다.
out = Path(__file__).resolve().parent / "outputs"
# mkdir(exist_ok=True): outputs 폴더를 만든다. 이미 있으면 에러 없이 그대로 둔다.
out.mkdir(exist_ok=True)
# f (type: io.TextIOWrapper): gd_history.csv를 쓰기("w") 모드로 연 파일 객체이다.
#   encoding="utf-8-sig": 파일 앞에 BOM을 붙여 엑셀에서 열어도 한글이 깨지지 않게 한다.
#   newline="": csv 모듈이 줄바꿈을 직접 처리하므로, 윈도우에서 빈 줄이 끼어드는 것을 막기 위해 지정한다.
with (out / "gd_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    # writer (type: _csv.writer): 파일 f에 CSV 행을 쓰는 객체이다.
    # writerow: 머리글 한 행(step, w_after, loss_after)을 쓴다. writerows: history의 10개 행을 한꺼번에 쓴다.
    # 세미콜론(;)은 한 줄에 여러 문장을 이어 쓰는 파이썬 문법이다. 세 문장이 차례대로 실행된다.
    writer = csv.writer(f); writer.writerow(["step", "w_after", "loss_after"]); writer.writerows(history)
