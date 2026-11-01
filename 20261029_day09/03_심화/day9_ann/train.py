"""학습 한 epoch와 평가를 분리합니다. 빈 데이터·불량 자료는 이번 단계의 과제가 아닙니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산·자동미분 라이브러리이다.
import torch

# -----------------------------------------------------------------------------
# 한 epoch 학습 함수
# -----------------------------------------------------------------------------
# train_one_epoch: train 데이터 전체를 batch 단위로 한 바퀴 돌며 모델을 학습시키고, 그 epoch의 표본 가중 평균 loss를 돌려준다.
#   - model (type: nn.Module): 학습할 모델이다. 이 프로젝트에서는 SimpleANN(2→8→2)이다.
#   - loader (type: DataLoader): train 데이터를 batch 단위(main.py 기준 32건)로 꺼내 주는 반복자이다.
#   - loss_fn (type: nn.Module): 손실 함수이다. 이 프로젝트에서는 nn.CrossEntropyLoss이며 batch 안의 평균 loss를 돌려준다.
#   - optimizer (type: torch.optim.Optimizer): 파라미터를 갱신하는 최적화기이다. 이 프로젝트에서는 SGD(lr=0.1)이다.
#   - device (type: torch.device): 계산할 장치(cpu 또는 cuda:0)이다.
#   - 반환값 (type: float): 이 epoch의 train_loss_online(각 batch 평균 loss×행 수의 합 ÷ 전체 행 수)이다.
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # 모델을 학습 모드로 바꾼다. Dropout·BatchNorm처럼 학습과 평가에서 다르게 동작하는 층을 학습용으로 동작하게 하는 스위치이다.
    # SimpleANN에는 그런 층이 없어 계산 결과는 같지만, evaluate가 .eval()로 바꿔 둔 상태를 매 epoch 학습 모드로 되돌리기 위해 호출한다.
    model.train()
    # total (type: float): batch 평균 loss × batch 행 수를 계속 더한 누적 합이다. 0.0으로 시작해 실수 덧셈임을 분명히 한다.
    # count (type: int): 지금까지 처리한 표본(행) 수의 누적 합이다.
    total, count = 0.0, 0
    # loader에서 batch를 하나씩 꺼낸다. main.py는 shuffle=True이므로 epoch마다 섞인 순서로 나온다.
    # xb (type: torch.Tensor, shape: (batch, 2), dtype: float32): 이번 batch의 입력 x1, x2이다.
    # yb (type: torch.Tensor, shape: (batch,), dtype: int64): 이번 batch의 정답 클래스 번호(0 또는 1)이다.
    for xb, yb in loader:
        # 입력과 정답을 모델과 같은 장치로 옮긴다. 모델 가중치와 데이터가 다른 장치에 있으면 계산이 오류로 멈추기 때문이다.
        xb, yb = xb.to(device), yb.to(device)
        # 이전 batch에서 쌓인 gradient를 지운다. PyTorch는 .grad에 새 gradient를 덮어쓰지 않고 더하므로(누적),
        # 지우지 않으면 지난 batch의 gradient가 섞여 잘못된 방향으로 갱신된다.
        optimizer.zero_grad()
        # 순전파(forward)로 logits(shape: (batch, 2))를 만들고, 정답 yb와 비교해 loss를 구한다.
        # loss (type: torch.Tensor, shape: (), dtype: float32): 이번 batch의 평균 CrossEntropy 손실이며, 계산 그래프를 기억하고 있다.
        loss = loss_fn(model(xb), yb)
        # loss.backward() 후 optimizer.step()을 호출한다.
        # 매 batch마다 초기화하고 새로운 gradient로 갱신해야 한다.
        # 역전파(backward): loss에서 출발해 계산 그래프를 거꾸로 따라가며 각 파라미터의 gradient(∂loss/∂w)를 계산해 p.grad에 저장한다.
        loss.backward()
        # 갱신(step): SGD 규칙 w ← w − lr × grad로 모든 파라미터를 한 번 고친다.
        # backward 다음에 호출해야 하는 이유: step은 p.grad에 들어 있는 값을 읽어 갱신하므로, 방금 계산한 gradient가 먼저 있어야 한다.
        optimizer.step()
        # loss.item()은 batch 평균 loss를 파이썬 float로 꺼낸다. 여기에 batch 행 수 len(yb)를 곱해 "이 batch의 loss 합"으로 되돌려 누적한다.
        # 평균에 행 수를 곱하는 이유: 마지막 batch는 24건(120 = 32×3 + 24)이라 32건 batch와 같은 무게로 평균하면 24건이 과대 반영되기 때문이다.
        total += loss.item() * len(yb)
        # 이번 batch의 행 수만큼 처리 표본 수를 늘린다. 끝나면 count는 train 행 수(120)와 같다.
        count += len(yb)
    # 마지막 작은 batch도 실제 샘플 수만큼 반영한다.
    # 전체 loss 합 ÷ 전체 표본 수 = 표본 1건당 평균 loss이다. 쉽게 말해, 반마다 인원이 달라도 학생 수를 세어 전교 평균을 정확히 내는 방식이다.
    # 매 batch마다 가중치가 바뀌는 중에 잰 값이므로, 학습이 끝난 뒤 train 전체를 한 번에 평가한 final_train_loss와는 다르다.
    return total / count

# -----------------------------------------------------------------------------
# 평가 함수
# -----------------------------------------------------------------------------
# evaluate: 가중치를 바꾸지 않고 주어진 데이터 전체의 loss, 정확도, 예측 클래스를 계산한다.
#   - model (type: nn.Module): 평가할 모델이다.
#   - X (type: torch.Tensor, shape: (N, 2), dtype: float32): 평가 입력 전체이다(train 120건 또는 valid 40건).
#   - y (type: torch.Tensor, shape: (N,), dtype: int64): 평가 정답 클래스 번호 전체이다.
#   - loss_fn (type: nn.Module): 손실 함수(CrossEntropyLoss)이다.
#   - device (type: torch.device): 계산할 장치이다.
#   - 반환값 (type: tuple[float, float, torch.Tensor]): (평균 loss, 정확도, CPU에 있는 예측 클래스 텐서)이다.
def evaluate(model, X, y, loss_fn, device):
    # 모델을 평가 모드로 바꾼다. 이것은 층의 "동작 방식"을 평가용으로 바꾸는 스위치이고, gradient 기록을 끄는 기능은 없다.
    model.eval()
    # no_grad 블록 안에서는 계산 그래프를 만들지 않는다. 그래프가 없으니 메모리를 아끼고, 결과 텐서로 backward할 수도 없다.
    # .eval()(층 동작 전환)과 no_grad()(그래프 기록 끄기)는 서로 다른 일을 하므로 평가에서는 둘 다 쓴다.
    with torch.no_grad():
        # logits (type: torch.Tensor, shape: (N, 2), dtype: float32): 각 행의 클래스 0·1 점수(softmax 전의 값)이다.
        # 입력 X를 모델이 있는 장치로 옮긴 뒤 한 번에 전부 순전파한다. 평가는 가중치를 바꾸지 않으므로 batch로 나눌 필요가 없다.
        logits = model(X.to(device))
        # loss (type: float): 전체 N건의 평균 CrossEntropy 손실이다. 정답도 같은 장치로 옮긴 뒤 계산하고 .item()으로 파이썬 float로 꺼낸다.
        loss = loss_fn(logits, y.to(device)).item()
        # logits에서 클래스 인덱스를 구하고 CPU로 옮겨 prediction에 저장한다.
        # 평가에서는 backward와 optimizer.step을 호출하지 않는다.
        # prediction (type: torch.Tensor, shape: (N,), dtype: int64, device: cpu): 각 행에서 점수가 가장 큰 클래스 번호이다.
        # argmax(dim=1)인 이유: logits의 0번 축은 행(표본), 1번 축은 클래스이므로 한 행 안에서 클래스끼리 비교해야 하기 때문이다.
        # .cpu()인 이유: 아래 줄에서 CPU에 있는 y.cpu()와 비교하고, main.py가 CSV로 저장하므로 장치를 CPU로 맞춰야 하기 때문이다.
        # softmax를 거치지 않는 이유: softmax는 값의 크기 순서를 바꾸지 않으므로 logits의 최댓값 위치와 확률의 최댓값 위치가 같다.
        prediction = logits.argmax(dim=1).cpu()
    # (prediction == y.cpu())는 맞힌 행이 True인 bool 텐서이고, .float()로 1.0/0.0으로 바꾼 뒤 .mean()으로 맞힌 비율(정확도)을 구한다.
    # 반환: loss (type: float), 정확도 (type: float), prediction (type: torch.Tensor)이다.
    return loss, (prediction == y.cpu()).float().mean().item(), prediction
