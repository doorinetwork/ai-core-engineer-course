"""모델 구조만 정의합니다. 이 파일을 실행해도 학습하지 않습니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch.nn: PyTorch의 신경망 구성 모듈이다.
from torch import nn

# -----------------------------------------------------------------------------
# 모델 정의
# -----------------------------------------------------------------------------
# SimpleANN: 입력 2개(x1, x2) → 은닉 8개 → 출력 2개(클래스 0·1의 logits)인 인공신경망이다.
# nn.Module을 상속하는 이유: 그래야 안에 넣은 층의 가중치가 자동으로 등록되어 parameters()·state_dict()·.to(device)가 동작한다.
class SimpleANN(nn.Module):
    # __init__: 모델을 만들 때 한 번 실행되어 층을 준비한다.
    def __init__(self):
        # 부모 클래스(nn.Module)의 초기화를 먼저 실행한다. 이것을 빼면 층을 속성으로 등록하는 내부 장치가 없어 오류가 난다.
        super().__init__()
        # self.net (type: nn.Sequential): 세 층을 순서대로 묶은 본체이다. state_dict 키가 "net.0.weight"처럼 이 이름과 순번으로 정해진다.
        #   - nn.Linear(2,8): 입력 2개를 은닉 뉴런 8개로 바꾼다. 가중치 shape (8, 2), 편향 shape (8,)이다.
        #   - nn.ReLU(): 은닉값의 음수를 0으로 만든다. 이 비선형 단계가 없으면 두 Linear가 하나의 직선 변환으로 합쳐져 곡선 경계를 배우지 못한다.
        #   - nn.Linear(8,2): 은닉 8개를 클래스 2개의 점수(logits)로 바꾼다. 가중치 shape (2, 8), 편향 shape (2,)이다.
        # 마지막에 softmax를 넣지 않는 이유: CrossEntropyLoss가 내부에서 softmax를 계산하므로, 모델은 logits를 그대로 내보내야 한다.
        self.net = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))

    # forward: 입력이 들어왔을 때 출력을 계산하는 규칙이다. model(x)처럼 호출하면 PyTorch가 이 메서드를 실행한다.
    #   - x (type: torch.Tensor, shape: (N, 2), dtype: float32): N개 표본의 입력이다.
    #   - 반환값 (type: torch.Tensor, shape: (N, 2), dtype: float32): 각 표본의 클래스 0·1 logits이다.
    def forward(self, x):
        # 입력을 Linear → ReLU → Linear 순서로 통과시킨 결과를 돌려준다.
        return self.net(x)
