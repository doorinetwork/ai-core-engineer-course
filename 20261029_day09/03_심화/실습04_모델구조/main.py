"""실습 04: 같은 파라미터의 Module과 Sequential을 비교합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 외부 라이브러리. 목적: 딥러닝용 Tensor를 다루기 위해 불러온다.
import torch
# 외부 라이브러리(torch의 하위 모듈). 목적: 신경망 층을 만들기 위해 불러온다. 
# 기능: nn.Linear 같은 층(layer) 클래스와 손실 함수 등을 제공한다.
from torch import nn

# -----------------------------------------------------------------------------
# nn.Module을 상속한 사용자 정의 모델 만들기
# -----------------------------------------------------------------------------
# SimpleANN (type: class, 부모: nn.Module): 입력 2 → 은닉 8 → 출력 2 구조의 인공신경망 클래스이다.
#   nn.Module을 상속하면 self에 붙인 층의 파라미터가 자동으로 등록되어 parameters(), state_dict(), .to(device) 등을 그대로 쓸 수 있다.
class SimpleANN(nn.Module):
    # __init__: 모델 객체를 만들 때 한 번 실행되어 "어떤 층을 가질지"를 정한다. 계산은 여기서 하지 않는다.
    def __init__(self):
        # super().__init__(): 부모 nn.Module의 초기화를 먼저 실행한다.
        #   이것을 빼면 파라미터를 등록할 내부 저장소가 만들어지지 않아, 아래에서 self.net을 붙이는 순간 오류가 난다.
        super().__init__()
        # self.net (type: nn.Sequential): 모델 내부의 층 묶음이다. 인덱스 0, 1, 2 순서로 저장된다.
        #   - 0: nn.Linear(2, 8) (type: nn.Linear): 특성 2개를 은닉 뉴런 8개로 바꾼다. weight (8, 2) + bias (8,) = 24개 파라미터이다.
        #   - 1: nn.ReLU() (type: nn.ReLU): 음수를 0으로 바꿔 비선형성을 준다. 파라미터는 0개이다.
        #   - 2: nn.Linear(8, 2) (type: nn.Linear): 은닉 8개를 클래스 점수 2개로 바꾼다. weight (2, 8) + bias (2,) = 18개 파라미터이다.
        #   nn.Linear는 생성되는 순간 가중치를 무작위로 초기화하므로, 이 줄이 실행될 때 난수가 소비된다.
        self.net = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))

    # forward: model(x)를 호출하면 PyTorch가 내부에서 이 메서드를 실행한다. "입력이 어떤 순서로 계산되는지"를 정한다.
    def forward(self, x):
        # x를 self.net에 넣어 Linear(2,8) → ReLU → Linear(8,2)를 차례로 통과시킨 결과를 돌려준다.
        #   shape 변화: (N, 2) → Linear → (N, 8) → ReLU → (N, 8) → Linear → (N, 2)이다.
        #   self.net.forward(x)가 아니라 self.net(x)로 호출하는 이유: 모듈 호출(__call__)을 거쳐야 PyTorch의 hook 처리 등 내부 절차가 정상적으로 실행되기 때문이다.
        #   softmax를 넣지 않는 이유: 분류 학습에 쓰는 CrossEntropyLoss가 내부에서 softmax를 계산하므로 모델은 softmax 전 점수(logits)를 내야 한다.
        #   또한 비교 대상인 sequential에도 softmax가 없으므로, 여기에 softmax를 넣으면 두 출력이 달라져 비교가 무의미해진다.
        return self.net(x)

# -----------------------------------------------------------------------------
# 재현성 확보
# -----------------------------------------------------------------------------
# torch.manual_seed(42): PyTorch 난수 생성기의 시작점을 42로 고정한다.
#   nn.Linear는 만들어질 때 가중치와 편향을 무작위 값으로 초기화한다. 시드를 고정하면 몇 번을 실행해도 같은 초기 가중치가 나오므로
#   출력 logits 값이 매번 같아져 결과를 비교·검증할 수 있다. 42 자체에 특별한 의미는 없고, 관례적으로 많이 쓰는 고정값이다.
#   반드시 모델을 만들기 "전에" 호출해야 한다. 모델을 만든 뒤에 호출하면 이미 초기화된 가중치에는 영향을 주지 못한다.
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 두 모델 만들기
# -----------------------------------------------------------------------------
# custom (type: SimpleANN): Module 상속으로 만든 모델이다. 시드 직후의 난수로 가중치가 초기화된다.
custom = SimpleANN()
# sequential (type: nn.Sequential): 같은 구조의 모델이다. 클래스를 정의하지 않고 한 줄로 만든다.
#   custom이 난수를 먼저 쓴 뒤 이어지는 난수로 초기화되므로, 이 시점의 가중치는 custom과 다르다.
#   층 순서가 custom.net과 같으므로 내부 이름도 "0", "1", "2"로 같다. 이것이 다음 줄의 가중치 복사가 가능한 이유이다.
sequential = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))

# -----------------------------------------------------------------------------
# 가중치 맞추기
# -----------------------------------------------------------------------------
# 서로 다른 랜덤 초기값을 비교하지 않도록 같은 값을 복사한다.
# custom.net.state_dict() (type: OrderedDict): {"0.weight", "0.bias", "2.weight", "2.bias"} → 각 텐서의 사전이다. ReLU(인덱스 1)는 파라미터가 없어 키가 없다.
#   custom.state_dict()가 아니라 custom.net.state_dict()를 쓰는 이유: custom 전체의 키는 "net.0.weight"처럼 앞에 "net."이 붙어
#   sequential의 키 "0.weight"와 이름이 맞지 않는다. 안쪽 net의 사전을 꺼내야 키 이름이 정확히 일치한다.
# sequential.load_state_dict(...): 키 이름이 같은 자리에 값을 덮어쓴다. 이후 두 모델의 가중치와 편향은 숫자 하나까지 같다.
#   기본값 strict=True이므로 키가 하나라도 다르거나 빠지면 오류가 나서, 복사가 빠짐없이 되었음을 보장한다.
sequential.load_state_dict(custom.net.state_dict())

# -----------------------------------------------------------------------------
# 같은 입력으로 두 모델 실행하기
# -----------------------------------------------------------------------------
# X (type: torch.Tensor, dtype: torch.float32, shape: (3, 2)): 샘플 3개, 특성 2개짜리 입력이다.
#   1., 2.처럼 소수점을 붙여 float32로 만든다. 정수(int64)로 만들면 float32 가중치와 행렬곱할 때 dtype 오류가 난다.
#   음수(-1.)를 넣은 샘플이 있어 ReLU가 일부 은닉값을 0으로 만드는 경우도 함께 확인된다.
X = torch.tensor([[1.,2.], [0.,1.], [-1.,0.]])
# a (type: torch.Tensor, shape: (3, 2)): custom 모델의 출력 logits이다. custom(X) → SimpleANN.forward(X) → self.net(X)가 실행된다.
# b (type: torch.Tensor, shape: (3, 2)): sequential 모델의 출력 logits이다. 같은 가중치·같은 입력이므로 a와 같은 값이 나와야 한다.
#   학습 전이므로 logits 값 자체에는 의미가 없고, 두 값이 같은지만 본다.
a, b = custom(X), sequential(X)

# -----------------------------------------------------------------------------
# 비교 결과 출력하기
# -----------------------------------------------------------------------------
# tuple(a.shape): torch.Size를 일반 tuple로 바꿔 "(3, 2)"처럼 깔끔하게 표시한다.
# 예상 출력: 출력 shape: (3, 2) (3, 2)
#   샘플 3개 × 클래스 점수 2개이다. 두 모델 모두 마지막 층이 Linear(8, 2)이므로 같다.
print("출력 shape:", tuple(a.shape), tuple(b.shape))
# torch.allclose(a, b) (type: bool): 두 텐서의 모든 원소가 허용 오차(기본 rtol=1e-05, atol=1e-08) 안에서 같으면 True이다.
#   == 대신 allclose를 쓰는 이유: 실수 계산은 아주 작은 반올림 오차가 생길 수 있으므로, "정확히 같은가"보다 "오차 범위 안에서 같은가"로 비교하는 것이 안전하다.
# 예상 출력: 동일 출력: True
#   같은 가중치, 같은 층 순서, 같은 입력이므로 계산 과정이 완전히 같다. 
print("동일 출력:", torch.allclose(a, b))
# sum(p.numel() for p in custom.parameters()) (type: int): custom의 학습 파라미터 총개수이다.
#   custom.parameters() (type: generator): nn.Module이 자동 등록한 net.0.weight, net.0.bias, net.2.weight, net.2.bias를 차례로 꺼낸다.
#   p.numel() (type: int): 텐서 p 안의 숫자 개수이다. 예: shape (8, 2)이면 16이다.
# 예상 출력: 파라미터: 42
#   계산: Linear(2,8) = 2×8 + 8 = 24, ReLU = 0, Linear(8,2) = 8×2 + 2 = 18 → 24 + 0 + 18 = 42이다.
#   sequential도 같은 층을 가지므로 파라미터 수는 42로 같다. 클래스로 감싸도 파라미터가 늘거나 줄지 않는다.
print("파라미터:", sum(p.numel() for p in custom.parameters()))
