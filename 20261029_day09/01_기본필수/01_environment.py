"""실습 01: 설치된 PyTorch와 CUDA 노출을 확인합니다. GPU가 없어도 CPU로 진행합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 외부 라이브러리. 목적: 딥러닝용 Tensor를 다루기 위해 불러온다.
import torch

# -----------------------------------------------------------------------------
# PyTorch 설치 정보 확인하기
# -----------------------------------------------------------------------------
# torch.__version__ (type: str) 은 설치된 PyTorch 버전 문자열이다. 예: "2.5.1+cu121"
# 버전을 먼저 출력하는 이유는, 교안·동료와 결과가 다를 때 버전 차이부터 확인하기 위해서이다.
print("PyTorch:", torch.__version__)
# torch.version.cuda (type: str 또는 None) 는 이 PyTorch가 어떤 CUDA 버전으로 빌드되었는지를 나타낸다.
# CPU 전용으로 설치한 PyTorch이면 None이 출력된다. 즉 None이면 GPU가 있어도 PyTorch가 GPU를 쓸 수 없다.
print("PyTorch CUDA 빌드:", torch.version.cuda)
# torch.cuda.is_available() (type: bool) 은 지금 이 실행에서 PyTorch가 GPU를 실제로 쓸 수 있는지를 True/False로 돌려준다.
# "CUDA 빌드"가 있어도 드라이버가 없거나 GPU가 없으면 False가 나온다. 그래서 두 값을 따로 확인한다.
print("CUDA 사용 가능:", torch.cuda.is_available())

# -----------------------------------------------------------------------------
# GPU 정보 확인하기 (GPU가 있을 때만)
# -----------------------------------------------------------------------------
# GPU 함수는 사용 가능한 경우에만 호출한다. 드라이버 번호는 터미널 nvidia-smi에서 별도로 기록한다.
# GPU가 없는데 get_device_name 같은 함수를 부르면 오류가 나므로, if로 먼저 사용 가능 여부를 검사한다.
# GPU 함수는 사용 가능한 경우에만 호출합니다. 드라이버 번호는 터미널 nvidia-smi에서 별도로 기록합니다.
if torch.cuda.is_available():
    # torch.cuda.get_device_name(0) (type: str) 은 0번 GPU(첫 번째 GPU)의 이름이다. 예: "NVIDIA GeForce RTX 4090"
    # 인자 0은 GPU 번호이다. GPU가 한 장이면 번호는 0 하나뿐이다.
    print("GPU:", torch.cuda.get_device_name(0))
    # get_device_properties(0).total_memory (type: int) 는 0번 GPU의 전체 메모리(VRAM)를 바이트 단위로 돌려준다.
    # 1024**3 (type: int) 으로 나누면 바이트가 GiB(기비바이트) 단위로 바뀐다. (1 GiB = 1024×1024×1024 바이트)
    # round(..., 2) (type: float) 는 소수점 둘째 자리까지 반올림해 읽기 쉽게 만든다.
    # VRAM 크기를 아는 이유는, 나중에 모델·배치 크기를 정할 때 GPU 메모리가 한계이기 때문이다.
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("GPU: 이번 실행에서는 사용하지 않음")
# 이 실습 단계는 GPU 유무와 상관없이 CPU에서 계산한다는 것을 명시한다.
# 아래 sample Tensor도 device를 지정하지 않았으므로 기본값인 cpu에 만들어진다.
print("이 단계의 연산 장치: cpu")

# -----------------------------------------------------------------------------
# Tensor 덧셈으로 계산 환경 점검하기
# -----------------------------------------------------------------------------
# torch.tensor([...]) 는 파이썬 리스트를 PyTorch Tensor로 바꾼다.
# sample (type: torch.Tensor, dtype: torch.float32, shape: (2,), device: cpu) 이다.
# 1, 2가 아니라 1.0, 2.0으로 쓰는 이유는 실수(float) Tensor를 만들기 위해서이다.
# 정수로 쓰면 dtype이 int64가 되고, 딥러닝 계산(가중치·기울기)은 실수형을 쓰므로 처음부터 float32로 맞춘다.
# Tensor는 "숫자들을 담는 상자"인데, 이 상자에 담긴 숫자들은 한꺼번에 같은 계산을 받을 수 있다.
sample = torch.tensor([1.0, 2.0])
# sample + 1 (type: torch.Tensor) 은 브로드캐스팅으로 모든 원소에 1을 더해 tensor([2., 3.])이 된다.
# .tolist() (type: list) 는 Tensor를 파이썬 리스트로 바꿔 [2.0, 3.0]처럼 보기 쉽게 출력한다.
print("덧셈:", (sample + 1).tolist())
