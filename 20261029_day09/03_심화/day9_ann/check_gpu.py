"""장치 선택과 기록만 담당하는 제공 코드입니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [외부 라이브러리] torch(PyTorch): 딥러닝용 텐서 계산 라이브러리이다.
import torch

# -----------------------------------------------------------------------------
# 장치 선택 함수
# -----------------------------------------------------------------------------
# choose_device: 계산 장치를 고르고 환경 정보를 모아 돌려준다.
#   - 반환값 (type: tuple[torch.device, dict]): (선택한 장치, 환경 정보 사전)이다.
def choose_device():
    # device (type: torch.device): CUDA GPU를 쓸 수 있으면 첫 번째 GPU("cuda:0"), 아니면 "cpu"이다.
    # "cuda"가 아니라 "cuda:0"으로 번호를 적는 이유: GPU가 여러 개인 컴퓨터에서도 어느 GPU를 썼는지 기록이 정확해지기 때문이다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # info (type: dict): 실행 환경 기록이다.
    #   - "torch" (type: str): PyTorch 버전 문자열이다(예: 2.10.0+cpu). str()로 감싸 JSON·텍스트로 쓰기 쉽게 한다.
    #   - "cuda_build" (type: str 또는 None): 이 PyTorch가 어떤 CUDA 버전으로 빌드됐는지이다. CPU 전용 빌드는 None이다.
    #   - "cuda_available" (type: bool): 지금 이 컴퓨터에서 CUDA GPU를 실제로 쓸 수 있는지이다. 빌드가 CUDA용이어도 드라이버·GPU가 없으면 False이다.
    #   - "selected_device" (type: str): 위에서 고른 장치 이름("cuda:0" 또는 "cpu")이다.
    info = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "selected_device": str(device)}
    # GPU를 골랐을 때만 GPU 정보를 추가한다. CPU일 때 이 함수를 부르면 CUDA가 없어 오류가 나므로 조건으로 막는다.
    if device.type == "cuda":
        # "gpu_name" (type: str): GPU 모델 이름이다(예: NVIDIA GeForce RTX ...).
        info["gpu_name"] = torch.cuda.get_device_name(device)
        # "vram_gib" (type: float): GPU 전체 메모리 용량을 GiB 단위로 바꾼 값이다. total_memory는 바이트이므로 1024³으로 나눈다.
        info["vram_gib"] = torch.cuda.get_device_properties(device).total_memory / 1024**3
    # 장치와 환경 기록을 함께 돌려준다.
    return device, info
