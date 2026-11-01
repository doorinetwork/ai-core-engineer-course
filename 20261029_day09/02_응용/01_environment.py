"""실습 01: CUDA가 있으면 사용하고 없으면 CPU로 진행합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# 외부 라이브러리. 목적: 딥러닝용 Tensor를 다루기 위해 불러온다.
import torch

# -----------------------------------------------------------------------------
# 결과 저장 폴더 준비
# -----------------------------------------------------------------------------
# out (type: Path): 이 스크립트 파일이 있는 폴더 아래의 outputs 폴더 경로이다.
#   __file__: 지금 실행 중인 이 .py 파일의 경로이다.
#   .resolve(): 상대 경로를 절대 경로로 바꾼다. 어느 폴더에서 python을 실행하든 같은 위치를 가리키게 하려는 것이다.
#   .parent: 파일이 들어 있는 폴더를 뜻한다. 여기에 "outputs"를 이어 붙인다.
#   현재 작업 폴더 기준("outputs")으로 쓰면 실행 위치에 따라 저장 위치가 달라지므로, 파일 위치 기준으로 고정한다.
out = Path(__file__).resolve().parent / "outputs"
# outputs 폴더를 만든다.
#   exist_ok=True: 폴더가 이미 있어도 오류(FileExistsError)를 내지 않고 그냥 넘어간다.
out.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 계산 장치 선택
# -----------------------------------------------------------------------------
# device (type: torch.device): 이후 텐서 계산을 할 장치이다.
#   torch.cuda.is_available() (type: bool): 이 컴퓨터에서 CUDA GPU를 지금 쓸 수 있으면 True, 아니면 False이다.
#   'cuda:0' if 조건 else 'cpu': 조건부 표현식이다. True면 'cuda:0', False면 'cpu' 문자열을 고른다.
#   'cuda:0': 0번(첫 번째) GPU를 뜻한다. 요구사항이 "cuda:0"을 명시하므로 인덱스를 붙여 쓴다.
#     인덱스를 적으면 str(device)가 "cuda:0"이 되어 기록(selected_device)이 요구사항 문자열과 정확히 일치한다.
#   GPU가 없을 때 오류를 내지 않고 CPU로 내려가는 것이 이 실습의 요구사항("없으면 CPU로 진행")이다.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 선택한 장치에서 텐서 만들기
# -----------------------------------------------------------------------------
# x (type: torch.Tensor): 값 [1.0, 2.0]을 가진 실수형(float32) 텐서이다.
#   1., 2.처럼 소수점을 붙이면 정수형이 아니라 실수형 텐서가 된다. 딥러닝 계산은 실수형으로 하므로 실수형으로 만든다.
#   .to(device): 텐서를 선택한 장치(CPU 메모리 또는 GPU 메모리)로 옮긴다.
#   계산은 텐서가 있는 장치에서 일어나므로, 이 줄이 "선택된 device를 아래 연산에 사용한다"는 요구사항을 실제로 지킨다.
x = torch.tensor([1., 2.]).to(device)

# -----------------------------------------------------------------------------
# 실행 환경 정보 기록
# -----------------------------------------------------------------------------
# info (type: dict): 실행 환경을 나중에 다시 확인할 수 있도록 남기는 기록이다.
#   - "torch" (type: str): PyTorch 버전이다. str()로 감싸 일반 문자열로 만든다.
#   - "cuda_build" (type: str 또는 None): PyTorch가 어떤 CUDA 버전으로 빌드되었는지이다. CPU 전용판이면 None이다.
#     빌드 버전이 있어도 GPU·드라이버가 없으면 쓸 수 없으므로, 아래 cuda_available과 따로 기록한다.
#   - "cuda_available" (type: bool): 지금 GPU를 실제로 쓸 수 있는지이다.
#   - "selected_device" (type: str): 선택된 장치 문자열("cpu" 또는 "cuda:0")이다.
info = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "selected_device": str(device)}

# device.type (type: str): 인덱스를 뺀 장치 종류("cuda" 또는 "cpu")이다. "cuda:0"이어도 type은 "cuda"이다.
#   GPU 정보 조회 함수는 GPU가 없을 때 호출하면 오류가 나므로, GPU를 선택한 경우에만 실행한다.
if device.type == "cuda":
    # info["gpu_name"] (type: str): 선택한 GPU의 모델 이름이다.
    info["gpu_name"] = torch.cuda.get_device_name(device)
    # info["vram_gib"] (type: float): 선택한 GPU의 전체 메모리(VRAM) 크기이다. 단위는 GiB이다.
    #   total_memory (type: int): 바이트 단위 전체 메모리이다.
    #   1024**3: 1 GiB = 1024×1024×1024 바이트이므로 이 값으로 나눠 GiB로 바꾼다.
    #   round(..., 2): 소수 둘째 자리까지 반올림해 읽기 쉽게 만든다.
    info["vram_gib"] = round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2)

# -----------------------------------------------------------------------------
# 결과 출력과 파일 저장
# -----------------------------------------------------------------------------
# info의 항목을 하나씩 꺼내 "키: 값" 형식으로 화면에 출력한다.
#   key (type: str): 항목 이름이다. value (type: str, bool, float 또는 None): 항목 값이다.
for key, value in info.items():
    print(f"{key}: {value}")
    # 선택한 장치에서 x + 1 덧셈을 실제로 해 보고 결과를 출력한다. 예상 출력은 "연산: [2.0, 3.0]"이다.
#   x + 1 (type: torch.Tensor): 텐서의 모든 원소에 1을 더한다. x가 GPU에 있으면 덧셈도 GPU에서 한다.
#   .cpu(): 결과 텐서를 CPU로 가져온다. GPU 메모리의 값은 파이썬 리스트로 바로 바꿀 수 없으므로 먼저 CPU로 옮긴다.
#   .tolist() (type: list): 텐서를 파이썬 리스트로 바꿔 사람이 읽기 쉬운 형태로 출력한다.
print("연산:", (x + 1).cpu().tolist())
# info를 "키: 값" 한 줄씩 이어 붙여 outputs/gpu_check.txt 파일에 저장한다.
#   "\n".join(...) (type: str): 각 줄을 줄바꿈 문자로 이어 하나의 문자열로 만든다.
#   k (type: str), v: info의 키와 값이다.
#   encoding="utf-8": 한글 등 모든 문자가 운영체제와 무관하게 깨지지 않도록 UTF-8로 저장한다.
#   화면 출력은 실행이 끝나면 사라지므로, 어떤 환경에서 실행했는지 증거를 파일로 남긴다.
(out / "gpu_check.txt").write_text("\n".join(f"{k}: {v}" for k, v in info.items()), encoding="utf-8")
