"""학습/추론 단계 분류, logits에서 softmax값 확인

- probs       : 약 [0.665, 0.245, 0.090]
- temperature : T=0.5 / 1.0 / 2.0 분포
- greedy      : 최고 확률 후보 index 0 ("출력A")
- 결과: outputs/inference_report.json
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 교시 디렉토리 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 디렉토리 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 디렉토리에서 실행해도 common 패키지를 찾습니다.
import torch
import torch.nn.functional as F
# 수준 공통 모듈 common/support.py: JSON 저장
from common.support import write_json

# ============================================================
# 2. 재현성 설정
# ============================================================
# torch.manual_seed(42)  : 난수 초기값 고정 (Embedding·Linear 초기 가중치가 실행마다 같음)
# torch.set_num_threads(1) : CPU 스레드 1개로 고정해 연산 순서 차이 최소화
torch.manual_seed(42)
torch.set_num_threads(1)

# ============================================================
# 3. 실습 본문: Logit -> 확률 -> 다음 Token 선택
# ============================================================
def main():
    # logits: 후보 3개에 대한 예시 점수 (확률이 아님, 합이 1이 아님)
    # labels: 각 index의 후보 이름 (0 -> "출력A", 1 -> "출력B", 2 -> "출력C")
    logits=torch.tensor([2.0,1.0,0.0])
    labels=["출력A","출력B","출력C"]

    # --------------------------------------------------------
    # 3-1. Logit을 Softmax 확률로 변환
    # --------------------------------------------------------
    # TODO D20-06-B01: Logit[2,1,0]을 Softmax로 바꾸어 확률의 합이 1인지 확인하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B01
    # torch.softmax(logits, dim=0) : exp(z) / sum(exp(z)) -> [0.6652, 0.2447, 0.0900]
    # 확인: 확률 합이 1 (오차 1e-6)
    prob=torch.softmax(logits,dim=0)  # TODO: torch.softmax(logits, dim=0)
    # TODO: assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(prob.sum(),torch.tensor(1.0),atol=1e-6)
    # 해설: Logit은 확률이 아니라 상대 점수입니다. Softmax 출력이 확률 분포입니다.
    # END D20-06-B01

    # --------------------------------------------------------
    # 3-2. Temperature별 분포 비교
    # --------------------------------------------------------
    # TODO D20-06-B02: Temperature 0.5·1.0·2.0에서 확률 분포를 비교하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B02
    # logits / t 를 Softmax -> 온도별 확률 (키는 문자열 "0.5", "1.0", "2.0")
    # - T=0.5 : [0.8668, 0.1173, 0.0159] 상위 후보로 집중
    # - T=1.0 : [0.6652, 0.2447, 0.0900] 기본 분포
    # - T=2.0 : [0.5065, 0.3072, 0.1863] 완만한 분포
    temperature_results={str(t):torch.softmax(logits/t,dim=0).tolist() for t in (0.5,1.0,2.0)}
    # 해설: 같은 Logit에서 T가 낮으면 상위 후보 집중, 높으면 분포 평탄화.
    # END D20-06-B02

    # --------------------------------------------------------
    # 3-3. Greedy 선택과 학습/추론 구분
    # --------------------------------------------------------
    # TODO D20-06-B03: Greedy로 다음 Token 후보를 선택하고 학습/추론 차이를 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B03
    # torch.argmax(prob) : 가장 큰 확률의 index Tensor
    # .item()            : 파이썬 정수로 변환 -> 0
    # selected_token = labels[0] = "출력A"
    # training_vs_inference 키 : 의미
    # - training       : loss -> backward -> optimizer.step으로 가중치 갱신
    # - inference      : 가중치 고정, 현재 문맥에서 다음 Token 선택
    # - autoregressive : 선택 Token을 문맥에 붙여 EOS/길이 제한까지 반복
    selected_index = int(torch.argmax(prob).item())    # TODO: int(torch.argmax(prob).item())
    selected_token = labels[selected_index]     # TODO: labels[selected_index]
    training_vs_inference = {
        "training": "정답 Token과 예측을 비교해 loss/backward/optimizer.step으로 파라미터 갱신",       # TODO: loss·backward·파라미터 업데이트
        "inference": "가중치를 갱신하지 않고 현재 문맥에서 다음 Token을 선택",                          # TODO: 가중치를 바꾸지 않고 다음 Token 예측
        "autoregressive": "선택한 Token을 문맥에 추가하고 EOS/길이 제한까지 반복",                      # TODO: 생성 Token을 다시 문맥에 추가
    }
    # END D20-06-B03
    top_k={}
    sampled_index=None
    sampled_token=None
    top_p={}
    training_check={}
    kv_note={"stores":"이전 K/V 재사용","not":"학습이나 장기 기억이 아님"}

    # --------------------------------------------------------
    # 3-4. 결과 JSON 저장
    # --------------------------------------------------------
    # BEGIN D20-06-B04
    # inference_report.json 키 : 의미
    # - logits, probs, temperature : 점수·확률·온도별 분포
    # - greedy                     : 최고 확률 선택 index·token
    # - sampling                   : seed 42 추출 index·token (응용·심화)
    # - top_k, top_p               : 후보 축소 결과 (응용 top_k, 심화 top_p)
    # - training_vs_inference      : 학습/추론 설명
    # - training_step              : (심화) 1 step 학습 확인
    # - kv_cache                   : KV Cache 설명
    # TODO D20-06-B04: 확률·선택·학습/추론 차이를 inference_report.json에 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B04
    report = {
        "logits": logits.tolist(),
        "probs": prob.tolist(),          # TODO: prob.tolist()
        "temperature": temperature_results,
        "greedy": {"index": selected_index, "token": selected_token},
        "sampling": {"index": sampled_index, "token": sampled_token},
        "top_k": top_k,
        "top_p": top_p,
        "training_vs_inference": training_vs_inference,
        "training_step": training_check,
        "kv_cache": kv_note,
    }
    write_json(HERE / "outputs" / "inference_report.json", report)
    # END D20-06-B04

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__=="__main__":
    main()
