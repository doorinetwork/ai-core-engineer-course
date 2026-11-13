"""PPT Q/K/V 예제로 Score -> Causal Mask -> Softmax Weight -> Output을 계산해 저장한다.

- scores  : Softmax 전 Query-Key 점수 (3,3)
- weights : 미래 칸 0, 행 합 1 (t3 행 약 [0.25, 0.25, 0.50])
- output  : weights @ v (t3 행 약 [0.75, 0.75])
- 결과: outputs/attention_report.json
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 디렉토리 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 디렉토리 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 디렉토리에서 실행해도 common 패키지를 찾습니다.
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 수준 공통 모듈 common/support.py: JSON 저장
from common.support import write_json

# ============================================================
# 2. 재현성 설정
# ============================================================
torch.manual_seed(42)
torch.set_num_threads(1)

# ============================================================
# 3. 실습 본문: Q/K/V 3x2 예제의 Causal Self-Attention 계산
# ============================================================
def main():
    # q: Token 3개 x 차원 2 (PPT 슬라이드 62) -> t1=[1,0], t2=[0,1], t3=[1,1]
    # k, v: q를 복제해 같은 값 사용 (학습된 Wq·Wk·Wv 투영 없이 원리만 확인)
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    k,v=q.clone(),q.clone()

    # --------------------------------------------------------
    # 3-1. Scaled Dot-Product Score 계산
    # --------------------------------------------------------
    # TODO D20-05-B01: Scaled Dot-Product Q@K.T/sqrt(d_k)로 (3,3) Score를 계산하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # q @ k.T                  : (3,2) @ (2,3) -> (3,3), [i,j] = Query i와 Key j의 내적
    # / math.sqrt(q.shape[-1]) : d_k=2 -> 1.4142로 나눔
    # 예: scores[2] = [0.7071, 0.7071, 1.4142]
    # BEGIN D20-05-B01
    # TODO: / math.sqrt(q.shape[-1])로 Scale 적용
    scores=q@k.T/math.sqrt(q.shape[-1]) 
    # TODO: assert scores.shape == (3, 3) 확인
    assert scores.shape==(3,3)
    # END D20-05-B01

    # --------------------------------------------------------
    # 3-2. 미래 Key를 막는 Causal Mask 적용
    # --------------------------------------------------------
    # TODO D20-05-B02: 미래 Key를 차단하는 upper-triangle Causal Mask를 작성하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # torch.triu(torch.ones(3,3, bool), diagonal=1) : 대각선 위만 True
    # -> [[F, T, T], [F, F, T], [F, F, F]]  (True = 미래 위치 차단)
    # scores.masked_fill(mask, -inf) : True 칸을 -inf로 바꿈 -> Softmax 후 0
    # BEGIN D20-05-B02            
    # TODO: torch.triu(..., diagonal=1)
    mask=torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)
    # TODO: mask=True 위치를 -inf로 채우기
    masked_scores=scores.masked_fill(mask,float("-inf"))
    # END D20-05-B02

    # --------------------------------------------------------
    # 3-3. Softmax Weight와 행 합 확인
    # --------------------------------------------------------
    # TODO D20-05-B03: Score를 Softmax Weight로 바꾸고 Mask된 칸=0, 행 합=1을 확인하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # torch.softmax(masked_scores, dim=-1) : Key 방향(각 행)으로 정규화
    # 결과 (반올림): t1 [1.00, 0.00, 0.00] / t2 [0.33, 0.67, 0.00] / t3 [0.25, 0.25, 0.50]
    # row_sums = weights.sum(dim=-1) : 각 Query 행의 합 -> [1, 1, 1]
    # 확인: 행 합이 1(오차 1e-6)이고 Mask=True 칸의 Weight가 정확히 0
    # BEGIN D20-05-B03
    weights=torch.softmax(masked_scores,dim=-1)  # TODO: masked_scores 사용
    row_sums = weights.sum(dim=-1)  # 각 Query 행의 합
    # TODO: assert torch.allclose(row_sums, torch.ones(3))
    assert torch.allclose(row_sums,torch.ones(3),atol=1e-6)
    # TODO: assert torch.all(weights[mask] == 0)
    assert torch.all(weights[mask]==0)
    # END D20-05-B03
    # Mask 비교·Multi-Head
    plain_weights=torch.empty(0)
    diff=None
    head_weights={}
    # interpretation_limit: Weight 해석의 한계
    interpretation_limit=["Attention Weight만으로 모델의 최종 판단 전체를 설명할 수 없다"]

    # --------------------------------------------------------
    # 3-4. Output 계산과 결과 JSON 저장
    # --------------------------------------------------------
    # TODO D20-05-B04: Score·Causal Mask·Weight·행 합과 Output을 JSON으로 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-B04
    # weights @ v : (3,3) @ (3,2) -> (3,2), 각 Query가 Value를 Weight 비율로 섞은 결과
    # 예: t3 = 0.25*[1,0] + 0.25*[0,1] + 0.50*[1,1] = 약 [0.75, 0.75]
    # attention_report.json 키 : 의미
    # - scores, causal_mask, weights, row_sums, output : 계산 단계별 값
    # - no_mask_weights, masked_vs_unmasked_l1         : Mask 비교 (응용·심화)
    # - multi_head                                     : Head 수 비교 (심화)
    # - limits                                         : 해석 한계
    output=weights@v  # TODO: weights @ v
    report = {
        "scores": scores.tolist(),        # TODO: scores.tolist()
        "causal_mask": mask.tolist(),   # TODO: mask.tolist()
        "weights": weights.tolist(),       # TODO: weights.tolist()
        "row_sums": row_sums.tolist(),      # TODO: row_sums.tolist()
        "output": output.tolist(),        # TODO: output.tolist()
        "no_mask_weights": plain_weights.tolist(),
        "masked_vs_unmasked_l1": diff,
        "multi_head": head_weights,
        "limits": interpretation_limit,
    }
    write_json(HERE / "outputs" / "attention_report.json", report)
    # END D20-05-B04

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__=="__main__":
    main()
