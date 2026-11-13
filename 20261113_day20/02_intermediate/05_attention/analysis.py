"""Causal Self-Attention을 계산하고 Mask 유무·Heatmap(·Multi-Head)을 비교해 저장한다.

- scores, weights, output : Score·Weight(행 합 1, 미래 0)·가중합
- no_mask_weights, diff   : Mask 없는 Weight와 L1 차이
- attention_heatmap.png   : 행=Query, 열=Key Heatmap
- multi_head              : (심화) Head 1개·2개의 Query 0 행 Weight
- 결과: outputs/attention_report.json, outputs/attention_heatmap.png
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
# 난수 초기값 고정 (Embedding·Linear 초기 가중치가 실행마다 같음)
torch.manual_seed(42)
# CPU 스레드 1개로 고정해 연산 순서 차이 최소화
torch.set_num_threads(1)

# ============================================================
# 3. 실습 본문: Q/K/V 3x2 예제의 Causal Self-Attention 계산
# ============================================================
def run_analysis():
    # q: Token 3개 x 차원 2 -> t1=[1,0], t2=[0,1], t3=[1,1]
    # k, v: q를 복제해 같은 값 사용 (학습된 Wq·Wk·Wv 투영 없이 원리만 확인)
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    k,v=q.clone(),q.clone()

    # TODO D20-05-I01: Scaled Dot-Product Q@K.T/sqrt(d_k)로 (3,3) Score를 계산하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I01
    # q @ k.T                  : (3,2) @ (2,3) -> (3,3), [i,j] = Query i와 Key j의 내적
    # / math.sqrt(q.shape[-1]) : d_k=2 -> 1.4142로 나눔
    # 예: scores[2] = [0.7071, 0.7071, 1.4142]
    scores=q@k.T/math.sqrt(q.shape[-1])
    assert scores.shape==(3,3)
    # 해설: 행은 Query, 열은 Key 위치입니다. Softmax 전 Score는 확률이 아닙니다.
    # END D20-05-I01

    # TODO D20-05-I02: 미래 Key를 차단하는 upper-triangle Causal Mask를 작성하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I02
    # torch.triu(torch.ones(3,3, bool), diagonal=1) : 대각선 위만 True
    # -> [[F, T, T], [F, F, T], [F, F, F]]  (True = 미래 위치 차단)
    # scores.masked_fill(mask, -inf) : True 칸을 -inf로 바꿈 -> Softmax 후 0
    mask=torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)
    masked_scores=scores.masked_fill(mask,float("-inf"))
    # 해설: Bool Mask=True를 막는 의미로 사용하는 여기의 표현은 MultiheadAttention과 같습니다.
    # END D20-05-I02

    # TODO D20-05-I03: Score를 Softmax Weight로 바꾸고 Mask된 칸=0, 행 합=1을 확인하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I03
    # torch.softmax(masked_scores, dim=-1) : Key 방향(각 행)으로 정규화
    # 결과 (반올림): t1 [1.00, 0.00, 0.00] / t2 [0.33, 0.67, 0.00] / t3 [0.25, 0.25, 0.50]
    # row_sums = weights.sum(dim=-1) : 각 Query 행의 합 -> [1, 1, 1]
    # 확인: 행 합이 1(오차 1e-6)이고 Mask=True 칸의 Weight가 정확히 0
    weights=torch.softmax(masked_scores,dim=-1)
    row_sums=weights.sum(dim=-1)
    assert torch.allclose(row_sums,torch.ones(3),atol=1e-6)
    assert torch.all(weights[mask]==0)
    # 해설: 모든 Query 행에 유효한 Key가 최소 한 개 있어야 NaN을 피할 수 있습니다.
    # END D20-05-I03

    # TODO D20-05-I04: Mask 없는 Softmax 결과와 Mask 있는 결과를 비교하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I04
    # plain_weights : Mask 없이 Softmax -> 미래 Key에도 Weight가 생김
    # diff          : 두 Weight 행렬의 원소별 절댓값 차이 합 (L1)
    # 예: t1 행은 Mask 없으면 [0.40, 0.20, 0.40] 수준으로 퍼지고, Mask 있으면 [1, 0, 0]
    plain_weights=torch.softmax(scores,dim=-1)
    diff=float(torch.abs(plain_weights-weights).sum())
    # 해설: 미래 Token의 Weight가 0으로 바뀌고 나머지 허용 위치는 재정규화됩니다.
    # END D20-05-I04

    # TODO D20-05-I05: Causal Weight를 Query행·Key열 기준으로 Heatmap에 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I05
    # ax.imshow(weights.numpy(), vmin=0, vmax=1, cmap="Blues") : 0~1 범위 파란색 Heatmap
    # xticks/yticks=range(3) : 칸 번호 0, 1, 2
    # xlabel="Key index"     : 열 = Key (참조되는 위치)
    # ylabel="Query index"   : 행 = Query (현재 계산 중인 위치)
    fig,ax=plt.subplots(figsize=(4.8,4))
    heat=ax.imshow(weights.numpy(),vmin=0,vmax=1,cmap="Blues")
    ax.set(xticks=range(3),yticks=range(3),xlabel="Key index",ylabel="Query index")
    # 각 칸 중앙에 Weight 값을 소수 둘째 자리로 표시 (j=열, i=행)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{weights[i,j]:.2f}",ha="center",va="center")
    # 색 막대 추가 -> 여백 정리 -> outputs/attention_heatmap.png 저장(dpi=130) -> 메모리 해제
    fig.colorbar(heat,ax=ax)
    fig.tight_layout()
    (HERE/"outputs").mkdir(exist_ok=True)
    fig.savefig(HERE/"outputs"/"attention_heatmap.png",dpi=130)
    plt.close(fig)
    # END D20-05-I05
    head_weights={}
    interpretation_limit=["Attention Weight만으로 모델의 최종 판단 전체를 설명할 수 없다"]

    # --------------------------------------------------------
    # 3-6. Output 계산과 결과 JSON 저장
    # --------------------------------------------------------
    # TODO D20-05-I06: Score·Causal Mask·Weight·행 합과 Output을 JSON으로 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-I06
    # weights @ v : (3,3) @ (3,2) -> (3,2), 각 Query가 Value를 Weight 비율로 섞은 결과
    # 예: t3 = 0.25*[1,0] + 0.25*[0,1] + 0.50*[1,1] = 약 [0.75, 0.75]
    # attention_report.json 키 : 의미
    # - scores, causal_mask, weights, row_sums, output : 계산 단계별 값
    # - no_mask_weights, masked_vs_unmasked_l1         : Mask 비교
    # - multi_head                                     : Head 수 비교
    # - limits                                         : 해석 한계
    output=weights@v
    write_json(HERE/"outputs"/"attention_report.json",{
        "scores":scores.tolist(),"causal_mask":mask.tolist(),
        "weights":weights.tolist(),"row_sums":row_sums.tolist(),
        "output":output.tolist(),"no_mask_weights":plain_weights.tolist(),
        "masked_vs_unmasked_l1":diff,"multi_head":head_weights,
        "limits":interpretation_limit,
    })
    print("[05]",weights.round(decimals=2).tolist(),"row_sums",row_sums.tolist())
    # END D20-05-I06

