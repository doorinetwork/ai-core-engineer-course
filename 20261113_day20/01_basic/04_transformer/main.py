"""Token ID (2,5) -> Embedding (2,5,8) -> EncoderLayer 출력 (2,5,8) shape를 확인해 저장한다.

- PAD(ID 0) 위치는 src_key_padding_mask=True로 Key에서 제외
- components : Block 구성요소 역할 (응용·심화에서 Residual·LayerNorm 추가)
- 결과: outputs/transformer_report.json
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 폴더 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
import torch
from torch import nn
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
# 3. 실습 본문: Embedding -> TransformerEncoderLayer shape 추적
# ============================================================
def main():
    token_ids=torch.tensor([[2,4,5,3,0],[2,6,7,3,0]],dtype=torch.long)

    # --------------------------------------------------------
    # 3-1. Token ID를 Token Vector로 변환
    # --------------------------------------------------------
    # TODO D20-04-B01: Token ID 텐서와 nn.Embedding 출력의 shape (B,T,D)를 구하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B01
    # nn.Embedding 인자
    # - num_embeddings=16 : 어휘 크기 (ID 0~15 조회 가능)
    # - embedding_dim=8   : Token 벡터 차원 D
    # embedding(token_ids) : 각 ID를 길이 8 실수 벡터로 조회 -> x shape (2, 5, 8)
    # id_shape=[2, 5], emb_shape=[2, 5, 8]
    embedding = nn.Embedding(num_embeddings=16, embedding_dim=8)  # ID 16종, 특성 8개
    x = embedding(token_ids)       # TODO: embedding(token_ids) 호출
    id_shape = list(token_ids.shape)  # TODO: token_ids.shape을 리스트로 변환
    emb_shape = list(x.shape) # TODO: x.shape을 리스트로 변환
    # END D20-04-B01

    # --------------------------------------------------------
    # 3-2. TransformerEncoderLayer 통과와 shape 보존 확인
    # --------------------------------------------------------
    # TODO D20-04-B02: EncoderLayer(d_model=8,nhead=2)로 전달하고 shape 보존을 검증하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B02
    # nn.TransformerEncoderLayer 인자
    # - d_model=8          : 입력·출력 feature 차원 (Embedding 차원과 같아야 함)
    # - nhead=2            : Attention Head 2개 (Head당 8/2=4차원)
    # - dim_feedforward=16 : FFN 은닉 차원 (8 -> 16 -> 8)
    # - batch_first=True   : 입력 shape를 (Batch, Token, Dim) 순서로 사용
    # - dropout=0.0        : 실행마다 같은 결과가 나오도록 Dropout 끔
    # layer.eval() : 추론 모드 (Dropout 등 학습 전용 동작 비활성)
    layer = nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0
    )
    layer.eval()
    pad_mask = token_ids == 0  # PAD 위치는 True로 표시
    # MultiheadAttention의 padding mask에서 True는 차단입니다.
    # torch.no_grad() : 기울기 계산 없이 순전파만 수행
    # src_key_padding_mask=pad_mask : True인 Key 위치(PAD)를 Attention에서 제외
    # 확인: 입력 x와 출력 y의 shape가 같다 -> out_shape=[2, 5, 8]
    with torch.no_grad():
        y=layer(x,src_key_padding_mask=pad_mask)    # TODO: torch.no_grad() 안에서 layer(x, src_key_padding_mask=pad_mask)
    assert x.shape==y.shape
    out_shape=list(y.shape) # TODO: y.shape을 리스트로 변환
    # TODO: assert로 Embedding 입력/Encoder 출력 shape가 같은지 비교

    # END D20-04-B02
    # Attention·FFN 역할, 출력 shape
    components={"attention":"토큰 간 정보","ffn":"특징 변환","shape":out_shape}
    # padding mask
    padding_positions=[]
    pad_values=None
    padding_note="PAD Mask는 Causal Mask와 별개"
    # head 제약·추가 shape
    head_divisibility_error=None
    extra_shapes={}
    architecture_note="이 코드는 Encoder Layer의 Shape 실습"

    # --------------------------------------------------------
    # 3-3. 결과 JSON 저장
    # --------------------------------------------------------
    # BEGIN D20-04-B03
    # transformer_report.json 키 : 의미
    # - ids, embedding, encoder_output : 단계별 shape ([2,5], [2,5,8], [2,5,8])
    # - pad_key_mask                   : PAD 위치 True Mask
    # - padding_positions              : PAD 인덱스 (응용·심화)
    # - padding_output_abs_sum         : PAD 위치 출력 절댓값 합 (응용·심화)
    # - components                     : Block 구성요소 역할
    # - bad_head_rejected              : (심화) nhead=3 거부 여부
    # - extra_shapes                   : (심화) 새 입력·출력 shape
    # TODO D20-04-B03: 모든 shape와 Mask 의미를 transformer_report.json에 정리하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B03
    report = {
        "ids": id_shape,             # TODO: id_shape
        "embedding": emb_shape,       # TODO: emb_shape
        "encoder_output": out_shape,  # TODO: out_shape
        "pad_key_mask": pad_mask.tolist(),    # TODO: pad_mask.tolist()
        "components": components,
        "padding_positions": padding_positions,
        "padding_output_abs_sum": pad_values,
        "padding_note": padding_note,
        "bad_head_rejected": head_divisibility_error,
        "extra_shapes": extra_shapes,
        "architecture_note": architecture_note,
    }
    write_json(HERE / "outputs" / "transformer_report.json", report)
    # END D20-04-B03

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__=="__main__":
    main()
