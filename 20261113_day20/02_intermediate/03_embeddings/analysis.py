"""교육용 2D 벡터의 Cosine 유사도·순위·shape를 분석하고 그림과 JSON으로 저장한다.

- s_ab, s_ac     : A-B 약 0.9701, A-C -1.0
- ranked         : D 추가 후 A 기준 유사도 내림차순 (B > D > C)
- shapes         : 벡터 하나 (2,), 행렬 (4,2)
- token_shape    : (심화) Token Embedding (2,4,4)
- pooled_shape   : (심화) 평균 Pooling 결과 (2,4)
- 결과: outputs/vectors.png, outputs/cosine_report.json
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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 수준 공통 모듈 common/support.py: Cosine 유사도 계산·JSON 저장
from common.support import cosine, write_json

# ============================================================
# 2. 재현성 설정
# ============================================================
# torch.manual_seed(42)  : 난수 초기값 고정 (Embedding·Linear 초기 가중치가 실행마다 같음)
# torch.set_num_threads(1) : CPU 스레드 1개로 고정해 연산 순서 차이 최소화
torch.manual_seed(42)
torch.set_num_threads(1)

# ============================================================
# 3. 실습 본문: 교육용 2D 벡터의 Cosine 유사도 분석
# ============================================================
def run_analysis():
    # PPT 33장과 같은 2D 벡터. SentenceTransformer가 만든 실제 의미 벡터가 아닙니다.
    # coords: 문장 ID -> 교육용 2차원 좌표
    # - A [1.0, 0.0]  : 기준 벡터
    # - B [0.8, 0.2]  : A와 거의 같은 방향
    # - C [-1.0, 0.0] : A와 정반대 방향
    coords = {"A":[1.0,0.0],"B":[0.8,0.2],"C":[-1.0,0.0]}

    # --------------------------------------------------------
    # 3-1. A-B, A-C Cosine 유사도 계산
    # --------------------------------------------------------
    # BEGIN D20-03-I01
    # TODO D20-03-I01 완료: A-B와 A-C 벡터의 Cosine Similarity를 계산하세요.
    # cosine(a, b) = (a·b) / (|a| |b|)
    # - A·B = 0.8, |A|=1, |B|=약 0.8246 -> 약 0.9701
    # - A·C = -1.0, |A|=1, |C|=1        -> -1.0
    s_ab=cosine(coords["A"],coords["B"])
    s_ac=cosine(coords["A"],coords["C"])
    # 해설: A-B는 약 0.9701, A-C는 -1.0입니다. 벡터 길이 대신 방향을 비교합니다.
    # END D20-03-I01

    # --------------------------------------------------------
    # 3-2. 점수의 부호·범위로 방향 해석
    # --------------------------------------------------------
    # BEGIN D20-03-I02
    # TODO D20-03-I02 완료: 점수의 부호·범위를 보고 어느 벡터 방향이 유사한지 판별하세요.
    # interpretation 키 : 의미
    # - A_B  : 0.9 초과면 "방향이 비슷함" (0.9701 -> 비슷함)
    # - A_C  : -0.99 미만이면 "반대 방향" (-1.0 -> 반대)
    # - note : 높은 Cosine이 RAG 답변의 사실성을 보장하지 않는다는 주의
    interpretation={"A_B":"방향이 비슷함" if s_ab>0.9 else "차이 있음",
                    "A_C":"반대 방향" if s_ac<-0.99 else "차이 있음",
                    "note":"코사인이 높다는 사실만으로 RAG 답변의 사실성을 보장하지 않는다"}
    # END D20-03-I02

    # --------------------------------------------------------
    # 3-3. 후보 D 추가와 A 기준 유사도 순위
    # --------------------------------------------------------
    # BEGIN D20-03-I03
    # TODO D20-03-I03 완료: 2차원 후보 D를 추가해 가장 유사한 벡터를 점수순으로 정리하세요.
    # coords["D"]=[0.5, 0.8] : A와 비스듬한 방향의 후보 추가
    # sorted(..., key=lambda x:(-x[1], x[0])) : 유사도 내림차순, 같으면 ID 오름차순
    # 결과: [("B", 0.9701), ("D", 0.5300), ("C", -1.0)]
    coords["D"]=[0.5,0.8]
    ranked=sorted(((key,cosine(coords["A"],val)) for key,val in coords.items() if key!="A"),
                  key=lambda x:(-x[1],x[0]))
    # END D20-03-I03

    # --------------------------------------------------------
    # 3-4. 벡터·행렬 shape 확인
    # --------------------------------------------------------
    # BEGIN D20-03-I04
    # TODO D20-03-I04 완료: 벡터를 행렬로 쌓은 Tensor shape (개수, 차원)를 기록하세요.
    # matrix: 좌표들을 행으로 쌓은 (문장 수, 차원) 행렬
    # shapes 키 : 의미
    # - vector : 벡터 하나의 shape (2,)
    # - matrix : 전체 행렬 shape
    matrix=torch.tensor(list(coords.values()),dtype=torch.float32)
    shapes={"vector":list(matrix[0].shape), "matrix":list(matrix.shape)}
    # END D20-03-I04
    # 심화 전용 결과 자리(token_shape, pooled_shape): 이 수준에서는 빈 값으로 기록
    token_shape = []
    pooled_shape = []
    # limitations: 결과 해석의 한계 (예시 벡터는 의미 품질 근거가 아님)
    limitations = ["이 숫자는 예시 벡터이며 문장 의미 품질을 증명하지 않는다"]

    # --------------------------------------------------------
    # 3-5. 2D 산점도와 cosine_report.json 저장
    # --------------------------------------------------------
    # BEGIN D20-03-I05
    # TODO D20-03-I05 완료: 2D 점 그래프와 cosine_report.json을 outputs에 저장하세요.
    # plt.subplots(figsize=(6,4)) : 가로 6인치·세로 4인치 그림 1개
    # ax.scatter(x, y, s=70)      : 좌표에 점 표시 (s=70 점 크기)
    # ax.annotate(label, ...)     : 점 오른쪽 위(5pt, 5pt)에 문장 ID 표시
    fig,ax=plt.subplots(figsize=(6,4))
    for label,(x,y) in coords.items():
        ax.scatter(x,y,s=70); ax.annotate(label,(x,y),xytext=(5,5),textcoords="offset points")
    # 축 이름·제목 설정, x=0·y=0 기준선(lw=0.6), 옅은 격자(alpha=0.2)
    ax.set(xlabel="x1",ylabel="x2",title="Educational 2D vectors")
    ax.axhline(0,lw=.6);ax.axvline(0,lw=.6);ax.grid(alpha=.2)
    # 여백 정리 후 outputs 폴더 생성, vectors.png로 저장(dpi=125), 메모리 해제(plt.close)
    fig.tight_layout()
    (HERE/"outputs").mkdir(exist_ok=True)
    fig.savefig(HERE/"outputs"/"vectors.png",dpi=125)
    plt.close(fig)
    # cosine_report.json 키 : 의미
    # - vectors        : 사용한 좌표
    # - cosine         : AB, AC 유사도
    # - interpretation : 방향 해석
    # - ranked         : A 기준 순위 (응용·심화)
    # - shapes         : 벡터·행렬 shape
    # - token_shape, pooled_shape : (심화) Token Embedding·Pooling shape
    # - limitations    : 해석 한계
    write_json(HERE/"outputs"/"cosine_report.json",{
        "vectors":coords,"cosine":{"AB":s_ab,"AC":s_ac},
        "interpretation":interpretation,"ranked":ranked,"shapes":shapes,
        "token_shape":token_shape,"pooled_shape":pooled_shape,"limitations":limitations,
    })
    # 콘솔 확인 예: [03] AB=0.9701 AC=-1.0000, shape=[4, 2]
    print(f"[03] AB={s_ab:.4f} AC={s_ac:.4f}, shape={shapes['matrix']}")
    # END D20-03-I05

