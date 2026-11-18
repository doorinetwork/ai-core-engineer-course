"""Day23 실습 03 · 응용 · tasks

응용 수준의 TODO 함수 5개를 모은 모듈이다. main.py가 import 해서 run()에서 호출한다.
E5 입력 문자열(query: / passage:)을 만들고 같은 Encoder로 인코딩한 뒤 ID↔벡터 행 대응과 벡터 계약을 검사한다.

이 파일의 TODO 함수:
- make_passages (D23-03-I01): 지식 목록을 E5 문서 입력 문자열 목록으로 변환하세요.
- make_queries (D23-03-I02): 질문 목록 각 요소 앞에 query: 공백을 붙이세요.
- encode_both (D23-03-I03): 같은 Encoder에 문서와 질문을 나누어 encode하세요.
- verify_alignment (D23-03-I04): 지식 행 수와 벡터 행 수가 같은지 확인하고 [(ID,행번호),...]를 반환하세요.
- check_shapes_vectors (D23-03-I05): 문서 벡터 shape=(doc_count,384), 질문 shape=(1,384)를 검사하세요.
"""

# ============================================================
# 1. 모듈 import
# ============================================================
import numpy as np
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (passage_text)
from day23kit.data import passage_text
# 제공 모듈 common/day23kit/embedding.py: Encoder 선택(fixture 모의 / live 실제 E5)·문서·질문 인코딩·벡터 계약 검사 (validate_vectors)
from day23kit.embedding import validate_vectors

# ============================================================
# 2. 문서 입력 문자열 목록 만들기
# ============================================================
# BEGIN D23-03-I01
def make_passages(rows):
    # 지식 목록을 같은 순서의 E5 문서 입력 문자열 목록으로 바꾼다.
    # - rows   : 지식 dict 목록 (기본필수 3건, 응용·심화 9건)
    # - 반환값 : 행마다 문자열 1개, 순서 유지 -> i번째 문자열 = rows[i]
    # 예: 101 -> "passage: 실습 파일 보관 실습 파일은 개인 폴더에 보관합니다."
    # [강사용 정답] D23-03-I01 · make_passages
    #     지식 목록을 E5 문서 입력 문자열 목록으로 변환하세요.
    #     문서에는 'passage: ' + 제목 + ' ' + 본문을 사용합니다.
    #     한 행마다 정확히 한 문자열을 만들어 순서를 유지하세요.
    # 해설 D23-03-I01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 문서마다 고정된 순서로 passage를 구성하여 ID와 행 인덱스를 보존합니다.
    # passage_text(row) : 제공 함수. 'passage: ' + 제목 + ' ' + 본문
    return [passage_text(row) for row in rows]
# END D23-03-I01

# ============================================================
# 3. 질문 입력 문자열 목록 만들기
# ============================================================
# BEGIN D23-03-I02
def make_queries(questions):
    # 질문 목록의 각 질문 앞에 'query: ' 접두어를 붙인다.
    # - questions : 질문 문자열 목록
    # - 반환값    : 같은 순서의 'query: 질문' 목록
    # 예: ['서버는 어디에서 실행하나요?'] -> ['query: 서버는 어디에서 실행하나요?']
    # 주의: 입력에 없는 질문(빈 질문 등)을 끼워 넣지 않는다
    # [강사용 정답] D23-03-I02 · make_queries
    #     질문 목록 각 요소 앞에 query: 공백을 붙이세요.
    #     출력: 입력과 동일한 순서의 문자열 리스트.
    #     빈 질문을 추가해 그럴듯한 벡터를 만들지 마세요.
    # 해설 D23-03-I02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return ['query: ' + q for q in questions]
# END D23-03-I02

# ============================================================
# 4. 같은 Encoder로 문서·질문 인코딩
# ============================================================
# BEGIN D23-03-I03
def encode_both(encoder, passages, queries):
    # 하나의 Encoder로 문서 입력과 질문 입력을 따로 인코딩해 float32 배열 두 개를 돌려준다.

    # - encoder  : fixture면 SimulatedEncoder(모의, E5 아님), live면 SentenceTransformer E5-small
    # - passages : make_passages() 결과 (n개)
    # - queries  : make_queries() 결과 (1개)
    # - 반환값   : (doc_vectors (n, 384), query_vectors (1, 384))
    # [강사용 정답] D23-03-I03 · encode_both
    #     같은 Encoder에 문서와 질문을 나누어 encode하세요.
    #     반드시 normalize_embeddings=True, convert_to_numpy=True를 전달합니다.
    #     결과: (doc_vectors, query_vectors). 기본필수의 모의 모드와
    #     --mode live의 E5 실측을 구분하여 같은 API를 사용합니다.
    # 해설 D23-03-I03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 동일 모델로 두 역할을 인코딩합니다. 인코딩은 모델을 새로 학습하지 않습니다.
    # encoder.encode 인자
    # - normalize_embeddings=True : 각 벡터 길이를 1로 맞춤 -> Cosine = 내적
    # - convert_to_numpy=True     : 결과를 NumPy 배열로 받음
    # - batch_size=8              : 문서를 8개씩 묶어 인코딩 (9건이면 2번)
    docs = encoder.encode(passages, normalize_embeddings = True, convert_to_numpy = True, batch_size = 8)
    queries_vec = encoder.encode(queries, normalize_embeddings = True, convert_to_numpy = True)
    # np.asarray(..., dtype=np.float32) : pgvector vector 타입과 같은 32비트 실수로 통일
    return np.asarray(docs, dtype = np.float32), np.asarray(queries_vec, dtype = np.float32)
# END D23-03-I03

# ============================================================
# 5. 지식 ID ↔ 벡터 행 번호 대응 확인
# ============================================================
# BEGIN D23-03-I04
def verify_alignment(rows, vectors):
    # 지식 행과 벡터 행이 1:1로 짝지어지는지 확인하고 (ID, 행 번호) 목록을 돌려준다.
    # - ID 중복 또는 행 수 불일치 : ValueError('id_vector_alignment')
    # - 정상                     : [(101, 0), (102, 1), ..., (109, 8)]
    # 예: 지식 102는 vectors[1]과 짝이다 (ID 102가 102번째 행이라는 뜻이 아님)
    # [강사용 정답] D23-03-I04 · verify_alignment
    #     지식 행 수와 벡터 행 수가 같은지 확인하고 [(ID,행번호),...]를 반환하세요.
    #     ID가 중복되어 있거나 행 수가 다르면 ValueError.
    #     예: 지식 102는 vectors[1]과 짝입니다.
    # 해설 D23-03-I04: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != len(vectors):
        raise ValueError('id_vector_alignment')
    # enumerate(rows) : (행 번호 i, 지식 r) 쌍
    return [(r['id'], i) for i, r in enumerate(rows)]
# END D23-03-I04

# ============================================================
# 6. 벡터 shape·정규화 계약 검사
# ============================================================
# BEGIN D23-03-I05
def check_shapes_vectors(docs, queries, doc_count):
    # 문서 벡터 (doc_count, 384)·질문 벡터 (1, 384)가 유한값·길이 1 계약을 지키는지 검사한다.
    # - shape·차원 오류 / NaN·Inf / 영벡터 / 길이 1 아님 : validate_vectors()가 ValueError
    # - 모두 통과                                       : True
    # [강사용 정답] D23-03-I05 · check_shapes_vectors
    #     문서 벡터 shape=(doc_count,384), 질문 shape=(1,384)를 검사하세요.
    #     두 배열 모두 유한하고 길이가 1인지 검증합니다.
    #     정상일 때 True, 오류일 때 ValueError.
    # 해설 D23-03-I05: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 모의 모드와 실제 E5 모두 같은 차원/정규화 계약을 갖습니다.
    # validate_vectors(배열, 행 수) : 제공 함수. shape=(행 수, 384)·유한값·영벡터·norm≈1(오차 1e-5) 검사
    validate_vectors(docs, doc_count)
    validate_vectors(queries, 1)
    return True
# END D23-03-I05
