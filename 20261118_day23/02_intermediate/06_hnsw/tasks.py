"""Day23 실습 06 · 응용 · tasks

응용 수준의 TODO 함수 5개를 모은 모듈이다. main.py가 import 해서 run()에서 호출한다.
HNSW 인덱스 SQL·EXPLAIN SQL을 만들고, 실행 계획 판독·연산자 클래스 확인·Exact/ANN 결과 일치율을 계산한다.

이 파일의 TODO 함수:
- create_index_sql (D23-06-I01): day23_search_docs.embedding의 Cosine HNSW 인덱스를 정의하세요.
- explain_sql (D23-06-I02): SELECT id FROM day23_search_docs를 거리순으로 검색하는
- read_scan_type (D23-06-I03): PostgreSQL EXPLAIN 텍스트에서 Index Scan / Seq Scan / 기타를 구분하세요.
- check_operator_compatibility (D23-06-I04): SQL이 cosine <=> 를 사용하면 vector_cosine_ops가 맞는지 검사하세요.
- compare_topk (D23-06-I05): Exact Top-k와 ANN Top-k의 ID 집합 일치율을 계산하세요.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (HNSW_SQL)
from day23kit.db import HNSW_SQL

# ============================================================
# 2. HNSW 인덱스 정의 SQL
# ============================================================
# BEGIN D23-06-I01
def create_index_sql():
    # day23_search_docs.embedding 열에 Cosine 거리용 HNSW 인덱스를 만드는 SQL을 돌려준다.
    # - USING hnsw               : 근사 최근접(ANN) 그래프 인덱스
    # - vector_cosine_ops        : <=>(Cosine distance) 연산자와 짝인 연산자 클래스
    # - IF NOT EXISTS            : 이미 있으면 다시 만들지 않음
    # 주의: 인덱스는 선택 항목이며 9건 데이터에서 빨라진다는 보장이 없다
    # [강사용 정답] D23-06-I01 · create_index_sql
    #     day23_search_docs.embedding의 Cosine HNSW 인덱스를 정의하세요.
    #     USING hnsw (embedding vector_cosine_ops)를 사용합니다.
    #     인덱스는 선택 항목이며 9건에서 빠른 성능을 보장하지 않습니다.
    # 해설 D23-06-I01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # HNSW_SQL : 제공 상수 (common/day23kit/db.py)
    return HNSW_SQL
# END D23-06-I01

# ============================================================
# 3. 실행 계획 확인 SQL (EXPLAIN)
# ============================================================
# BEGIN D23-06-I02
def explain_sql():
    # 거리순 Top-3 검색의 실제 실행 계획을 보는 EXPLAIN (ANALYZE, BUFFERS) SQL을 돌려준다.
    # %s 바인딩 순서 : (모델 이름, 질문 벡터) 두 개
    # - ANALYZE : 실제로 실행해 행 수·시간 측정
    # - BUFFERS : 읽은 버퍼(페이지) 수 표시
    # 주의: EXPLAIN의 cost는 ms가 아닌 상대 비용이다. 인덱스 사용 여부는 계획을 읽어서만 말한다
    # [강사용 정답] D23-06-I02 · explain_sql
    #     SELECT id FROM day23_search_docs를 거리순으로 검색하는
    #     EXPLAIN (ANALYZE, BUFFERS) SQL을 작성하세요. 바인딩은 %s 두 개.
    #     인덱스 사용 여부를 임의로 주장하면 안 됩니다.
    # 해설 D23-06-I02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return """EXPLAIN (ANALYZE, BUFFERS)
    SELECT id FROM day23_search_docs
    WHERE embedding_model=%s
    ORDER BY embedding <=> %s ASC LIMIT 3"""
# END D23-06-I02

# ============================================================
# 4. 실행 계획에서 Scan 종류 판독
# ============================================================
# BEGIN D23-06-I03
def read_scan_type(plan_lines):
    # EXPLAIN 결과 줄 목록에서 Index Scan / Seq Scan / 그 밖을 구분해 이름 하나로 돌려준다.
    # - 'Index Scan' 포함 : 'index'
    # - 'Seq Scan' 포함   : 'seq'  (9건 같은 작은 테이블에서는 정상)
    # - 계획 없음·그 밖   : 'unknown'
    # 예: ['Seq Scan on day23_search_docs (cost=...)', 'Sort Key: ...'] -> 'seq'
    # [강사용 정답] D23-06-I03 · read_scan_type
    #     PostgreSQL EXPLAIN 텍스트에서 Index Scan / Seq Scan / 기타를 구분하세요.
    #     계획이 없으면 unknown, index가 있으면 index, seq면 seq로 반환합니다.
    #     작은 9건 테이블에서 Seq Scan은 오류가 아닙니다.
    # 해설 D23-06-I03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # ' '.join(plan_lines) : 여러 줄을 한 문자열로 합쳐 한 번에 검색
    s = ' '.join(plan_lines)
    if 'Index Scan' in s:
        return 'index'
    if 'Seq Scan' in s:
        return 'seq'
    return 'unknown'
# END D23-06-I03

# ============================================================
# 5. 거리 연산자와 인덱스 연산자 클래스 일치 확인
# ============================================================
# BEGIN D23-06-I04
def check_operator_compatibility(sql, operator_class):
    # SQL이 <=>(Cosine)를 쓰면 인덱스 연산자 클래스가 vector_cosine_ops 인지 확인한다.
    # - <=> 사용 + vector_cosine_ops : True
    # - <=> 사용 + 다른 클래스       : ValueError('operator_class_mismatch')  예: vector_l2_ops
    # 연산자 ↔ 연산자 클래스 : <=> Cosine ↔ vector_cosine_ops, <-> L2 ↔ vector_l2_ops, <#> 음의 내적 ↔ vector_ip_ops
    # [강사용 정답] D23-06-I04 · check_operator_compatibility
    #     SQL이 cosine <=> 를 사용하면 vector_cosine_ops가 맞는지 검사하세요.
    #     불일치 시 ValueError, 정상 시 True.
    #     L2 / negative inner product와 혼동하지 않습니다.
    # 해설 D23-06-I04: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    if '<=>' in sql and operator_class != 'vector_cosine_ops':
        raise ValueError('operator_class_mismatch')
    return True
# END D23-06-I04

# ============================================================
# 6. Exact Top-k와 ANN Top-k 일치율
# ============================================================
# BEGIN D23-06-I05
def compare_topk(exact_ids, ann_ids):
    # Exact 결과 ID 중 ANN 결과에도 있는 비율(겹치는 ID 수 / Exact ID 수)을 계산한다.
    # - exact_ids 비어 있음 : None (비교 기준 없음)
    # - 정상               : 0~1 float
    # 예: exact [102, 106, 107], ann [102, 107, 101] -> 겹침 {102, 107} 2개 / 3 = 0.6667
    # [강사용 정답] D23-06-I05 · compare_topk
    #     Exact Top-k와 ANN Top-k의 ID 집합 일치율을 계산하세요.
    #     겹치는 ID 수 / exact_ids 수. 빈 exact_ids는 None.
    #     인덱스 속도나 모델 품질을 만들어 내면 안 됩니다.
    # 해설 D23-06-I05: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    if not exact_ids:
        return None
    # set(a) & set(b) : 두 집합의 교집합
    return len(set(exact_ids) & set(ann_ids)) / len(set(exact_ids))
# END D23-06-I05
