"""Day23 실습 05 · 응용 · tasks

응용 수준의 TODO 함수 5개를 모은 모듈이다. main.py가 import 해서 run()에서 호출한다.
day23_search_docs 테이블 정의, 저장할 문서 행 준비, Cosine Top-k·category 필터 SQL, 검색 옵션 검사를 담당한다.

이 파일의 TODO 함수:
- schema_sql (D23-05-I01): 기존 knowledge 테이블을 건드리지 않는 day23_search_docs의 CREATE TABLE SQL을 작성하세요.
- prepare_document_rows (D23-05-I02): 지식 101~109의 ID/제목/본문/분류/버전 및 passage 입력을 순서대로 묶으세요.
- topk_sql (D23-05-I03): Cosine distance(<=>) 오름차순, k개 제한 SQL을 작성하세요.
- filtered_sql (D23-05-I04): category 필터가 포함된 Top-k SQL을 작성하세요.
- check_search_options (D23-05-I05): k는 bool을 제외한 정수 1~9, category는 None 또는 course/development/support인지 검사하세요.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (passage_text)
from day23kit.data import passage_text
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (TABLE_SQL, SQL_TOPK, SQL_CATEGORY)
from day23kit.db import TABLE_SQL, SQL_TOPK, SQL_CATEGORY

# ============================================================
# 2. 검색 테이블 정의 SQL
# ============================================================
# BEGIN D23-05-I01
def schema_sql():
    # 기존 knowledge 테이블은 건드리지 않고 Day23 전용 day23_search_docs 테이블을 만드는 SQL을 돌려준다.
    # 테이블 열 : 의미
    # - id integer PRIMARY KEY : 지식 ID (같은 ID는 한 행만)
    # - title / content        : 원문 제목·본문 (벡터와 같은 행에 원문 보관)
    # - category               : course / development / support (Metadata 필터용)
    # - source_version         : 원문 버전
    # - embedding_model        : 벡터를 만든 모델 이름 (다른 벡터 공간 혼합 방지)
    # - embedding vector(384)  : E5-small 384차원 벡터
    # 주의: CREATE TABLE IF NOT EXISTS 만 사용한다. DROP·TRUNCATE 없음
    # [강사용 정답] D23-05-I01 · schema_sql
    #     기존 knowledge 테이블을 건드리지 않는 day23_search_docs의 CREATE TABLE SQL을 작성하세요.
    #     필드: id(PK),title,content,category,source_version,embedding_model,embedding vector(384).
    #     DROP/TRUNCATE를 사용하지 않습니다.
    # 해설 D23-05-I01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 강의의 실제 테이블 정의와 동일한 SQL입니다.
    # TABLE_SQL : 제공 상수 (common/day23kit/db.py)
    return TABLE_SQL
# END D23-05-I01

# ============================================================
# 3. 저장할 문서 행 준비
# ============================================================
# BEGIN D23-05-I02
def prepare_document_rows(rows):
    # 지식 101~109 각 행에 E5 문서 입력 passage_input을 더한 새 dict 목록을 같은 순서로 만든다.
    # - rows   : 지식 dict 목록 9건
    # - 반환값 : 원래 키(id·title·content·category·source_version) + passage_input
    # 예: {'id': 101, ..., 'passage_input': 'passage: 실습 파일 보관 실습 파일은 개인 폴더에 보관합니다.'}
    # [강사용 정답] D23-05-I02 · prepare_document_rows
    #     지식 101~109의 ID/제목/본문/분류/버전 및 passage 입력을 순서대로 묶으세요.
    #     반환: [{'id', 'passage_input', ...}, ...]. 같은 ID를 새로 만들지 않습니다.
    # 해설 D23-05-I02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # {**dict(r), 'passage_input': ...} : 원본 행을 복사한 뒤 키 하나를 추가 (ID를 새로 만들지 않음)
    return [{**dict(r), 'passage_input': passage_text(r)} for r in rows]
# END D23-05-I02

# ============================================================
# 4. Cosine Top-k 검색 SQL
# ============================================================
# BEGIN D23-05-I03
def topk_sql():
    # 질문 벡터와의 Cosine distance(<=>) 오름차순으로 k개를 고르는 SQL을 돌려준다.
    # %s 바인딩 순서 : (질문 벡터, 모델 이름, 질문 벡터, k)
    # - embedding <=> %s AS distance : 거리 값 (0에 가까울수록 가깝다)
    # - WHERE embedding_model=%s     : E5로 만든 벡터만 비교 (다른 공간 혼합 금지)
    # - ORDER BY ... ASC LIMIT %s    : 가까운 순서로 k개
    # [강사용 정답] D23-05-I03 · topk_sql
    #     Cosine distance(<=>) 오름차순, k개 제한 SQL을 작성하세요.
    #     SELECT에 id,title,content,category,source_version,embedding_model,distance 포함.
    #     E5 외 다른 공간 혼합을 막도록 embedding_model을 필터합니다.
    # 해설 D23-05-I03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # SQL_TOPK : 제공 상수 (common/day23kit/db.py)
    return SQL_TOPK
# END D23-05-I03

# ============================================================
# 5. category 필터 Top-k SQL
# ============================================================
# BEGIN D23-05-I04
def filtered_sql():
    # Top-k SQL에 category 조건을 더한 SQL을 돌려준다.
    # %s 바인딩 순서 : (질문 벡터, 모델 이름, category, 질문 벡터, k)
    # 주의: 값은 문자열 합치기가 아니라 %s 바인딩으로 넘긴다 (SQL 주입 방지)
    # [강사용 정답] D23-05-I04 · filtered_sql
    #     category 필터가 포함된 Top-k SQL을 작성하세요.
    #     쿼리 벡터와 category, 모델, k는 문자열 합치기가 아닌 %s 바인딩입니다.
    # 해설 D23-05-I04: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # SQL_CATEGORY : 제공 상수 (common/day23kit/db.py)
    return SQL_CATEGORY
# END D23-05-I04

# ============================================================
# 6. 검색 옵션(k·category) 검사
# ============================================================
# BEGIN D23-05-I05
def check_search_options(k, category):
    # 검색 전에 k와 category 값이 허용 범위인지 확인한다.
    # - k가 int가 아니거나(bool 포함) 1~9 밖 : ValueError('invalid_k')     예: 0, 10, True, '3'
    # - category가 허용 값이 아님            : ValueError('invalid_category') 예: 'sales'
    # - 정상                                 : {'k': k, 'category': category}  예: {'k': 3, 'category': 'development'}
    # [강사용 정답] D23-05-I05 · check_search_options
    #     k는 bool을 제외한 정수 1~9, category는 None 또는 course/development/support인지 검사하세요.
    #     잘못된 값은 ValueError. 정상 시 {'k':k,'category':category} 반환.
    # 해설 D23-05-I05: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # type(k) is not int : bool(True)은 int 하위 타입이라 isinstance로는 통과하므로 type으로 엄격히 검사
    if type(k) is not int or not 1 <= k <= 9:
        raise ValueError('invalid_k')
    # category=None : 필터 없음 (전체 9건에서 검색)
    if category is not None and category not in ('course', 'development', 'support'):
        raise ValueError('invalid_category')
    return {'k': k, 'category': category}
# END D23-05-I05
