"""Day23 실습 05 · 원문·벡터 저장과 Top-k.

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 05 기본필수: 준비한 벡터를 저장하고 SQL로 상위 3건의 ID·원문을 반환합니다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
import argparse
import sys
from pathlib import Path
# 공통 모듈 폴더를 import 경로 맨 앞에 추가
# Path(__file__).resolve().parents[2] : <수준>/<교시>_주석/main.py 기준 두 단계 위 = mentee 폴더
# -> mentee/common 을 넣어 day23kit 패키지를 import 할 수 있게 함
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (load_knowledge, passage_text, E5_MODEL_ID, TABLE_NAME)
from day23kit.data import load_knowledge, passage_text, E5_MODEL_ID, TABLE_NAME
# 제공 모듈 common/day23kit/fixtures.py: 교육용 2차원 모의 벡터와 PostgreSQL 없이 도는 모의 Top-k 검색 (fixture_search)
from day23kit.fixtures import fixture_search
# 제공 모듈 common/day23kit/embedding.py: Encoder 선택(fixture 모의 / live 실제 E5)·문서·질문 인코딩·벡터 계약 검사 (get_encoder, encode_passages, encode_question, validate_vectors)
from day23kit.embedding import get_encoder, encode_passages, encode_question, validate_vectors
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (connect_db, ensure_table, upsert_rows, exact_topk, TABLE_SQL, SQL_TOPK)
from day23kit.db import connect_db, ensure_table, upsert_rows, exact_topk, TABLE_SQL, SQL_TOPK
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result


# ============================================================
# 2. 검색 테이블 정의 SQL
# ============================================================
# BEGIN D23-05-B01
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
    # [강사용 정답] D23-05-B01 · schema_sql
    #     기존 knowledge 테이블을 건드리지 않는 day23_search_docs의 CREATE TABLE SQL을 작성하세요.
    #     필드: id(PK),title,content,category,source_version,embedding_model,embedding vector(384).
    #     DROP/TRUNCATE를 사용하지 않습니다.
    # 해설 D23-05-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 강의의 실제 테이블 정의와 동일한 SQL입니다.
    # TABLE_SQL : 제공 상수 (common/day23kit/db.py)
    return TABLE_SQL
# END D23-05-B01

# ============================================================
# 3. 저장할 문서 행 준비
# ============================================================
# BEGIN D23-05-B02
def prepare_document_rows(rows):
    # 지식 101~109 각 행에 E5 문서 입력 passage_input을 더한 새 dict 목록을 같은 순서로 만든다.
    # - rows   : 지식 dict 목록 9건
    # - 반환값 : 원래 키(id·title·content·category·source_version) + passage_input
    # 예: {'id': 101, ..., 'passage_input': 'passage: 실습 파일 보관 실습 파일은 개인 폴더에 보관합니다.'}
    # [강사용 정답] D23-05-B02 · prepare_document_rows
    #     지식 101~109의 ID/제목/본문/분류/버전 및 passage 입력을 순서대로 묶으세요.
    #     반환: [{'id', 'passage_input', ...}, ...]. 같은 ID를 새로 만들지 않습니다.
    # 해설 D23-05-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # {**dict(r), 'passage_input': ...} : 원본 행을 복사한 뒤 키 하나를 추가 (ID를 새로 만들지 않음)
    return [{**dict(r), 'passage_input': passage_text(r)} for r in rows]
# END D23-05-B02

# ============================================================
# 4. Cosine Top-k 검색 SQL
# ============================================================
# BEGIN D23-05-B03
def topk_sql():
    # 질문 벡터와의 Cosine distance(<=>) 오름차순으로 k개를 고르는 SQL을 돌려준다.
    # %s 바인딩 순서 : (질문 벡터, 모델 이름, 질문 벡터, k)
    # - embedding <=> %s AS distance : 거리 값 (0에 가까울수록 가깝다)
    # - WHERE embedding_model=%s     : E5로 만든 벡터만 비교 (다른 공간 혼합 금지)
    # - ORDER BY ... ASC LIMIT %s    : 가까운 순서로 k개
    # [강사용 정답] D23-05-B03 · topk_sql
    #     Cosine distance(<=>) 오름차순, k개 제한 SQL을 작성하세요.
    #     SELECT에 id,title,content,category,source_version,embedding_model,distance 포함.
    #     E5 외 다른 공간 혼합을 막도록 embedding_model을 필터합니다.
    # 해설 D23-05-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # SQL_TOPK : 제공 상수 (common/day23kit/db.py)
    return SQL_TOPK
# END D23-05-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # 테이블·검색 SQL과 저장할 문서 행을 준비하고, 질문 하나로 Top-3 후보를 검색해 기록한다.
    # - mode='fixture' : fixture_search() 모의 Top-k (PostgreSQL 미접속, 교육용 2차원 좌표 거리)
    # - mode='live'    : 실제 E5로 9건 인코딩 -> day23_search_docs 에 upsert -> pgvector Exact Top-k
    # - level          : 1=필터 없음, 2·3=category='development' 필터와 옵션 검사, 3=ID·벡터 계약 검사
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 filtered_sql 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. SQL과 문서 행 준비
    # ----------------------------------------------------
    rows = load_knowledge()
    sql_schema = schema_sql()
    inputs = prepare_document_rows(rows)
    sql_query = topk_sql()
    question = '서버는 어디에서 실행하나요?'
    # result 키 : 의미
    # - source_mode    : 'simulated'(fixture) 또는 'pgvector_exact'(live 실제 SQL)
    # - table / model  : day23_search_docs / intfloat/multilingual-e5-small
    # - prepared_count : 준비한 문서 행 수 9
    # - first_passage  : 첫 문서(101) E5 입력
    # - sql_schema / sql_topk : 위 SQL 원문
    result = dict(exercise = 5, level = level, source_mode = 'simulated' if mode == 'fixture' else 'pgvector_exact',
                  table = TABLE_NAME, model = E5_MODEL_ID, prepared_count = len(inputs),
                  first_passage = inputs[0]['passage_input'],
                  sql_schema = sql_schema, sql_topk = sql_query)
    # category : 기본필수는 필터 없음(None), 응용·심화는 'development'
    category = None if level == 1 else 'development'
    if level >= 2:
        result['sql_filtered'] = filtered_sql()
        result['options'] = check_search_options(3, category)
    # ----------------------------------------------------
    # 5-2. 검색: fixture 모의 또는 live 실제 저장·검색
    # ----------------------------------------------------
    if mode == 'fixture':
        # fixture_search(질문, k, category) : 제공 모의 검색. 결과 행마다 source_mode='simulated'
        hits = fixture_search(question, 3, category = category)
    else:
        # live 순서 : 실제 E5 로드 -> 문서 9건 인코딩 -> 벡터 계약 검사 -> 질문 인코딩 -> DB 저장 -> Exact Top-k
        encoder = get_encoder('live')
        vectors = encode_passages(encoder, rows)
        # 심화는 ID·버전까지 포함한 검사, 그 외는 벡터 계약만 검사 (저장 전에 멈추기 위해)
        if level >= 3:
            check_id_vector_integrity(rows, vectors)
        else:
            validate_vectors(vectors, len(rows))
        qvec = encode_question(encoder, question)
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # ensure_table : CREATE TABLE IF NOT EXISTS 후 embedding 열이 vector(384)인지 확인
            ensure_table(conn)
            # upsert_rows : 같은 ID가 있으면 UPDATE, 없으면 INSERT (ON CONFLICT (id) DO UPDATE)
            upsert_rows(conn, rows, vectors, E5_MODEL_ID)
            hits = exact_topk(conn, qvec, k = 3, category = category)
    # hits 요약 : ID·본문·거리만 기록 (fixture 기본필수 예: [102 0.0, 106 0.001896, 107 0.008496])
    result['hits'] = [{'id': h['id'], 'content': h['content'], 'distance': round(h['distance'], 6)} for h in hits]
    # ----------------------------------------------------
    # 5-3. 심화: ID·벡터 계약 결과 기록
    # ----------------------------------------------------
    if level >= 3:
        # 이전 단계의 벡터를 재사용합니다. live에서는 같은 모델을 다시 다운로드/로드하지 않습니다.
        # fixture는 앞에서 벡터를 만들지 않았으므로 모의 Encoder로 9건 벡터를 만든다
        if mode == 'fixture':
            encoder = get_encoder('fixture')
            vectors = encode_passages(encoder, rows)
        result['id_vector_integrity'] = check_id_vector_integrity(rows, vectors)
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_05.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 05 · 원문·벡터 저장과 Top-k')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_05.json', result)
    import json
    # json.dumps 인자
    # - ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 출력
    # - indent=2           : 사람이 읽기 쉽게 두 칸 들여쓰기
    # - default=str        : JSON으로 바꿀 수 없는 값(Path 등)은 문자열로 변환
    print(json.dumps(result, ensure_ascii = False, indent = 2, default = str))
    print('결과 저장:', path)

# ============================================================
# 7. 스크립트 진입점
# ============================================================
if __name__ == '__main__':
    main()
