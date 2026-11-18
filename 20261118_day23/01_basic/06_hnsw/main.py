"""Day23 실습 06 · Exact·HNSW·EXPLAIN.

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 06 기본필수: Exact Search와 HNSW의 원리·정렬·한계를 말로 설명합니다.
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
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (E5_MODEL_ID)
from day23kit.data import E5_MODEL_ID
# 제공 모듈 common/day23kit/embedding.py: Encoder 선택(fixture 모의 / live 실제 E5)·문서·질문 인코딩·벡터 계약 검사 (get_encoder, encode_question)
from day23kit.embedding import get_encoder, encode_question
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (connect_db, exact_topk, check_data_ready, HNSW_SQL)
from day23kit.db import connect_db, exact_topk, check_data_ready, HNSW_SQL
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result


# ============================================================
# 2. HNSW 인덱스 정의 SQL
# ============================================================
# BEGIN D23-06-B01
def create_index_sql():
    # day23_search_docs.embedding 열에 Cosine 거리용 HNSW 인덱스를 만드는 SQL을 돌려준다.
    # - USING hnsw               : 근사 최근접(ANN) 그래프 인덱스
    # - vector_cosine_ops        : <=>(Cosine distance) 연산자와 짝인 연산자 클래스
    # - IF NOT EXISTS            : 이미 있으면 다시 만들지 않음
    # 주의: 인덱스는 선택 항목이며 9건 데이터에서 빨라진다는 보장이 없다
    # [강사용 정답] D23-06-B01 · create_index_sql
    #     day23_search_docs.embedding의 Cosine HNSW 인덱스를 정의하세요.
    #     USING hnsw (embedding vector_cosine_ops)를 사용합니다.
    #     인덱스는 선택 항목이며 9건에서 빠른 성능을 보장하지 않습니다.
    # 해설 D23-06-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # HNSW_SQL : 제공 상수 (common/day23kit/db.py)
    return HNSW_SQL
# END D23-06-B01

# ============================================================
# 3. 실행 계획 확인 SQL (EXPLAIN)
# ============================================================
# BEGIN D23-06-B02
def explain_sql():
    # 거리순 Top-3 검색의 실제 실행 계획을 보는 EXPLAIN (ANALYZE, BUFFERS) SQL을 돌려준다.
    # %s 바인딩 순서 : (모델 이름, 질문 벡터) 두 개
    # - ANALYZE : 실제로 실행해 행 수·시간 측정
    # - BUFFERS : 읽은 버퍼(페이지) 수 표시
    # 주의: EXPLAIN의 cost는 ms가 아닌 상대 비용이다. 인덱스 사용 여부는 계획을 읽어서만 말한다
    # [강사용 정답] D23-06-B02 · explain_sql
    #     SELECT id FROM day23_search_docs를 거리순으로 검색하는
    #     EXPLAIN (ANALYZE, BUFFERS) SQL을 작성하세요. 바인딩은 %s 두 개.
    #     인덱스 사용 여부를 임의로 주장하면 안 됩니다.
    # 해설 D23-06-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return """EXPLAIN (ANALYZE, BUFFERS)
    SELECT id FROM day23_search_docs
    WHERE embedding_model=%s
    ORDER BY embedding <=> %s ASC LIMIT 3"""
# END D23-06-B02

# ============================================================
# 4. 실행 계획에서 Scan 종류 판독
# ============================================================
# BEGIN D23-06-B03
def read_scan_type(plan_lines):
    # EXPLAIN 결과 줄 목록에서 Index Scan / Seq Scan / 그 밖을 구분해 이름 하나로 돌려준다.
    # - 'Index Scan' 포함 : 'index'
    # - 'Seq Scan' 포함   : 'seq'  (9건 같은 작은 테이블에서는 정상)
    # - 계획 없음·그 밖   : 'unknown'
    # 예: ['Seq Scan on day23_search_docs (cost=...)', 'Sort Key: ...'] -> 'seq'
    # [강사용 정답] D23-06-B03 · read_scan_type
    #     PostgreSQL EXPLAIN 텍스트에서 Index Scan / Seq Scan / 기타를 구분하세요.
    #     계획이 없으면 unknown, index가 있으면 index, seq면 seq로 반환합니다.
    #     작은 9건 테이블에서 Seq Scan은 오류가 아닙니다.
    # 해설 D23-06-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # ' '.join(plan_lines) : 여러 줄을 한 문자열로 합쳐 한 번에 검색
    s = ' '.join(plan_lines)
    if 'Index Scan' in s:
        return 'index'
    if 'Seq Scan' in s:
        return 'seq'
    return 'unknown'
# END D23-06-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # HNSW 인덱스·EXPLAIN SQL을 기록하고 실행 계획의 Scan 종류를 판독한다.
    # - mode='fixture' : DB 미접속. 미리 적어 둔 예시 계획 두 줄로 판독 연습 (plan_is_simulated=True)
    # - mode='live'    : 실제 DB에서 EXPLAIN 실행 (--create-index 를 줄 때만 HNSW 실제 생성)
    # - level          : 2 이상은 연산자 클래스·일치율 예시, 3은 Exact/강제 인덱스 계획 비교(live)와 필터 진단
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 compare_topk 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. SQL 준비
    # ----------------------------------------------------
    sql_index = create_index_sql()
    sql_plan = explain_sql()
    result = dict(exercise = 6, level = level, source_mode = 'simulated' if mode == 'fixture' else 'pgvector_live',
                  index_sql = sql_index, explain_sql = sql_plan)
    # ----------------------------------------------------
    # 5-2. fixture: 예시 계획 판독 / live: 실제 계획 판독
    # ----------------------------------------------------
    if mode == 'fixture':
        # sample : 실제 DB 결과가 아니라 판독 연습용으로 적어 둔 계획 두 줄 -> 'seq'
        sample = ['Seq Scan on day23_search_docs (cost=...)', 'Sort Key: (embedding <=> ...)']
        result['scan_type'] = read_scan_type(sample)
        result['plan_is_simulated'] = True
    else:
        encoder = get_encoder('live')
        qvec = encode_question(encoder, '서버는 어디에서 실행하나요?')
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # check_data_ready : 테이블이 있고 E5 벡터 9건이 저장돼 있는지 확인 (없으면 실습 05 live 먼저)
            check_data_ready(conn)
            # --create-index 를 명시했을 때만 인덱스 생성 후 통계 갱신(ANALYZE)
            if getattr(args, 'create_index', False):
                conn.execute(sql_index)
                conn.execute('ANALYZE day23_search_docs')
            # EXPLAIN 결과는 한 줄씩 행으로 온다 -> r[0] 문자열 목록
            plan_lines = [str(r[0]) for r in conn.execute(sql_plan, (E5_MODEL_ID, qvec)).fetchall()]
            result['scan_type'] = read_scan_type(plan_lines)
            result['explain_lines'] = plan_lines
            # --------------------------------------------
            # 5-3. 심화 live: Exact 기준선과 인덱스 유도 계획 비교
            # --------------------------------------------
            if level >= 3:
                # Exact 기준선과 인덱스 유도 계획을 같은 질문/데이터/필터에서 실제 비교합니다.
                # SET LOCAL은 수업용 트랜잭션 동안만 적용되며 결과의 우열을 단정하지 않습니다.
                # ① 인덱스 스캔을 끄고 Exact(순차 스캔) 결과
                conn.execute('SET LOCAL enable_indexscan=off')
                conn.execute('SET LOCAL enable_bitmapscan=off')
                exact_ids = [h['id'] for h in exact_topk(conn, qvec, k = 3, category = 'development')]
                # ② 인덱스 스캔을 켜고 순차 스캔을 꺼서 인덱스 사용을 유도한 계획·결과
                conn.execute('SET LOCAL enable_indexscan=on')
                conn.execute('SET LOCAL enable_bitmapscan=on')
                conn.execute('SET LOCAL enable_seqscan=off')
                indexed_plan = [str(r[0]) for r in conn.execute(sql_plan, (E5_MODEL_ID, qvec)).fetchall()]
                indexed_ids = [h['id'] for h in exact_topk(conn, qvec, k = 3, category = 'development')]
                conn.execute('SET LOCAL enable_seqscan=on')
                # confirmed_index : 유도한 계획에 실제로 Index Scan이 보였는지
                confirmed_index = (read_scan_type(indexed_plan) == 'index')
                # comparison 키 : 의미
                # - exact_ids / forced_plan_ids : 두 계획의 Top-3 ID
                # - forced_plan_scan            : 유도 계획의 Scan 종류
                # - hnsw_index_observed         : Index Scan 관측 여부
                # - exact_vs_ann_overlap        : 관측됐을 때만 일치율, 아니면 None
                result['comparison'] = {
                    'category': 'development', 'k': 3, 'exact_ids': exact_ids,
                    'forced_plan_ids': indexed_ids,
                    'forced_plan_scan': read_scan_type(indexed_plan),
                    'hnsw_index_observed': confirmed_index,
                    'exact_vs_ann_overlap': compare_topk(exact_ids, indexed_ids) if confirmed_index else None,
                    'note': 'Index Scan이 관측됐을 때만 ANN 비교라 기록합니다. 9건에서 속도 향상 주장 금지.',
                }
    # ----------------------------------------------------
    # 5-4. 응용·심화: 판정 함수 예시
    # ----------------------------------------------------
    # operator_compatible : <=> + vector_cosine_ops -> True
    # example_overlap     : [102, 106, 107] vs [102, 107, 101] -> 0.6666666666666666
    if level >= 2:
        result['operator_compatible'] = check_operator_compatibility(sql_plan, 'vector_cosine_ops')
        result['example_overlap'] = compare_topk([102, 106, 107], [102, 107, 101])
    # ann_filtered_example : k=3, 반환 1, 조건 행 4 -> 'review_ann_filter'
    if level >= 3:
        result['ann_filtered_example'] = filtered_ann_warning(k = 3, returned = 1, exact_count = 4)
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 예시 계획 판독 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 DB에서 EXPLAIN 실행
    # - --create-index : live에서만 HNSW 인덱스를 실제로 만든다 (명시해야만 생성)
    # - 저장 위치      : 이 파일 옆 outputs/exercise_06.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 06 · Exact·HNSW·EXPLAIN')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')

    # action='store_true' : 옵션을 쓰면 True, 안 쓰면 False (args.create_index)
    parser.add_argument('--create-index', action = 'store_true', help = 'live에서만 HNSW 인덱스를 실제 생성')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체 (create_index 사용)
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_06.json', result)
    import json
    # ensure_ascii=False : 한글 그대로 출력 / indent=2 : 두 칸 들여쓰기 / default=str : 변환 불가 값은 문자열로
    print(json.dumps(result, ensure_ascii = False, indent = 2, default = str))
    print('결과 저장:', path)

# ============================================================
# 7. 스크립트 진입점
# ============================================================
if __name__ == '__main__':
    main()
