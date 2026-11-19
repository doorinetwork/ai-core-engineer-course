"""실습 07 Advanced · --live-db에서 PostgreSQL 실제 버전 전환/회귀.

실습 07 pgvector Indexing · 심화
- 역할: DOC-ATTEND v1·v2 를 같은 Parser 로 준비하고, --live-db 에서 v1 공개 -> v2 Stage -> 의도적 실패 Rollback -> v2 공개·검색 회귀를 실제 DB 로 확인한다
- 기능: prepare_revision()(A01) 판별 청크 준비, embeddings_before_db()(A02) DB 전 임베딩, stage_revision()(A03) Stage,
        current_pointer()(A04) 공개 포인터 조회, test_rollback()(A05) Rollback 검사, publish_and_verify()(A06) 공개·검색 검증
- 결과: outputs/version_regression.json 
- 실행: python main.py            (DB·모델 없이 청크 준비만)
- 실제: python main.py --live-db  (E5 모델·PostgreSQL 18.6 + pgvector·.env 필요, Schema d24_adv07)
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# HERE.parents[1] : <수준>/<실습>_주석 기준 두 단계 위 = mentee 폴더
# -> mentee 를 import 경로 맨 앞에 넣어 coursekit 패키지를 불러온다
sys.path.insert(0, str(HERE.parents[1]))
# 제공 모듈 coursekit/io.py: 공통 경로(DATA)·JSON 읽기/저장·완료 메시지
from coursekit.io import DATA, save_json, show_success
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import by_doc, parse_one, parse_markdown_units
# 제공 모듈 coursekit/processing.py: 정제·문자 창 Chunking·Span 검증·Coverage·안정 chunk_id·E5 입력 문자열
from coursekit.processing import make_chunks

# ============================================================
# 2. 원문 판별 청크 준비 (TODO D24-07-A01)
# ============================================================
# DOC-ATTEND 의 v1(data/raw) 또는 v2(data/versions) 를 같은 Markdown Parser 로 청킹한다.
# - 'v1' : parse_one(doc) -> sec-1·sec-2 청크
# - 'v2' : attendance_v2.md 를 parse_markdown_units 로 파싱, source_version 'v2'
# - 그 밖 : ValueError('unsupported source version')
def prepare_revision(doc_id: str, version: str) -> tuple[dict, list[dict]]:
    # BEGIN D24-07-A01
    # 개념·실무 해설(mentor 원문): Indexing 입력은 검증된 Chunk의 내용과 출처입니다. 임의로 만든 384개 숫자를 E5 Embedding이라 부르지 않습니다.
    # TODO D24-07-A01 완료: DOC-ATTEND v1·v2를 동일한 Markdown Parser로 처리하고, 문서 ID·원문 버전·처리 설정을 보존합니다.
    # .copy() : manifest dict 를 복사해 source_version 을 바꿔도 원본 조회 결과에 영향이 없게 함
    doc = by_doc(doc_id).copy()
    if version not in {'v1', 'v2'}:
        raise ValueError('unsupported source version')
    if version == 'v1':
        parsed = parse_one(doc)
    else:
        # 개정본 원문 data/versions/attendance_v2.md
        src = (DATA / 'versions' / 'attendance_v2.md').read_text(encoding = 'utf-8')
        # 개정 v2도 v1과 동일한 Markdown Parser를 사용합니다.
        # 문서 구조(# 제목/## 절)가 바뀌지 않았으므로 원문 버전만 구분합니다.
        doc['source_version'] = 'v2'
        parsed = {'status': 'parsed', 'units': parse_markdown_units(src, 'attendance_v2.md')}
    if parsed['status'] != 'parsed':
        raise ValueError(parsed['status'])
    chunks = []
    # unit 마다 110자/15자 겹침 청크
    for u in parsed['units']:
        chunks.extend(make_chunks(doc, u, 110, 15))
    return (doc, chunks)
    # END D24-07-A01

# ============================================================
# 3. DB 전 임베딩 (TODO D24-07-A02)
# ============================================================
# DB Transaction 을 열기 전에 E5 passage 벡터를 계산하고 개수를 확인한다.
# - 벡터 수 != 청크 수 : ValueError('vector count mismatch')
def embeddings_before_db(model, chunks: list[dict]):
    # BEGIN D24-07-A02
    # 개념·실무 해설(mentor 원문): E5 모델은 한 번 로드해서 CPU로 추론합니다. passage와 query의 접두어, 모델 ID, 정규화, 차원을 동일 계약으로 유지합니다.
    # TODO D24-07-A02 완료: E5 passage Embedding을 DB Transaction 전에 계산·검증합니다.
    # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
    from coursekit.live import encode_passages
    vectors = encode_passages(model, chunks)
    if len(chunks) != len(vectors):
        raise ValueError('vector count mismatch')
    return vectors
    # END D24-07-A02

# ============================================================
# 4. 새 판 Stage (TODO D24-07-A03)
# ============================================================
# 짧은 Transaction 에서 청크·벡터를 'staging' 상태로 저장한다 (검색에는 아직 노출되지 않음).
# - 같은 chunk_id 에 다른 본문 : stage_version 이 ValueError -> Rollback
def stage_revision(conn, doc: dict, chunks: list[dict], vectors) -> None:
    # BEGIN D24-07-A03
    # 개념·실무 해설(mentor 원문): Stage는 아직 조회에 노출되지 않은 새 버전을 DB에 저장하는 단계입니다. 같은 Chunk ID의 본문 불일치는 조용히 덮어쓰지 않습니다.
    # TODO D24-07-A03 완료: 벡터·원문·출처를 새 버전으로 Stage하고 ID 충돌을 검사합니다.
    # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
    from coursekit.db import stage_version
    with conn.transaction():
        stage_version(conn, doc, doc['file'], chunks, vectors)
    # END D24-07-A03

# ============================================================
# 5. 공개 포인터 조회 (TODO D24-07-A04)
# ============================================================
# ingest_documents 에서 문서의 (current_version, current_pipeline_version) 을 읽는다.
# - 예: ('v1', 'ingest-e5-v1') / 아직 공개 전: (None, None) / 문서 행 없음: None
# 주의: 조회만 한다. Publish 는 A06 Transaction 에서 수행
def current_pointer(conn, doc_id: str) -> tuple | None:
    # BEGIN D24-07-A04
    # 개념·실무 해설(mentor 원문): 현재 포인터를 읽는 SELECT는 데이터를 변경하지 않습니다. 실제 Ready 검증·Publish는 A06의 Transaction에서 수행합니다.
    # TODO D24-07-A04 완료: 현재 문서의 공개 포인터(source_version, pipeline_version)를 조회합니다. Publish는 A06에서 처리합니다.
    # %s : psycopg 파라미터 바인딩 자리. (doc_id,) 튜플 값이 안전하게 들어감 (문자열 이어 붙이기 금지)
    return conn.execute('SELECT current_version,current_pipeline_version FROM ingest_documents '
                            'WHERE doc_id=%s', (doc_id,)).fetchone()
    # END D24-07-A04

# ============================================================
# 6. 의도적 실패 Rollback 검사 (TODO D24-07-A05)
# ============================================================
# Publish 직후 일부러 예외를 일으켜 Transaction 이 Rollback 되고 공개 포인터가 그대로인지 확인한다.
# - before == after : rollback_ok True
# - 다르면          : ValueError('rollback failed: pointer changed')
def test_rollback(conn, doc: dict, chunks: list[dict]) -> dict:
    # BEGIN D24-07-A05
    # 개념·실무 해설(mentor 원문): 의도적 오류는 학생 개인의 ingest_ 전용 Schema 안에서만 실행합니다. Rollback 이후 current pointer가 바뀌지 않은지 확인합니다.
    # TODO D24-07-A05 완료: 강제로 예외를 발생시킨 Transaction에서 Rollback 후 현재 포인터가 유지되는지 확인합니다.
    # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
    from coursekit.db import publish_version
    # 실패 시도 전 포인터 (예: ('v1', 'ingest-e5-v1'))
    before = current_pointer(conn, doc['doc_id'])
    try:
        with conn.transaction():
            publish_version(
                conn,
                doc['doc_id'],
                doc['source_version'],
                chunks[0]['pipeline_version'],
            )
            # 커밋 전에 예외 -> with conn.transaction() 이 ROLLBACK
            raise RuntimeError('injected_rollback_before_commit')
    except RuntimeError as exc:
        # 주입한 예외가 아닌 다른 RuntimeError 는 숨기지 않고 다시 발생
        if str(exc) != 'injected_rollback_before_commit':
            raise
    after = current_pointer(conn, doc['doc_id'])
    if before != after:
        raise ValueError('rollback failed: pointer changed')
    return {'rollback_ok': True, 'pointer_before': before, 'pointer_after': after}
    # END D24-07-A05

# ============================================================
# 7. v2 공개와 검색 검증 (TODO D24-07-A06)
# ============================================================
# v2 를 Publish 하고 포인터가 v2 로 바뀌었는지, 검색 결과에 이 문서의 v1 청크가 섞이지 않는지 확인한다.
# - 포인터 불일치      : ValueError('publish pointer mismatch')
# - 검색에 v1 청크 있음 : ValueError('stale version in current search')
def publish_and_verify(conn, doc: dict, chunks: list[dict], model) -> dict:
    # BEGIN D24-07-A06
    # 개념·실무 해설(mentor 원문): v2 공개 후에는 이전 v1 청크가 현재 검색 결과에 섞이지 않아야 합니다. 반환된 원문·ID·버전을 확인합니다. 이 예제는 청크 수 감소를 검증하지 않습니다.
    # TODO D24-07-A06 완료: v2를 짧은 Transaction에서 Publish하고, 정확 검색에서 이전 v1 청크가 제외되는지 확인합니다.
    # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
    from coursekit.db import publish_version, search_current
    # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
    from coursekit.live import encode_query
    with conn.transaction():
        publish_version(conn, doc['doc_id'], doc['source_version'], chunks[0]['pipeline_version'])
    pointer = current_pointer(conn, doc['doc_id'])
    if pointer != (doc['source_version'], chunks[0]['pipeline_version']):
        raise ValueError('publish pointer mismatch')
    # 질의 'query: 15분 늦으면 어떻게 기록되나요?' (출석 규정 관련)
    query = encode_query(model, '15분 늦으면 어떻게 기록되나요?')
    hits = search_current(conn, query, '학사', k = 10)
    # h[4] : doc_id, h[5] : source_version (search_current SELECT 순서)
    mine = [h for h in hits if h[4] == doc['doc_id']]
    if any(h[5] != doc['source_version'] for h in mine):
        raise ValueError('stale version in current search')
    # 반환 dict 키 : pointer(공개 포인터), hits_for_this_document(이 문서 검색 건수), source_versions(검색된 판 목록)
    return {
        'pointer': pointer,
        'hits_for_this_document': len(mine),
        'source_versions': [h[5] for h in mine],
    }
    # END D24-07-A06

# ============================================================
# 8. 실행 흐름 (main)
# ============================================================
# 기본: v1·v2 청크 수만 기록 / --live-db: v1 공개 -> v2 Stage -> Rollback 검사 -> v2 공개·검색 검증
def main():
    v1, c1 = prepare_revision('DOC-ATTEND', 'v1')
    v2, c2 = prepare_revision('DOC-ATTEND', 'v2')
    report = {
        'source_mode': 'prepared_only_not_db',
        'v1_chunks': len(c1),
        'v2_chunks': len(c2),
        'db_status': 'not_run',
    }

    # ----------------------------------------------------
    # 8-1. --live-db 실제 실행
    # ----------------------------------------------------
    if '--live-db' in sys.argv:
        # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
        from coursekit.live import connect_db, load_encoder
        # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
        from coursekit.db import initialize, publish_version, use_schema
        model = load_encoder()
        # 두 판의 벡터를 DB 연결 전에 모두 계산
        vec1 = embeddings_before_db(model, c1)
        vec2 = embeddings_before_db(model, c2)
        with connect_db() as conn:
            # 심화 07 전용 Schema d24_adv07
            use_schema(conn, 'd24_adv07')
            initialize(conn)
            stage_revision(conn, v1, c1, vec1)
            # 처음 실행에서는 v1을 정상 공개해 비교 기준을 준비합니다.
            with conn.transaction():
                publish_version(conn, v1['doc_id'], 'v1', c1[0]['pipeline_version'])
            stage_revision(conn, v2, c2, vec2)
            # v2 공개를 일부러 실패시켜 v1 포인터 유지 확인
            report['injected_failure'] = test_rollback(conn, v2, c2)
            # 정상 재시도로 v2 공개 후 검색 결과 판 확인
            report['after_retry'] = publish_and_verify(conn, v2, c2, model)
            report['source_mode'] = 'e5_pgvector_live'
            report['db_status'] = 'verified'
    out = save_json(HERE, 'version_regression.json', report)
    show_success('07 Advanced', out)
