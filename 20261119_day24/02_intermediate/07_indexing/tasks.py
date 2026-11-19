"""실습 07 Intermediate · --live-db 필수: 실제 DB 결과를 확인할 때.

실습 07 pgvector Indexing · 응용
- 역할: DOC-ASSIGN 청크를 준비하고, --live-db 에서 E5 임베딩 -> Stage -> Publish -> Top-k 검색을 짧은 Transaction 단위로 실행한다
- 기능: prepare()(I01) 청크 준비, embed()(I02) passage 임베딩, stage()(I03) Staging 저장,
        publish_search()(I04) 공개 전환과 query 검색
- 결과: outputs/indexing_result.json 
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
from coursekit.io import save_json, show_success
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import by_doc, parse_one
# 제공 모듈 coursekit/processing.py: 정제·문자 창 Chunking·Span 검증·Coverage·안정 chunk_id·E5 입력 문자열
from coursekit.processing import make_chunks

# ============================================================
# 2. 색인 입력 청크 준비 (TODO D24-07-I01)
# ============================================================
# 문서의 모든 ParsedUnit 을 정제·분할해 같은 출처·버전 계약의 청크 목록을 만든다.
# - 'DOC-ASSIGN' : (manifest dict, 청크 1건)
# - parsed 가 아닌 문서 : ValueError(status) (예: 'needs_ocr')
# - 청크 0건     : ValueError('empty chunk set')
def prepare(doc_id: str) -> tuple[dict, list[dict]]:
    # BEGIN D24-07-I01
    # 개념·실무 해설(mentor 원문): Indexing 입력은 검증된 Chunk의 내용과 출처입니다. 임의로 만든 384개 숫자를 E5 Embedding이라 부르지 않습니다.
    # TODO D24-07-I01 완료: 여러 ParsedUnit을 정제·분할하고 입력 청크의 출처·버전 계약을 통일합니다.
    doc = by_doc(doc_id)
    parsed = parse_one(doc)
    if parsed['status'] != 'parsed':
        raise ValueError(parsed['status'])
    chunks = []
    # unit 마다 make_chunks(size = 110, overlap = 15) 결과를 이어 붙임
    for unit in parsed['units']:
        chunks.extend(make_chunks(doc, unit, size = 110, overlap = 15))
    if not chunks:
        raise ValueError('empty chunk set')
    return (doc, chunks)
    # END D24-07-I01

# ============================================================
# 3. E5 passage 임베딩 (TODO D24-07-I02)
# ============================================================
# 실제 E5 Encoder 로 청크마다 384차원 정규화 벡터를 만든다 (--live-db 에서만 호출).
# - encode_passages 검사: 'passage: ' 접두어, 토큰 길이 <= max_seq_length, shape (청크 수, 384), 유한값, 길이 1
# - 하나라도 어기면 ValueError (DB 에 넣기 전에 중단)
def embed(model, chunks: list[dict]):
    # BEGIN D24-07-I02
    # 개념·실무 해설(mentor 원문): E5 모델은 한 번 로드해서 CPU로 추론합니다. passage와 query의 접두어, 모델 ID, 정규화, 차원을 동일 계약으로 유지합니다.
    # TODO D24-07-I02 완료: 실제 E5 Encoder로 384차원 passage 벡터를 만들고 개수·길이·유한값을 확인합니다.
    # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
    from coursekit.live import encode_passages
    return encode_passages(model, chunks)
    # END D24-07-I02

# ============================================================
# 4. Staging 저장 (TODO D24-07-I03)
# ============================================================
# 짧은 Transaction 하나에서 청크·벡터를 'staging' 버전으로 저장한다 (아직 검색에 노출되지 않음).
# - 같은 chunk_id 에 다른 본문이 있으면 stage_version 이 ValueError -> Transaction 전체 Rollback
def stage(conn, doc: dict, chunks: list[dict], vectors) -> None:
    # BEGIN D24-07-I03
    # 개념·실무 해설(mentor 원문): Stage는 아직 조회에 노출되지 않은 새 버전을 DB에 저장하는 단계입니다. 같은 Chunk ID의 본문 불일치는 조용히 덮어쓰지 않습니다.
    # TODO D24-07-I03 완료: 짧은 DB Transaction 안에서 Chunk·벡터를 Staging 합니다.
    # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
    from coursekit.db import stage_version
    # with 블록이 예외 없이 끝나면 COMMIT, 예외가 나면 ROLLBACK
    with conn.transaction():
        stage_version(conn, doc, doc['file'], chunks, vectors)
    # END D24-07-I03

# ============================================================
# 5. 공개 전환과 query 검색 (TODO D24-07-I04)
# ============================================================
# 검증된 버전만 Publish(현재 포인터 변경)하고, 같은 모델의 query 벡터로 Top-3 를 검색한다.
# - publish_version : Stage 된 청크 수 = expected_count 일 때만 'ready' 로 바꾸고 포인터 변경
# - search_current  : 현재 공개 버전 + category '학사' 청크만 Cosine 거리 오름차순 3건
def publish_search(conn, doc: dict, chunks: list[dict], model) -> list[tuple]:
    # BEGIN D24-07-I04
    # 개념·실무 해설(mentor 원문): Publish는 staging된 대상의 수와 상태를 검증하고 현재 포인터 쌍을 변경하는 짧은 Transaction으로 처리합니다.
    # TODO D24-07-I04 완료: 검증된 버전만 Publish하고 동일 계약의 query: 벡터로 Top-k를 조회합니다.
    # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
    from coursekit.db import publish_version, search_current
    # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
    from coursekit.live import encode_query
    with conn.transaction():
        publish_version(conn, doc['doc_id'], doc['source_version'], chunks[0]['pipeline_version'])
    # 질의는 'query: ' 접두어, 문서는 'passage: ' 접두어 (E5 입력 계약)
    qvec = encode_query(model, '과제 파일은 어디에 보관하나요?')
    return search_current(conn, qvec, category = '학사', k = 3)
    # END D24-07-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# 기본: 청크 준비 결과만 저장 / --live-db: 임베딩·저장·공개·검색 후 결과 저장
def main():
    doc, chunks = prepare('DOC-ASSIGN')
    # result 키 : 의미
    # - source_mode : 'prepared_only_not_db' -> live 성공 시 'e5_pgvector_live'
    # - doc_id      : 'DOC-ASSIGN'
    # - chunk_count : 준비한 청크 수
    # - stored      : 'not_run' -> live 성공 시 저장 청크 수
    # - retrieved   : 'not_run' -> live 성공 시 검색 결과 목록
    result = {
        'source_mode': 'prepared_only_not_db',
        'doc_id': doc['doc_id'],
        'chunk_count': len(chunks),
        'stored': 'not_run',
        'retrieved': 'not_run',
    }

    # ----------------------------------------------------
    # 6-1. --live-db 실제 실행
    # ----------------------------------------------------
    if '--live-db' in sys.argv:
        # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
        from coursekit.live import load_encoder, connect_db
        # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
        from coursekit.db import initialize, use_schema
        model = load_encoder()
        # 주의: 임베딩은 DB 연결 전에 계산 (모델 추론 동안 Transaction 을 열어 두지 않음)
        vectors = embed(model, chunks)
        with connect_db() as conn:
            # 응용 07 전용 Schema d24_inter07 사용
            use_schema(conn, 'd24_inter07')
            initialize(conn)
            stage(conn, doc, chunks, vectors)
            hits = publish_search(conn, doc, chunks, model)
            # retrieved 항목 키 : 의미
            # - citation_id    : r[0] DB 정수 인용 번호
            # - chunk_id       : r[1]
            # - body           : r[2]
            # - source_locator : r[3]
            # - distance       : r[6] Cosine 거리
            result.update(
                source_mode = 'e5_pgvector_live',
                stored = len(chunks),
                retrieved = [
                    {
                        'citation_id': r[0],
                        'chunk_id': r[1],
                        'body': r[2],
                        'source_locator': r[3],
                        'distance': r[6],
                    }
                    for r in hits
                ],
            )
    out = save_json(HERE, 'indexing_result.json', result)
    show_success('07 Intermediate', out)
