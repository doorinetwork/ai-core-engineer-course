"""실습 07 Basic · 기본 실행은 데이터 계약, --live-db는 실제 E5/DB 실행.

실습 07 pgvector Indexing · 기본필수
- 역할: 검증된 청크 1건을 E5 입력(passage:)으로 준비하고, DB 검색 결과 행을 Citation dict로 바꾸는 계약을 만든다
- 기능: select_index_chunk()(D24-07-B01) 색인할 청크 선택, format_hit()(D24-07-B02) 검색 행 -> dict
- 결과: outputs/indexing.json 
- 실행: python main.py            (DB·모델 없이 데이터 계약만 확인)
- 실제: python main.py --live-db  (E5 모델 다운로드·PostgreSQL 18.6 + pgvector·.env 필요, Schema d24_basic07)
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
# 2. 색인할 청크 1건 선택 (TODO D24-07-B01)
# ============================================================
# 문서를 파싱·청킹해 첫 번째 청크를 고르고 E5 passage 입력인지 확인한다.
# - 'DOC-ASSIGN' : parsed -> 청크 1건 (0~106자, 'passage: 과제 안내...')
# - 파싱 실패 문서 (예: 'DOC-SCAN' needs_ocr) : ValueError('not indexable')
# - 반환값       : (문서 manifest dict, 첫 청크 dict)
def select_index_chunk(doc_id: str) -> tuple[dict, dict]:
    # BEGIN D24-07-B01
    # 개념·실무 해설(mentor 원문): Indexing 입력은 검증된 Chunk의 내용과 출처입니다. 임의로 만든 384개 숫자를 E5 Embedding이라 부르지 않습니다.
    # TODO D24-07-B01 완료: 기존 DOC-ASSIGN 텍스트에서 유효한 청크 1건을 선택하고 E5 passage 입력을 확인합니다.
    # by_doc : manifest.json 에서 doc_id가 정확히 1건인 행 반환 (없거나 중복이면 ValueError)
    doc = by_doc(doc_id)
    # parse_one : 형식별 Parser 결과 {'doc', 'status', 'units', 'reason'}
    parsed = parse_one(doc)
    if parsed['status'] != 'parsed':
        raise ValueError('not indexable')
    # make_chunks 인자 (정제 -> 문자 창 -> 청크 dict 까지 한 번에)
    # - doc                 : 문서 정보
    # - parsed['units'][0]  : 첫 ParsedUnit (TXT는 'text-1' 하나)
    # - size = 110          : 창 최대 110자
    # - overlap = 15        : 인접 창 15자 겹침
    chunks = make_chunks(doc, parsed['units'][0], size = 110, overlap = 15)
    first = chunks[0]
    # 본문이 비었거나 embedding_text가 'passage: '로 시작하지 않으면 E5 입력 계약 위반
    if not first['body'].strip() or not first['embedding_text'].startswith('passage: '):
        raise ValueError('bad embedding input')
    return (doc, first)
    # END D24-07-B01

# ============================================================
# 3. 검색 결과 행을 Citation dict로 (TODO D24-07-B02)
# ============================================================
# DB가 돌려준 튜플 7칸을 이름 있는 dict로 바꾸고 citation_id가 양의 정수인지 확인한다.
# - (201, 'b93c...', '과제 안내...', 'assignment.txt', 'DOC-ASSIGN', 'v1', 0.0) -> dict
# - citation_id 0·음수·문자열 : ValueError('invalid citation ID')
def format_hit(row: tuple) -> dict:
    # DB 행: citation_id,chunk_id,body,locator,doc_id,version,distance.
    # BEGIN D24-07-B02
    # 개념·실무 해설(mentor 원문): 실제 DB가 발급한 citation_id는 양의 정수입니다. 반환 튜플의 필드 순서를 원문·출처·거리와 함께 명시적으로 매핑합니다.
    # TODO D24-07-B02 완료: DB 검색 결과 행을 정수 citation_id·chunk_id·본문·출처·거리의 딕셔너리로 변환합니다.
    # 튜플 순서는 coursekit/db.py search_current() SELECT 순서와 같다
    # (citation_id, chunk_id, body, source_locator, doc_id, source_version, cosine_distance)
    cid, chunk_id, body, locator, doc_id, source_version, distance = row
    if type(cid) is not int or cid <= 0:
        raise ValueError('invalid citation ID')
    # 반환 dict 키 : 의미
    # - citation_id     : DB가 발급한 정수 인용 번호 (답변의 source_ids에 쓰는 값)
    # - chunk_id        : 문자열 청크 ID (SHA-256 64자)
    # - body            : 청크 본문
    # - source_locator  : 원문 위치
    # - doc_id          : 원문 문서 ID
    # - source_version  : 원문 판
    # - cosine_distance : 질의 벡터와의 Cosine 거리 (작을수록 가까움). float로 변환
    return {
        'citation_id': cid,
        'chunk_id': chunk_id,
        'body': body,
        'source_locator': locator,
        'doc_id': doc_id,
        'source_version': source_version,
        'cosine_distance': float(distance),
    }
    # END D24-07-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 기본: 청크 준비 + 형식 확인용 검색 행 예시만 저장 / --live-db: 실제 E5 임베딩·저장·검색
def main():
    # ----------------------------------------------------
    # 4-1. 기본 실행 (DB 미접속)
    # ----------------------------------------------------
    doc, chunk = select_index_chunk('DOC-ASSIGN')
    # output 키 : 의미
    # - source_mode         : 'prepared_only_not_db' (DB에 넣지 않은 준비 단계 결과)
    # - chunk               : 색인 대상 청크
    # - live_db             : 'not_run' (실제 DB 검색을 하지 않았음을 명시)
    # - sample_hit_contract : citation_id 201·거리 0.0 으로 만든 형식 확인용 예시 (실제 검색 결과 아님)
    output = {
        'source_mode': 'prepared_only_not_db',
        'chunk': chunk,
        'live_db': 'not_run',
        'sample_hit_contract': format_hit((
            201,
            chunk['chunk_id'],
            chunk['body'],
            chunk['source_locator'],
            doc['doc_id'],
            'v1',
            0.0,
        )),
    }

    # ----------------------------------------------------
    # 4-2. --live-db 실제 실행 (E5 + PostgreSQL/pgvector)
    # ----------------------------------------------------
    if '--live-db' in sys.argv:
        # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
        from coursekit.live import connect_db, load_encoder, encode_passages, encode_query
        # 제공 모듈 coursekit/db.py: pgvector 실습 테이블 생성·Stage·Publish·현재 버전 Top-k 검색 SQL
        from coursekit.db import initialize, stage_version, publish_version, search_current, use_schema
        # 실제 모델 추론은 DB 트랜잭션 전에 수행합니다.
        # 주의: 모델 추론(임베딩)은 DB 트랜잭션을 열기 전에 끝낸다 (트랜잭션을 오래 잡지 않기 위함)
        # load_encoder : intfloat/multilingual-e5-small 을 CPU로 로드, 출력 384차원 확인
        model = load_encoder()
        # encode_passages : 'passage: ' 접두어·토큰 길이·정규화(길이 1)를 검사한 384차원 벡터 배열
        vectors = encode_passages(model, [chunk])
        # encode_query : 'query: 과제 파일은 어디에 보관하나요?' 를 같은 모델로 384차원 벡터화
        query = encode_query(model, '과제 파일은 어디에 보관하나요?')
        # connect_db : mentee/.env 의 PG* 설정으로 접속하고 vector 확장 활성화 여부 확인
        with connect_db() as conn:
            # 기본필수 07 전용 Schema d24_basic07 만 사용 (다른 수업 테이블과 섞지 않음)
            use_schema(conn, 'd24_basic07')
            # ingest_documents·ingest_versions·ingest_chunks 테이블이 없으면 생성
            initialize(conn)
            # 짧은 트랜잭션: Stage(저장) -> Publish(현재 버전 포인터 변경)를 한 번에 커밋
            with conn.transaction():
                stage_version(conn, doc, doc['file'], [chunk], vectors)
                publish_version(
                    conn,
                    doc['doc_id'],
                    doc['source_version'],
                    chunk['pipeline_version'],
                )
            # search_current 인자
            # - conn            : DB 연결
            # - query           : 질의 벡터
            # - category = '학사' : 분류 필터
            # - k = 3           : 상위 3건 (Exact Top-k, <=> Cosine 거리 오름차순)
            result = search_current(conn, query, category = '학사', k = 3)
            output['source_mode'] = 'e5_pgvector_live'
            # 실제 검색 행을 format_hit 로 Citation dict 목록으로 변환
            output['live_db'] = [format_hit(r) for r in result]
    out = save_json(HERE, 'indexing.json', output)
    show_success('07 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
