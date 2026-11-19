"""실습 08 Basic · Checkpoint를 연결합니다. LLM 답변 생성은 하지 않습니다.

실습 08 Daily Project · 기본필수
- 역할: 문서 1건(DOC-ASSIGN)을 Parsing -> Cleaning -> Chunking 으로 연결하고 결과를 검증 보고서로 남긴다
- 기능: run_pipeline()(D24-08-B01) 파이프라인 실행, check_report()(D24-08-B02) 문서 ID·Span 검증과 보고서
- 결과: outputs/daily_project.json 
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
from coursekit.processing import clean_prose, char_windows, build_chunk, validate_span

# ============================================================
# 2. Parsing -> Cleaning -> Chunking 연결 (TODO D24-08-B01)
# ============================================================
# 문서를 파싱하고 첫 단위를 정제한 뒤 110자/15자 겹침 창으로 청크를 만든다.
# - 'DOC-ASSIGN' : (manifest dict, clean_text 106자, 청크 1건)
# - 파싱 실패    : ValueError('parse not ready')
def run_pipeline(doc_id: str) -> tuple[dict, str, list[dict]]:
    # BEGIN D24-08-B01
    # 개념·실무 해설(mentor 원문): 원본 목록에서 입력 1건이 사라지면 성공률 해석이 왜곡됩니다. 모든 doc_id의 최종 상태를 추적합니다.
    # TODO D24-08-B01 완료: DOC-ASSIGN의 Parsing 결과를 읽어 정제하고 110자/15자 설정으로 청크를 만듭니다.
    # manifest 에서 문서 정보 조회
    doc = by_doc(doc_id)
    result = parse_one(doc)
    if result['status'] != 'parsed':
        raise ValueError('parse not ready')
    # TXT 문서는 ParsedUnit 이 'text-1' 하나
    unit = result['units'][0]
    # clean_prose : CRLF -> LF, NFC 정규화, 가로 공백 축약 (실습 03과 같은 규칙)
    clean = clean_prose(unit['raw_text'])
    # char_windows(clean, 110, 15) : [start, end) 문자 창 생성기
    # build_chunk(doc, unit, clean, span) : Span 검증 후 출처·chunk_id·passage 입력을 가진 청크 dict
    chunks = [build_chunk(doc, unit, clean, span) for span in char_windows(clean, 110, 15)]
    return (doc, clean, chunks)
    # END D24-08-B01

# ============================================================
# 3. 청크 검증과 보고서 (TODO D24-08-B02)
# ============================================================
# 모든 청크가 같은 문서이고 Span 본문이 clean_text 와 일치할 때만 보고서를 만든다.
# - 청크 0건       : ValueError('empty chunk list')
# - 다른 문서 청크 : ValueError('doc_id mismatch')
# - Span 불일치    : validate_span 이 ValueError
def check_report(doc: dict, clean: str, chunks: list[dict]) -> dict:
    # BEGIN D24-08-B02
    # 개념·실무 해설(mentor 원문): 단계별로 ParsedUnit → CleanUnit → Chunk가 같은 문서/버전/원문 위치를 유지해야 합니다. clean_text 기반 Span을 검증합니다.
    # TODO D24-08-B02 완료: 원문 ID·Chunk 수·Span 일치 여부를 기록하고 문제가 있으면 성공으로 처리하지 않습니다.
    if not chunks:
        raise ValueError('empty chunk list')

    # ----------------------------------------------------
    # 3-1. 청크마다 문서 ID·Span 검사
    # ----------------------------------------------------
    for c in chunks:
        if c['doc_id'] != doc['doc_id']:
            raise ValueError('doc_id mismatch')
        # validate_span 은 'text' 키를 읽으므로 청크의 body 를 text 로 바꿔 넘긴다
        validate_span(
            {'char_start': c['char_start'], 'char_end': c['char_end'], 'text': c['body']},
            clean,
        )
    # 보고서 dict 키 : 의미
    # - doc_id          : 검사한 문서 ID
    # - status          : 'prepared_not_embedded' (청크 준비 완료, 임베딩·DB 저장은 하지 않음)
    # - chunk_count     : 청크 수 (DOC-ASSIGN은 1)
    # - source_versions : 사용한 원문 판 목록 ['v1']
    # - chunk_ids       : 청크 ID 목록
    return {
        'doc_id': doc['doc_id'],
        'status': 'prepared_not_embedded',
        'chunk_count': len(chunks),
        'source_versions': [doc['source_version']],
        'chunk_ids': [c['chunk_id'] for c in chunks],
    }
    # END D24-08-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 파이프라인 실행 -> 검증 보고서 -> 보고서와 청크를 함께 저장
def main():
    doc, clean, chunks = run_pipeline('DOC-ASSIGN')
    report = check_report(doc, clean, chunks)
    out = save_json(HERE, 'daily_project.json', {'report': report, 'chunks': chunks})
    show_success('08 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
