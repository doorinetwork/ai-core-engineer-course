"""실습 04 Advanced · 구조 우선, 긴 부분 재분할; --live-e5는 모델 길이 측정.

[주석본 안내] 실습 04 Chunking · 심화
- 역할: 문단 구조를 먼저 자르고 긴 문단만 문자 창으로 재분할하며, E5 입력(제목 선택)·토큰 길이·재실행 안정성을 검사한다
- 기능: paragraph_spans()(A01) 문단 Span, split_long()(A02) 긴 문단 재분할, make_records()(A03) 청크 생성,
        embedding_input()(A04) passage 입력, check_model_lengths()(A05) 토큰 길이·격리, check_repeat()(A06) 재실행 비교
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import hashlib, sys
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
from coursekit.processing import clean_prose, char_windows, build_chunk, token_len

# ============================================================
# 2. 문단 Span 추출 (TODO D24-04-A01)
# ============================================================
# 줄(\n) 단위 문단마다 clean_text 안의 정확한 [start, end) 위치를 찾는다.
# - 'A문장\n\nB문장' -> [{0, 3, 'A문장'}, {5, 8, 'B문장'}] (빈 줄은 건너뜀)
# - 찾은 위치가 없으면 ValueError('paragraph position mismatch')
def paragraph_spans(text: str) -> list[dict]:
    # BEGIN D24-04-A01
    # 개념·실무 해설(mentor 원문): Chunk Span은 본문의 Python 문자 Offset입니다. 바이트 위치나 Token ID가 아닙니다. 원문 구간을 그대로 슬라이스해야 역추적할 수 있습니다.
    # TODO D24-04-A01 완료: 문단 경계 후보를 찾아 원문 구간을 정확히 추출합니다.
    result = []
    # cursor : 다음 문단을 찾기 시작할 위치 (같은 문장이 두 번 나와도 앞쪽을 다시 잡지 않음)
    cursor = 0
    for para in text.split('\n'):
        # 빈 줄: 길이 + 줄바꿈 1글자만큼 cursor 이동
        if not para.strip():
            cursor += len(para) + 1
            continue
        # text.find(para, cursor) : cursor 이후에서 문단이 처음 나오는 위치
        index = text.find(para, cursor)
        if index < 0:
            raise ValueError('paragraph position mismatch')
        result.append({'char_start': index, 'char_end': index + len(para), 'text': para})
        cursor = index + len(para) + 1
    return result
    # END D24-04-A01

# ============================================================
# 3. 긴 문단만 재분할 (TODO D24-04-A02)
# ============================================================
# size(90자) 이하 문단은 그대로 두고, 긴 문단만 90자/15자 겹침 창으로 나눈다.
# - 창 위치는 문단 안 상대 위치 + 문단 시작 위치 = clean_text 기준 절대 위치
def split_long(text: str, spans: list[dict], size = 90, overlap = 15) -> list[dict]:
    # BEGIN D24-04-A02
    # 개념·실무 해설(mentor 원문): Chunk Size·Overlap은 0≤overlap<size를 만족해야 합니다. 창이 끝에 닿은 순간 종료해야 중복 꼬리 Chunk가 생기지 않습니다.
    # TODO D24-04-A02 완료: 긴 문단만 문자 창으로 재분할하며 짧은 문단은 그대로 유지합니다.
    out = []
    for s in spans:
        if len(s['text']) <= size:
            out.append(s)
        else:
            for w in char_windows(s['text'], size, overlap):
                # 예: 문단 시작 120, 창 [0, 90) -> 절대 [120, 210)
                a = s['char_start'] + w['char_start']
                b = s['char_start'] + w['char_end']
                # text[a:b] : 절대 위치로 다시 잘라 본문 일치 보장
                out.append({'char_start': a, 'char_end': b, 'text': text[a:b]})
    return out
    # END D24-04-A02

# ============================================================
# 4. Span -> 청크 레코드 (TODO D24-04-A03)
# ============================================================
# Span 마다 build_chunk 로 같은 문서·unit 의 Metadata 를 상속한 청크를 만든다.
def make_records(doc: dict, unit: dict, clean: str, spans: list[dict]) -> list[dict]:
    # BEGIN D24-04-A03
    # 개념·실무 해설(mentor 원문): 문서·unit의 경계를 넘어 데이터를 합치지 않습니다. 같은 메타데이터를 Chunk로 정확히 상속시켜 출처가 뒤섞이지 않도록 합니다.
    # TODO D24-04-A03 완료: 각 Span을 doc_id·unit_id·source_locator에 연결해 Chunk를 만듭니다.
    return [build_chunk(doc, unit, clean, s) for s in spans]
    # END D24-04-A03

# ============================================================
# 5. E5 passage 입력 만들기 (TODO D24-04-A04)
# ============================================================
# 본문(선택적으로 제목 포함) 앞에 'passage: ' 를 붙인다. Offset 계산에는 영향을 주지 않는다.
# - include_title True, 제목 '지각과 결석' -> 'passage: 지각과 결석\n수업 시작 후 ...'
# - include_title False 또는 제목 없음     -> 'passage: 수업 시작 후 ...'
def embedding_input(chunk: dict, include_title: bool) -> str:
    # BEGIN D24-04-A04
    # 개념·실무 해설(mentor 원문): E5 본문은 반드시 `passage: ` 접두어를 포함하고, title 추가 여부는 입력 계약으로 고정합니다. Offset에는 title 접두어를 섞지 않습니다.
    # TODO D24-04-A04 완료: 본문과 선택적 제목을 합쳐 `passage: ` 입력을 만들고 Offset과 분리합니다.
    title = chunk.get('section_title', '').strip()
    body = title + '\n' + chunk['body'] if include_title and title else chunk['body']
    return 'passage: ' + body
    # END D24-04-A04

# ============================================================
# 6. 토큰 길이 검사와 격리 (TODO D24-04-A05)
# ============================================================
# 모델이 있으면 제목 포함 입력의 실제 토큰 수를 재고, max_seq_length 초과 청크를 격리한다.
# - model None : status 'not_run' (토큰 수를 추정하지 않음)
# - 초과 없음  : 'pass' / 초과 있음 : 'quarantine' + oversize_ids
# 주의: 384 는 벡터 차원이지 허용 토큰 수가 아니다
def check_model_lengths(chunks: list[dict], model = None) -> dict:
    # BEGIN D24-04-A05
    # 개념·실무 해설(mentor 원문): Token 길이는 실제 E5 Tokenizer로 `add_special_tokens=True`, `truncation=False` 상태에서 재야 합니다. 384차원은 384 Token 허용을 의미하지 않습니다.
    # TODO D24-04-A05 완료: 모델이 제공되면 Tokenizer로 길이를 측정하고 초과 결과를 격리합니다.
    if model is None:
        return {'status': 'not_run', 'checked': 0, 'oversize_ids': []}
    overs = []
    for chunk in chunks:
        # 실제로 모델에 넣을 문자열 전체(제목 포함)를 같은 Tokenizer 로 측정
        t = embedding_input(chunk, True)
        if token_len(model, t) > model.max_seq_length:
            overs.append(chunk['chunk_id'])
    return {
        'status': 'pass' if not overs else 'quarantine',
        'checked': len(chunks),
        'oversize_ids': overs,
    }
    # END D24-04-A05

# ============================================================
# 7. 재실행 안정성 검사 (TODO D24-04-A06)
# ============================================================
# 같은 원문·설정으로 두 번 청킹해 chunk_id·Span·본문이 모두 같은지 확인한다.
# - 다르면 ValueError('unstable chunking')
def check_repeat(doc: dict, unit: dict) -> dict:
    # BEGIN D24-04-A06
    # 개념·실무 해설(mentor 원문): 분할의 재실행에서 Chunk ID·Span·본문이 같아야 합니다. 모델/분할 조건이 바뀌면 pipeline_version도 변경해야 합니다.
    # TODO D24-04-A06 완료: 동일 원문·설정의 재실행에서 Chunk ID·본문·문자 범위가 동일한지 검사합니다.
    clean = clean_prose(unit['raw_text'])
    # make : 문단 Span -> 긴 문단 재분할 -> 청크 레코드 를 한 번에 수행하는 내부 함수
    def make():
        return make_records(doc, unit, clean, split_long(clean, paragraph_spans(clean)))
    first = make()
    second = make()
    # 비교 대상 4가지: (chunk_id, char_start, char_end, body)
    a = [(r['chunk_id'], r['char_start'], r['char_end'], r['body']) for r in first]
    b = [(r['chunk_id'], r['char_start'], r['char_end'], r['body']) for r in second]
    if a != b:
        raise ValueError('unstable chunking')
    return {'chunks': first, 'stable': True}
    # END D24-04-A06

# ============================================================
# 8. 실행 흐름 (main)
# ============================================================
# DOC-ATTEND 첫 unit 을 구조 우선으로 청킹하고 (선택) 토큰 길이를 검사해 저장한다.
def main():
    doc = by_doc('DOC-ATTEND')
    unit = parse_one(doc)['units'][0]
    result = check_repeat(doc, unit)
    model = None
    if '--live-e5' in sys.argv:
        # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
        from coursekit.live import load_encoder
        model = load_encoder()
    result['token_check'] = check_model_lengths(result['chunks'], model)
    # 입력 계약 표시: 'passage: [optional title] + body'
    result['embedding_input_contract'] = 'passage: [optional title] + body'
    out = save_json(HERE, 'structure_chunks.json', result)
    show_success('04 Advanced', out)
