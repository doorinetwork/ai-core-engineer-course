"""실습 04 Intermediate · --live-e5 시 실제 토크나이저 사용.

[주석본 안내] 실습 04 Chunking · 응용
- 역할: DOC-ATTEND·DOC-ASSIGN 의 ParsedUnit 을 70자/10자 겹침 창으로 나눠 출처가 붙은 청크를 만들고 토큰 길이 검사 결과를 기록한다
- 기능: split_one()(I01) 문자 창 분할, attach_chunks()(I02) 출처·passage 입력 부착,
        measure()(I03) 토큰 수 측정 또는 not_run, enforce_limit()(I04) 최대 길이 초과 검사
- 결과: outputs/chunking_report.json (기본 실행은 token_status 'not_run', token_validation status 'not_run')
- 실행: python main.py            (모델 없이 문자 창만)
- 실제: python main.py --live-e5  (intfloat/multilingual-e5-small Tokenizer 로 실제 길이 측정, 모델 다운로드 필요)
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
from coursekit.processing import clean_prose, char_windows, build_chunk, token_len

# ============================================================
# 2. ParsedUnit 하나를 문자 창으로 분할 (TODO D24-04-I01)
# ============================================================
# raw_text 를 정제한 clean_text 위에서 문자 창 [start, end) 목록을 만든다.
# - size = 70, overlap = 10 (기본값)
# - 반환값 : (clean_text, 창 목록). 창 위치는 clean_text 기준 Python 문자 Offset
def split_one(doc: dict, unit: dict, size = 70, overlap = 10) -> tuple[str, list[dict]]:
    # BEGIN D24-04-I01
    # 개념·실무 해설(mentor 원문): Chunk Span은 본문의 Python 문자 Offset입니다. 바이트 위치나 Token ID가 아닙니다. 원문 구간을 그대로 슬라이스해야 역추적할 수 있습니다.
    # TODO D24-04-I01 완료: 하나의 ParsedUnit을 clean_text에서 문자 창으로 분할합니다.
    # clean_prose : 줄바꿈 LF·NFC·가로 공백 축약 (위치 기준 문자열을 먼저 고정)
    text = clean_prose(unit['raw_text'])
    # char_windows : 0 <= overlap < size 검사 후 창 dict 를 하나씩 생성 (본문 끝에서 종료)
    return (text, list(char_windows(text, size, overlap)))
    # END D24-04-I01

# ============================================================
# 3. 창에 출처·passage 입력 붙이기 (TODO D24-04-I02)
# ============================================================
# 창마다 build_chunk 로 doc_id·unit_id·버전·Locator·chunk_id·embedding_text 를 붙인다.
# - embedding_text : 'passage: ' + 본문 (E5 문서 입력 계약)
# - Span 이 clean_text 와 다르면 build_chunk 안의 validate_span 이 ValueError
def attach_chunks(doc: dict, unit: dict, text: str, spans: list[dict]) -> list[dict]:
    # BEGIN D24-04-I02
    # 개념·실무 해설(mentor 원문): Chunk Size·Overlap은 0≤overlap<size를 만족해야 합니다. 창이 끝에 닿은 순간 종료해야 중복 꼬리 Chunk가 생기지 않습니다.
    # TODO D24-04-I02 완료: 각 창에 doc_id·unit_id·버전·Locator와 passage: 접두어를 전달합니다.
    return [build_chunk(doc, unit, text, s) for s in spans]
    # END D24-04-I02


# ============================================================
# 4. 토큰 수 측정 (TODO D24-04-I03)
# ============================================================
# 모델이 있으면 실제 Tokenizer 로 embedding_text 토큰 수를 재고, 없으면 not_run 으로 남긴다.
# - model None : token_count None, token_status 'not_run' (추정값을 만들지 않음)
# - model 있음 : token_count = 특수 토큰 포함·truncation 없는 실제 길이, token_status 'measured'
def measure(chunks: list[dict], model = None) -> list[dict]:
    # BEGIN D24-04-I03
    # 개념·실무 해설(mentor 원문): 문서·unit의 경계를 넘어 데이터를 합치지 않습니다. 같은 메타데이터를 Chunk로 정확히 상속시켜 출처가 뒤섞이지 않도록 합니다.
    # TODO D24-04-I03 완료: SentenceTransformer 모델이 주어졌을 때 특수 토큰 포함 길이를 측정합니다. 모델이 없으면 not_run으로 기록합니다.
    if model is None:
        # {**c, ...} : 기존 청크 dict 를 복사하고 두 키를 덧붙인 새 dict
        return [{**c, 'token_count': None, 'token_status': 'not_run'} for c in chunks]
    # token_len(model, text) : model.tokenizer(text, add_special_tokens = True, truncation = False) 의 input_ids 길이
    return [
        {**c, 'token_count': token_len(model, c['embedding_text']), 'token_status': 'measured'}
        for c in chunks
    ]
    # END D24-04-I03

# ============================================================
# 5. 최대 입력 길이 초과 검사 (TODO D24-04-I04)
# ============================================================
# 측정한 토큰 수가 model.max_seq_length 를 넘는 청크 ID 를 모은다.
# - model None     : status 'not_run', checked 0
# - 초과 없음      : status 'pass'
# - 초과 1건 이상  : status 'failed', too_long_ids 에 chunk_id 기록 (통과시키지 않음)
def enforce_limit(chunks: list[dict], model = None) -> dict:
    # BEGIN D24-04-I04
    # 개념·실무 해설(mentor 원문): E5 본문은 반드시 `passage: ` 접두어를 포함하고, title 추가 여부는 입력 계약으로 고정합니다. Offset에는 title 접두어를 섞지 않습니다.
    # TODO D24-04-I04 완료: 최대 입력 길이를 초과한 청크를 통과시키지 않고 개수와 ID를 기록합니다.
    if model is None:
        return {'status': 'not_run', 'too_long_ids': [], 'checked': 0}
    # token_count 가 None 이 아니고 max_seq_length 를 넘는 청크만 선택
    overs = [
        c['chunk_id']
        for c in chunks
        if c['token_count'] is not None and c['token_count'] > model.max_seq_length
    ]
    # 반환 dict 키 : 의미
    # - status       : 'pass' / 'failed'
    # - too_long_ids : 초과 청크 ID 목록
    # - checked      : 검사한 청크 수
    return {'status': 'failed' if overs else 'pass', 'too_long_ids': overs, 'checked': len(chunks)}
    # END D24-04-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# 두 문서의 모든 unit 을 청크로 만들고 (선택) 실제 토큰 길이를 검사해 저장한다.
def main():
    # 기본은 모델 없음 -> 토큰 검사 not_run
    model = None
    # --live-e5 옵션이 있을 때만 실제 E5 모델(CPU)을 로드
    if '--live-e5' in sys.argv:
        # 제공 모듈 coursekit/live.py: 실제 E5 모델·PostgreSQL 연결 (--live-e5 / --live-db 에서만 사용)
        from coursekit.live import load_encoder
        model = load_encoder()
    all_chunks = []

    # ----------------------------------------------------
    # 6-1. 문서·unit 별 청크 생성
    # ----------------------------------------------------
    for doc_id in ('DOC-ATTEND', 'DOC-ASSIGN'):
        doc = by_doc(doc_id)
        for unit in parse_one(doc)['units']:
            text, spans = split_one(doc, unit)
            all_chunks.extend(attach_chunks(doc, unit, text, spans))

    # ----------------------------------------------------
    # 6-2. 토큰 길이 측정과 초과 검사
    # ----------------------------------------------------
    measured = measure(all_chunks, model)
    report = enforce_limit(measured, model)
    # 저장 dict 키 : 의미
    # - chunks           : 토큰 측정 결과가 붙은 청크 목록
    # - token_validation : enforce_limit 결과
    out = save_json(HERE, 'chunking_report.json', {'chunks': measured, 'token_validation': report})
    show_success('04 Intermediate', out)
