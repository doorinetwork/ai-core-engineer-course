"""실습 03 Advanced · 구조별 정제 정책과 실패 회귀.

실습 03 Cleaning · 심화
- 역할: 본문 유형(prose/code/table)별 정제 정책을 나누고, 핵심 구절 손실·멱등성·clean_hash 를 감사하며 손상 규칙을 회귀 검사한다
- 기능: choose_policy()(A01) 정책 선택, clean_block()(A02) 유형별 정제, check_critical()(A03) 핵심 구절 손실,
        check_idempotent()(A04) 반복 정제 일치, clean_fingerprint()(A05) clean_hash, audit_unit()(A06) unit 감사
- 결과: outputs/cleaning_audit.json (sec-1·sec-2 감사, bad_rule passed false·lost_phrases ['10분'])
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import hashlib, re, sys, unicodedata
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# HERE.parents[1] : <수준>/<실습>_주석 기준 두 단계 위 = mentee 폴더
# -> mentee 를 import 경로 맨 앞에 넣어 coursekit 패키지를 불러온다
sys.path.insert(0, str(HERE.parents[1]))
# 제공 모듈 coursekit/io.py: 공통 경로(DATA)·JSON 읽기/저장·완료 메시지
from coursekit.io import save_json, show_success
# 제공 모듈 coursekit/processing.py: 정제·문자 창 Chunking·Span 검증·Coverage·안정 chunk_id·E5 입력 문자열
from coursekit.processing import clean_prose
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import by_doc, parse_one

# ============================================================
# 2. 정제 정책 선택 (TODO D24-03-A01)
# ============================================================
# 본문 유형에 맞는 정제 정책 이름을 돌려준다.
# - 'prose'          -> 'nfc_prose' (NFC·LF·가로 공백 축약)
# - 'code' / 'table' -> 'preserve_structure' (들여쓰기·탭 구분 보존)
# - 그 밖           -> ValueError('unknown block')
def choose_policy(block_type: str) -> str:
    # BEGIN D24-03-A01
    # 개념·실무 해설(mentor 원문): NFC는 Unicode 정준 정규화이며, NFKC는 ① 등의 호환 문자까지 바꿀 수 있습니다. 손실 범위가 더 큰 NFKC를 무조건 적용하지 않습니다.
    # TODO D24-03-A01 완료: 본문 유형 prose/code/table을 구분해 정제 정책을 선택합니다.
    if block_type not in ('prose', 'code', 'table'):
        raise ValueError('unknown block')
    return 'nfc_prose' if block_type == 'prose' else 'preserve_structure'
    # END D24-03-A01

# ============================================================
# 3. 유형별 정제 (TODO D24-03-A02)
# ============================================================
# prose 는 clean_prose, code·table 은 줄바꿈만 LF 로 통일한다.
# - prose : '10분  초과\r\n' -> '10분 초과'
# - code  : '    return x\r\n' -> '    return x\n' (들여쓰기 4칸 유지)
def clean_block(text: str, block_type: str) -> str:
    # BEGIN D24-03-A02
    # 개념·실무 해설(mentor 원문): 일반 prose의 연속된 가로 공백만 축약하고 줄바꿈 경계는 유지합니다. 코드 들여쓰기와 표의 탭 구분에는 같은 규칙을 적용하지 않습니다.
    # TODO D24-03-A02 완료: 일반 본문에만 NFC·LF·가로 공백 축약을 수행하고 코드·표 구조를 보존합니다.
    policy = choose_policy(block_type)
    if policy == 'nfc_prose':
        return clean_prose(text)
    # 정렬된 표·Python 들여쓰기를 공백으로 뭉개지 않습니다.
    return text.replace('\r\n', '\n').replace('\r', '\n')
    # END D24-03-A02

# ============================================================
# 4. 핵심 구절 손실 검사 (TODO D24-03-A03)
# ============================================================
# 원문에 있던 핵심 구절이 정제 결과에서 사라졌는지 찾는다.
# - 원문 '10분 초과는 결석.' / 결과 ' 초과는 결석.' / ['10분'] -> passed False, lost_phrases ['10분']
# - 원문에 처음부터 없는 구절은 손실로 세지 않음
def check_critical(source: str, cleaned: str, required: list[str]) -> dict:
    # BEGIN D24-03-A03
    # 개념·실무 해설(mentor 원문): 정제 중 부정어, 숫자, 단위, 조건, 예외가 사라지면 업무 규정이 뒤집힐 수 있습니다. 핵심 문자열은 원문에서 추출해 비교합니다.
    # TODO D24-03-A03 완료: 숫자·조건·부정 표현의 삭제를 발견해 실패로 표시합니다.
    missing = [p for p in required if p in source and p not in cleaned]
    # 반환 dict 키 : passed(손실 없음 여부), lost_phrases(사라진 구절 목록)
    return {'passed': not missing, 'lost_phrases': missing}
    # END D24-03-A03

# ============================================================
# 5. 반복 정제 일치 검사 (TODO D24-03-A04)
# ============================================================
# 이미 정제한 문자열을 같은 규칙으로 한 번 더 정제해도 그대로인지 확인한다.
# 주의: True 는 일관성일 뿐, 의미 보존은 check_critical 로 따로 확인한다
def check_idempotent(cleaned: str, block_type: str) -> bool:
    # BEGIN D24-03-A04
    # 개념·실무 해설(mentor 원문): 동일 입력을 다시 정제했을 때 결과가 같아야 합니다. 멱등성은 일관성 검사일 뿐 의미 보존의 충분조건이 아닙니다.
    # TODO D24-03-A04 완료: 같은 Cleaner를 반복 적용했을 때 결과가 동일한지 검사합니다.
    return clean_block(cleaned, block_type) == cleaned
    # END D24-03-A04

# ============================================================
# 6. clean_hash 계산 (TODO D24-03-A05)
# ============================================================
# 정제된 문자열을 UTF-8 로 인코딩한 SHA-256 을 돌려준다.
# - raw_sha256 (원본 bytes 해시) 와 다른 값: 정제 규칙이 바뀌면 clean_hash 만 바뀐다
def clean_fingerprint(text: str) -> str:
    # BEGIN D24-03-A05
    # 개념·실무 해설(mentor 원문): clean_hash는 정제한 문자열을 UTF-8로 인코딩한 뒤 계산합니다. raw_sha256은 정제 전 원본 바이트의 해시이므로 둘을 혼동하지 않습니다.
    # TODO D24-03-A05 완료: 원본 bytes 해시와 별도로 clean_text UTF-8 해시를 기록합니다.
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
    # END D24-03-A05

# ============================================================
# 7. unit 정제 감사 (TODO D24-03-A06)
# ============================================================
# ParsedUnit 1건을 prose 로 정제하고 핵심 구절·멱등성·clean_hash 를 감사 기록으로 남긴다.
# - 핵심 구절 손실 : ValueError('critical text removed') (감사 기록을 성공으로 만들지 않음)
def audit_unit(doc_id: str, unit: dict, source_version: str) -> dict:
    # BEGIN D24-03-A06
    # 개념·실무 해설(mentor 원문): Cleaner 버전·문서 ID·unit ID·정제 전후 차이를 함께 남겨야 검색 결과 문제를 앞단 Cleaning 단계까지 추적할 수 있습니다.
    # TODO D24-03-A06 완료: 문서 ID·단위 ID·버전·정제 규칙과 실패 이유를 감사 보고서로 남깁니다.
    raw = unit['raw_text']
    clean = clean_block(raw, 'prose')
    # 반드시 남아야 하는 구절: '10분', '결석', '사전 승인' (원문에 있는 것만 검사)
    check = check_critical(raw, clean, ['10분', '결석', '사전 승인'])
    if not check['passed']:
        raise ValueError('critical text removed')
    # 반환 dict 키 : 의미
    # - doc_id / source_version / unit_id : 어느 문서·판·단위인가
    # - cleaner_version : 'nfc-prose-v1'
    # - clean_text      : 정제 결과
    # - clean_hash      : clean_fingerprint 결과
    # - meaning_check   : check_critical 결과
    # - idempotent      : check_idempotent 결과
    return {
        'doc_id': doc_id,
        'source_version': source_version,
        'unit_id': unit['unit_id'],
        'cleaner_version': 'nfc-prose-v1',
        'clean_text': clean,
        'clean_hash': clean_fingerprint(clean),
        'meaning_check': check,
        'idempotent': check_idempotent(clean, 'prose'),
    }
    # END D24-03-A06

# ============================================================
# 8. 실행 흐름 (main)
# ============================================================
# DOC-ATTEND 의 모든 unit 을 감사하고, 숫자를 지우는 손상 규칙이 반드시 탐지되는지 확인한다.
def main():
    doc = by_doc('DOC-ATTEND')
    result = parse_one(doc)
    units = [audit_unit(doc['doc_id'], u, doc['source_version']) for u in result['units']]

    # ----------------------------------------------------
    # 8-1. 손상 규칙 회귀 검사
    # ----------------------------------------------------
    # corrupt : '10분' 을 지운 손상 결과 ' 초과는 결석.'
    corrupt = '10분 초과는 결석.'.replace('10분', '')
    bad = check_critical('10분 초과는 결석.', corrupt, ['10분'])
    # 손상을 통과시키면 검사 로직 오류이므로 AssertionError
    if bad['passed']:
        raise AssertionError('must detect lost number')
    out = save_json(HERE, 'cleaning_audit.json', {'units': units, 'bad_rule': bad})
    show_success('03 Advanced', out)
