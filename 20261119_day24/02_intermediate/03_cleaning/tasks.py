"""실습 03 Intermediate · 정제 계약과 보고서.

실습 03 Cleaning · 응용
- 역할: 출석 안내(DOC-ATTEND sec-1)를 NFC·줄바꿈·공백 규칙으로 정제하고 핵심 구절 보존·멱등성을 감사한다
- 기능: apply_nfc()(I01) NFC 정규화, normalize_prose()(I02) prose 정제,
        check_phrases()(I03) 핵심 구절 검사, make_audit()(I04) 정제 감사 기록
- 결과: outputs/clean_report.json (82자 -> 82자, '10분'·'결석'·'사전 승인' 모두 보존, idempotent true)
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys, re, unicodedata, hashlib
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# HERE.parents[1] : <수준>/<실습>_주석 기준 두 단계 위 = mentee 폴더
# -> mentee 를 import 경로 맨 앞에 넣어 coursekit 패키지를 불러온다
sys.path.insert(0, str(HERE.parents[1]))
# 제공 모듈 coursekit/io.py: 공통 경로(DATA)·JSON 읽기/저장·완료 메시지
from coursekit.io import DATA, save_json, show_success
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import by_doc, parse_one

# ============================================================
# 2. Unicode NFC 정규화 (TODO D24-03-I01)
# ============================================================
# 같은 글자의 서로 다른 Unicode 표현을 NFC 한 가지로 맞춘다.
# - 'ᄒ'+'ᅡ'+'ᆫ' (자모 분리형) -> '한' (완성형)
# - '①' -> '①' 그대로 (NFKC 였다면 '1' 로 바뀌어 번호 표기가 사라짐)
def apply_nfc(text: str) -> str:
    # BEGIN D24-03-I01
    # 개념·실무 해설(mentor 원문): NFC는 Unicode 정준 정규화이며, NFKC는 ① 등의 호환 문자까지 바꿀 수 있습니다. 손실 범위가 더 큰 NFKC를 무조건 적용하지 않습니다.
    # TODO D24-03-I01 완료: 원문을 Unicode NFC로 정규화합니다. NFKC로 동그라미 번호를 무조건 지우지 않습니다.
    return unicodedata.normalize('NFC', text)
    # END D24-03-I01

# ============================================================
# 3. prose 정제 (TODO D24-03-I02)
# ============================================================
# 줄바꿈 LF 통일 -> NFC -> 줄마다 가로 공백 축약 순서로 일반 문장을 정제한다.
# - '10분  초과\r\n단,  사전' -> '10분 초과\n단, 사전'
# - 빈 줄(문단 경계)은 유지
def normalize_prose(text: str) -> str:
    # BEGIN D24-03-I02
    # 개념·실무 해설(mentor 원문): 일반 prose의 연속된 가로 공백만 축약하고 줄바꿈 경계는 유지합니다. 코드 들여쓰기와 표의 탭 구분에는 같은 규칙을 적용하지 않습니다.
    # TODO D24-03-I02 완료: CRLF·CR·가로 공백만 정리하며 문단 경계를 보존합니다.
    # CRLF·CR 을 LF 로 바꾼 뒤 NFC 정규화
    text = apply_nfc(text.replace('\r\n', '\n').replace('\r', '\n'))
    # 줄 단위로 [ \t]+ -> ' ' 축약, 줄 앞뒤 공백 제거, 전체 앞뒤 공백 제거
    return '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')).strip()
    # END D24-03-I02

# ============================================================
# 4. 핵심 구절 보존 검사 (TODO D24-03-I03)
# ============================================================
# 숫자·조건·예외 같은 핵심 구절이 정제 결과에 남았는지 나눠 기록한다.
# - phrases ['10분', '결석', '사전 승인'] -> present 3개, missing []
def check_phrases(text: str, phrases: list[str]) -> dict:
    # BEGIN D24-03-I03
    # 개념·실무 해설(mentor 원문): 정제 중 부정어, 숫자, 단위, 조건, 예외가 사라지면 업무 규정이 뒤집힐 수 있습니다. 핵심 문자열은 원문에서 추출해 비교합니다.
    # TODO D24-03-I03 완료: 핵심 구절 목록을 받아 부정어·숫자·예외의 존재 여부를 검사합니다.
    # 반환 dict 키 : 의미
    # - present : text 안에 있는 구절
    # - missing : text 에서 사라진 구절 (하나라도 있으면 규정 의미가 바뀌었을 수 있음)
    return {
        'present': [p for p in phrases if p in text],
        'missing': [p for p in phrases if p not in text],
    }
    # END D24-03-I03

# ============================================================
# 5. 정제 감사 기록 (TODO D24-03-I04)
# ============================================================
# 정제 규칙 버전·전후 길이·clean_hash·구절 검사·멱등성을 한 dict 로 남긴다.
def make_audit(original: str, cleaned: str, phrases: list[str]) -> dict:
    # BEGIN D24-03-I04
    # 개념·실무 해설(mentor 원문): 동일 입력을 다시 정제했을 때 결과가 같아야 합니다. 멱등성은 일관성 검사일 뿐 의미 보존의 충분조건이 아닙니다.
    # TODO D24-03-I04 완료: 정제 규칙 버전·전후 길이·해시·검사 결과를 출력합니다.
    # 반환 dict 키 : 의미
    # - cleaner_version : 정제 규칙 버전 'nfc-prose-v1' (규칙이 바뀌면 버전도 바꿈)
    # - chars_before    : 정제 전 글자 수 (예: 82)
    # - chars_after     : 정제 후 글자 수 (예: 82)
    # - clean_hash      : 정제 결과를 UTF-8 로 인코딩한 SHA-256 (원본 bytes 해시 raw_sha256 과 다름)
    # - phrase_check    : check_phrases 결과
    # - idempotent      : 정제 결과를 한 번 더 정제해도 같은가 (True 여야 함)
    # 주의: 멱등성 True 는 일관성일 뿐 의미 보존을 보증하지 않는다 (phrase_check 를 함께 확인)
    return {
        'cleaner_version': 'nfc-prose-v1',
        'chars_before': len(original),
        'chars_after': len(cleaned),
        'clean_hash': hashlib.sha256(cleaned.encode('utf-8')).hexdigest(),
        'phrase_check': check_phrases(cleaned, phrases),
        'idempotent': normalize_prose(cleaned) == cleaned,
    }
    # END D24-03-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# DOC-ATTEND 첫 단위(sec-1 '지각과 결석')를 정제·감사하고 저장한다.
def main():
    doc = by_doc('DOC-ATTEND')
    # attendance.md 의 첫 Markdown 단위 (sec-1)
    unit = parse_one(doc)['units'][0]
    original = unit['raw_text']
    cleaned = normalize_prose(original)
    # make_audit 인자
    # - original : 정제 전 raw_text
    # - cleaned  : 정제 후 문자열
    # - ['10분', '결석', '사전 승인'] : 반드시 남아야 하는 핵심 구절
    audit = make_audit(original, cleaned, ['10분', '결석', '사전 승인'])
    # 저장 dict 키 : 의미
    # - doc_id     : 'DOC-ATTEND'
    # - unit_id    : 'sec-1'
    # - clean_text : 정제 결과
    # - audit      : 감사 기록
    out = save_json(
        HERE,
        'clean_report.json',
        {
            'doc_id': doc['doc_id'],
            'unit_id': unit['unit_id'],
            'clean_text': cleaned,
            'audit': audit,
        },
    )
    show_success('03 Intermediate', out)
