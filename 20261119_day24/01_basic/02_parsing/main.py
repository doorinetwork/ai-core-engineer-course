"""실습 02 Basic · TXT parsing의 기본.

실습 02 Parsing · 기본필수
- 역할: TXT 원문 bytes를 UTF-8 문자열로 해석해 출처 정보가 붙은 ParsedUnit 1건으로 만든다
- 기능: decode_text()(D24-02-B01) UTF-8 디코딩, make_unit()(D24-02-B02) ParsedUnit dict 생성
- 결과: outputs/parsed_unit.json
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

# ============================================================
# 2. 원문 bytes를 UTF-8로 해석 (TODO D24-02-B01)
# ============================================================
# 원문 바이트를 UTF-8 문자열로 바꾼다.
# - 정상 UTF-8          : str 반환 (예: b'\xea\xb3\xbc...' -> '과제 안내\r\n...')
# - 맨 앞 BOM(EF BB BF) : 'utf-8-sig'가 BOM을 떼고 해석
# - 잘못된 바이트       : UnicodeDecodeError로 중단 (errors='ignore'로 지우지 않음)
def decode_text(raw: bytes) -> str:
    # errors='ignore'를 쓰면 누락을 탐지하기 어렵습니다.
    # BEGIN D24-02-B01
    # 개념·실무 해설(mentor 원문): Parsing은 Byte를 해석하여 텍스트뿐 아니라 제목·문단·페이지·표의 관계를 얻는 작업입니다. UTF-8 디코딩과 PDF Parser는 별도 구현입니다.
    # TODO D24-02-B01 완료: 원문 바이트를 UTF-8로 디코딩해 str을 반환합니다. 잘못된 입력은 무시하지 않습니다.
    # 'utf-8-sig' : UTF-8 + 선택적 BOM 제거. BOM이 없으면 'utf-8'과 같은 결과
    # 주의: 줄바꿈 \r\n은 그대로 남는다. 줄바꿈 통일은 실습 03 Cleaning에서 수행
    return raw.decode('utf-8-sig')
    # END D24-02-B01

# ============================================================
# 3. ParsedUnit 1건 만들기 (TODO D24-02-B02)
# ============================================================
# 추출한 문자열에 문서 ID·버전·위치 정보를 붙여 ParsedUnit 한 건을 만든다.
# - doc_id         : 문서 ID (예: 'DOC-ASSIGN')
# - source_version : 원문 판 (예: 'v1'). 원문이 개정되면 'v2' 등으로 바뀜
# - text           : decode_text()가 만든 원문 문자열
def make_unit(doc_id: str, source_version: str, text: str) -> dict:
    # TXT에는 페이지 번호가 없으므로 None. 가짜 page 1을 만들지 않습니다.
    # BEGIN D24-02-B02
    # 개념·실무 해설(mentor 원문): ParsedUnit에는 raw_text만이 아니라 원문 문서 ID·source_version·unit_id·Locator가 함께 있어야 문제의 위치를 다시 찾을 수 있습니다.
    # TODO D24-02-B02 완료: 추출된 문자열을 doc_id·버전·unit_id·source_locator와 함께 한 건의 ParsedUnit으로 묶습니다.
    # 반환 dict 키 : 의미
    # - doc_id         : 어느 문서인가
    # - source_version : 어느 원문 판인가
    # - unit_id        : 문서 안 단위 ID. TXT는 통째로 한 단위라 'text-1'
    # - source_locator : 원문 위치 표시. TXT는 파일 이름 자체 ('assignment.txt')
    # - page_number    : TXT에는 페이지가 없으므로 None (가짜 1페이지를 만들지 않음)
    # - raw_text       : 정제 전 원문 문자열
    return {
        'doc_id': doc_id,
        'source_version': source_version,
        'unit_id': 'text-1',
        'source_locator': 'assignment.txt',
        'page_number': None,
        'raw_text': text,
    }
    # END D24-02-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# assignment.txt를 읽어 해석하고, 빈 결과가 아니면 ParsedUnit으로 저장한다.
def main():
    # 원문 bytes 읽기 (실습 01과 같은 원문 267 bytes)
    raw = (DATA / 'raw' / 'assignment.txt').read_bytes()
    text = decode_text(raw)
    # 공백뿐인 결과는 파싱 실패로 보고 저장하지 않음
    if not text.strip():
        raise ValueError('empty text')
    # make_unit 인자
    # - 'DOC-ASSIGN' : 과제 안내 문서 ID
    # - 'v1'         : 원문 판
    # - text         : 해석한 원문 문자열
    unit = make_unit('DOC-ASSIGN', 'v1', text)
    out = save_json(HERE, 'parsed_unit.json', unit)
    show_success('02 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
