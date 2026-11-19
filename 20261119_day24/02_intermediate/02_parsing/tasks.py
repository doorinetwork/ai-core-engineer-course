"""실습 02 Intermediate · format dispatch와 공통 ParsedUnit.

실습 02 Parsing · 응용
- 역할: 확장자로 Parser 경로를 고르고 6건 모두를 공통 ParsedUnit 결과로 만든다
- 기능: kind_of()(I01) 형식 판정, parse_txt()(I02) TXT, parse_markup()(I03) Markdown·HTML,
        parse_dispatch()(I04) 형식별 분기와 unsupported 기록
- 결과: outputs/parsing_report.json (parsed 4, needs_ocr 1, unsupported 1)
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
from coursekit.io import DATA, load_json, save_json, show_success
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import parse_one, raw_path

# ============================================================
# 2. 파일 형식 판정 (TODO D24-02-I01)
# ============================================================
# 확장자를 Parser 종류 이름으로 바꾼다.
# - '.txt' -> 'txt', '.md' -> 'markdown', '.html' -> 'html', '.pdf' -> 'pdf'
# - 그 밖 (예: '.xyz') -> 'unsupported'
def kind_of(doc: dict) -> str:
    # BEGIN D24-02-I01
    # 개념·실무 해설(mentor 원문): Parsing은 Byte를 해석하여 텍스트뿐 아니라 제목·문단·페이지·표의 관계를 얻는 작업입니다. UTF-8 디코딩과 PDF Parser는 별도 구현입니다.
    # TODO D24-02-I01 완료: 파일 확장자로 파서 경로를 결정합니다. 미지원 확장자를 정상 취급하지 않습니다.
    suffix = Path(doc['file']).suffix.lower()
    # dict.get(suffix, 'unsupported') : 표에 없는 확장자는 기본값 'unsupported'
    return {
        '.txt': 'txt',
        '.md': 'markdown',
        '.html': 'html',
        '.pdf': 'pdf',
    }.get(suffix, 'unsupported')
    # END D24-02-I01

# ============================================================
# 3. TXT Parser (TODO D24-02-I02)
# ============================================================
# TXT 문서만 받아 coursekit parse_one() 으로 UTF-8 해석한다.
# - TXT 가 아닌 문서 : ValueError('wrong parser') (잘못된 Parser 경로 방지)
# - 반환값          : {'doc', 'status', 'units', 'reason'} (TXT 는 unit 'text-1' 하나)
def parse_txt(doc: dict) -> dict:
    # BEGIN D24-02-I02
    # 개념·실무 해설(mentor 원문): ParsedUnit에는 raw_text만이 아니라 원문 문서 ID·source_version·unit_id·Locator가 함께 있어야 문제의 위치를 다시 찾을 수 있습니다.
    # TODO D24-02-I02 완료: TXT를 UTF-8로 읽고 오류와 빈 결과를 구분합니다.
    if kind_of(doc) != 'txt':
        raise ValueError('wrong parser')
    return parse_one(doc)
    # END D24-02-I02

# ============================================================
# 4. Markdown·HTML Parser (TODO D24-02-I03)
# ============================================================
# Markdown·HTML 문서만 받아 구조를 가진 ParsedUnit 으로 만든다.
# - Markdown : '#' 제목마다 unit (예: attendance.md -> sec-1 '지각과 결석', sec-2 '출석 문의')
# - HTML     : <main> 안 본문과 표만 추출 (nav·script·style 제외, 표는 '시설: 실습실 | 안내: ...' 형식)
# - 다른 형식 : ValueError('wrong parser')
def parse_markup(doc: dict) -> dict:
    # BEGIN D24-02-I03
    # 개념·실무 해설(mentor 원문): HTML의 nav·script는 본문이 아닙니다. 표는 시설 이름과 안내 값의 열·행 관계가 유지되는지 확인합니다.
    # TODO D24-02-I03 완료: Markdown·HTML에서 제목·본문·표의 구조를 가진 ParsedUnit을 추출합니다.
    if kind_of(doc) not in {'markdown', 'html'}:
        raise ValueError('wrong parser')
    return parse_one(doc)
    # END D24-02-I03

# ============================================================
# 5. 형식별 분기 (TODO D24-02-I04)
# ============================================================
# kind_of 결과에 따라 Parser 를 고르고, 지원하지 않는 형식도 결과에서 빠뜨리지 않는다.
# - txt              : parse_txt
# - markdown / html  : parse_markup
# - pdf              : parse_one (페이지마다 unit, 텍스트층 없는 scan.pdf 는 needs_ocr)
# - unsupported      : status 'unsupported', reason 'unsupported_extension'
def parse_dispatch(doc: dict) -> dict:
    # BEGIN D24-02-I04
    # 개념·실무 해설(mentor 원문): pypdf의 extract_text()는 이미지 OCR이 아닙니다. `scan.pdf`는 자료 제작 시 이미지 전용임을 알고 분리한 샘플이며 다른 빈 PDF는 원문 확인이 필요합니다.
    # TODO D24-02-I04 완료: PDF는 페이지별 위치를 기록하고 이미지 PDF는 별도 needs_ocr로 분류합니다.
    kind = kind_of(doc)
    if kind == 'txt':
        return parse_txt(doc)
    if kind in {'markdown', 'html'}:
        return parse_markup(doc)
    if kind == 'pdf':
        return parse_one(doc)
    # 미지원 결과도 같은 4키 구조 {'doc', 'status', 'units', 'reason'} 로 맞춘다 (units 는 빈 목록)
    return {
        'doc': {'doc_id': doc['doc_id']},
        'status': 'unsupported',
        'units': [],
        'reason': 'unsupported_extension',
    }
    # END D24-02-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# manifest 6건을 모두 분기 처리하고 상태별 건수와 함께 저장한다.
def main():
    rows = load_json(DATA / 'manifest.json')
    results = [parse_dispatch(r) for r in rows]
    # 저장 dict 키 : 의미
    # - items         : 문서별 Parsing 결과 6건
    # - status_counts : 상태별 건수 (예: {'needs_ocr': 1, 'parsed': 4, 'unsupported': 1})
    out = save_json(
        HERE,
        'parsing_report.json',
        {
            'items': results,
            'status_counts': {
                s: sum(r['status'] == s for r in results)
                for s in sorted({r['status'] for r in results})
            },
        },
    )
    show_success('02 Intermediate', out)
