"""실습 06 Intermediate · 버전 쌍과 분류 조건.

실습 06 Metadata · 응용
- 역할: 청크 Metadata 를 검사하고 현재 공개 버전 쌍(source_version, pipeline_version)과 category 로 검색 후보를 고른다
- 기능: require_metadata()(I01) 필수 필드, validate_locator()(I02) Span·page_number,
        select_current()(I03) 현재 버전 쌍 선택, filter_and_report()(I04) 필터와 제외 사유 집계
- 결과: outputs/metadata_filter.json (3건 중 C2 선택, 다른 버전 1건·다른 분류 1건 제외)
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

# ============================================================
# 2. 필수 Metadata 필드 검사 (TODO D24-06-I01)
# ============================================================
# 여섯 필드가 모두 비어 있지 않은 문자열인지 확인한다.
# - 필드: doc_id, source_version, pipeline_version, source_locator, category, chunk_id
# - 하나라도 없거나 '' / '  ' / 숫자 : ValueError('missing <필드명>')
def require_metadata(row: dict) -> None:
    # BEGIN D24-06-I01
    # 개념·실무 해설(mentor 원문): Metadata는 본문이 아닌 관리 정보입니다. 문서 ID·원문 판·처리 판·위치를 함께 가져야 검색 결과의 근거를 설명할 수 있습니다.
    # TODO D24-06-I01 완료: 필수 Metadata 필드의 존재와 비어 있지 않음을 검사합니다.
    for key in (
        'doc_id',
        'source_version',
        'pipeline_version',
        'source_locator',
        'category',
        'chunk_id',
    ):
        # row.get(key) : 키가 없으면 None -> isinstance(None, str) False -> 오류
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'missing {key}')
    # END D24-06-I01

# ============================================================
# 3. Span·page_number 검사 (TODO D24-06-I02)
# ============================================================
# 필수 필드 확인 후 문자 범위와 페이지 번호의 자료형·조건을 검사한다.
# - char_start/char_end : 정수이고 0 <= start < end
# - page_number         : None(TXT·Markdown·HTML) 이거나 1 이상 정수(PDF)
# - page_number 0 / '1' : ValueError('invalid page')
def validate_locator(row: dict) -> None:
    # BEGIN D24-06-I02
    # 개념·실무 해설(mentor 원문): 문자 Span과 PDF page_number는 서로 다른 위치 단위입니다. TXT/Markdown에 존재하지 않는 페이지 번호를 만들어 넣지 않습니다.
    # TODO D24-06-I02 완료: 문자 범위와 PDF page_number 등 자료형·조건을 확인합니다.
    require_metadata(row)
    a, b = (row.get('char_start'), row.get('char_end'))
    if type(a) is not int or type(b) is not int or (not 0 <= a < b):
        raise ValueError('invalid span')
    # 주의: TXT·Markdown 에는 페이지가 없으므로 None 이 정상이다 (가짜 1페이지를 만들지 않음)
    page = row.get('page_number')
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError('invalid page')
    # END D24-06-I02

# ============================================================
# 4. 현재 공개 버전 쌍 선택 (TODO D24-06-I03)
# ============================================================
# 문서별 공개 포인터 (source_version, pipeline_version) 와 정확히 같은 청크만 고른다.
# - pointers {'DOC-ATTEND': ('v2', 'ingest-e5-v1')}
# - C1 (v1, ingest-e5-v1) : 제외 (원문 판 다름)
# - C2 (v2, ingest-e5-v1) : 선택
# - 포인터가 없는 문서    : pointers.get 이 None 이라 제외
def select_current(rows: list[dict], pointers: dict) -> list[dict]:
    # BEGIN D24-06-I03
    # 개념·실무 해설(mentor 원문): category는 검색 편의 필터이지 사용자 접근 권한이 아닙니다. 실제 허용 문서 ID는 서버 측 인증/정책 확인 결과를 사용해야 합니다.
    # TODO D24-06-I03 완료: 현재 공개 중인 source_version·pipeline_version 쌍만 선택합니다.
    chosen = []
    for row in rows:
        match = pointers.get(row['doc_id'])
        if match == (row['source_version'], row['pipeline_version']):
            chosen.append(row)
    return chosen
    # END D24-06-I03

# ============================================================
# 5. 필터와 제외 사유 집계 (TODO D24-06-I04)
# ============================================================
# 검증 실패 -> 공개 버전 불일치 -> 분류 불일치 순서로 걸러 내고 단계별 제외 수를 기록한다.
def filter_and_report(rows: list[dict], pointers: dict, category: str) -> dict:
    # BEGIN D24-06-I04
    # 개념·실무 해설(mentor 원문): 현재 공개 포인터는 (source_version,pipeline_version) 쌍입니다. 둘 중 하나만 검사하면 다른 정제·모델 계약의 청크가 섞입니다.
    # TODO D24-06-I04 완료: category 필터 후 반환 건수와 제외 이유를 기록합니다.

    # ----------------------------------------------------
    # 5-1. Metadata 검증 (실패는 사유와 함께 invalid 로)
    # ----------------------------------------------------
    valid = []
    invalid = []
    for row in rows:
        try:
            validate_locator(row)
        except ValueError as exc:
            invalid.append({'chunk_id': row.get('chunk_id'), 'reason': str(exc)})
        else:
            valid.append(row)

    # ----------------------------------------------------
    # 5-2. 현재 버전 쌍 -> category 필터
    # ----------------------------------------------------
    current = select_current(valid, pointers)
    selected = [r for r in current if r['category'] == category]
    # 반환 dict 키 : 의미
    # - total                   : 입력 청크 수
    # - selected                : 최종 검색 후보
    # - invalid                 : Metadata 오류 청크와 사유
    # - excluded_other_version  : 검증은 통과했지만 공개 버전 쌍이 아닌 수
    # - excluded_other_category : 공개 버전이지만 category 가 다른 수
    return {
        'total': len(rows),
        'selected': selected,
        'invalid': invalid,
        'excluded_other_version': len(valid) - len(current),
        'excluded_other_category': len(current) - len(selected),
    }
    # END D24-06-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# 같은 출처의 청크 3건(v1 학사, v2 학사, v2 시설)에 v2 공개 포인터와 '학사' 필터를 적용한다.
def main():
    # base : 세 청크가 공유하는 Metadata (DOC-ATTEND, 학사, ingest-e5-v1, Span 0~20)
    base = {
        'doc_id': 'DOC-ATTEND',
        'source_locator': 'attendance.md#sec-1',
        'category': '학사',
        'pipeline_version': 'ingest-e5-v1',
        'char_start': 0,
        'char_end': 20,
        'page_number': None,
    }
    # C1 : v1 (공개 판 아님) / C2 : v2 학사 (선택 대상) / C3 : v2 지만 category '시설'
    items = [
        {**base, 'chunk_id': 'C1', 'source_version': 'v1'},
        {**base, 'chunk_id': 'C2', 'source_version': 'v2'},
        {**base, 'chunk_id': 'C3', 'source_version': 'v2', 'category': '시설'},
    ]
    # filter_and_report 인자
    # - items                                : 청크 3건
    # - {'DOC-ATTEND': ('v2', 'ingest-e5-v1')} : 문서별 현재 공개 버전 쌍
    # - '학사'                               : category 필터
    report = filter_and_report(items, {'DOC-ATTEND': ('v2', 'ingest-e5-v1')}, '학사')
    out = save_json(HERE, 'metadata_filter.json', report)
    show_success('06 Intermediate', out)
