"""실습 06 Advanced · 공개 범위와 정수형 Citation Contract.

실습 06 Metadata · 심화
- 역할: Metadata 검증·안정 chunk_id 재계산 -> 서버 확인 허용 문서 -> 공개 버전·분류 필터 -> 정수 Citation 매핑 순서로 감사한다
- 기능: check_meta()(A01) 필드·Span·page, stable_id()(A02) 안정 ID, restrict_allowed()(A03) 허용 문서 제한,
        select_published()(A04) 공개 버전·분류, citation_map()(A05) 정수 ID 매핑, audit()(A06) 감사 보고
- 결과: outputs/metadata_audit.json (citation 202 1건, 권한 제외 1, 버전·분류 제외 1)
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
# 제공 모듈 coursekit/processing.py: 정제·문자 창 Chunking·Span 검증·Coverage·안정 chunk_id·E5 입력 문자열
from coursekit.processing import stable_chunk_id

# ============================================================
# 2. Metadata 필드·Span·page 검사 (TODO D24-06-A01)
# ============================================================
# 문자열 필드 7개, 정수 Span, page_number(None 또는 1 이상 정수)를 검사한다.
# - 필드: doc_id, source_version, pipeline_version, unit_id, source_locator, category, chunk_id
# - 실패 예: 'missing category', 'bad span', 'bad page'
def check_meta(row: dict) -> None:
    # BEGIN D24-06-A01
    # 개념·실무 해설(mentor 원문): Metadata는 본문이 아닌 관리 정보입니다. 문서 ID·원문 판·처리 판·위치를 함께 가져야 검색 결과의 근거를 설명할 수 있습니다.
    # TODO D24-06-A01 완료: doc_id·version·category·Span·locator와 page_number 자료형을 검증합니다.
    for name in (
        'doc_id',
        'source_version',
        'pipeline_version',
        'unit_id',
        'source_locator',
        'category',
        'chunk_id',
    ):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError('missing ' + name)

    # ----------------------------------------------------
    # 2-1. Span·page_number 검사
    # ----------------------------------------------------
    a, b = (row.get('char_start'), row.get('char_end'))
    if type(a) is not int or type(b) is not int or (not 0 <= a < b):
        raise ValueError('bad span')
    page = row.get('page_number')
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError('bad page')
    # END D24-06-A01

# ============================================================
# 3. 안정 chunk_id 계산 (TODO D24-06-A02)
# ============================================================
# doc_id·source_version·pipeline_version·unit_id·Span 으로 같은 입력이면 항상 같은 SHA-256 ID 를 만든다.
# - 같은 내용 재실행 : 같은 ID
# - 원문 판(v1 -> v2) 이나 처리 판이 바뀜 : 다른 ID
def stable_id(row: dict) -> str:
    # BEGIN D24-06-A02
    # 개념·실무 해설(mentor 원문): 문자 Span과 PDF page_number는 서로 다른 위치 단위입니다. TXT/Markdown에 존재하지 않는 페이지 번호를 만들어 넣지 않습니다.
    # TODO D24-06-A02 완료: doc_id·원문 판·처리 판·unit_id·Span으로 안정적인 chunk_id를 만듭니다.
    # stable_chunk_id 인자
    # - row                     : doc_id·source_version 을 읽을 dict
    # - row['unit_id']          : 단위 ID (예: 'sec-1')
    # - row['pipeline_version'] : 처리 판 (예: 'ingest-e5-v1')
    # - row['char_start'] / row['char_end'] : Span
    return stable_chunk_id(
        row,
        row['unit_id'],
        row['pipeline_version'],
        row['char_start'],
        row['char_end'],
    )
    # END D24-06-A02

# ============================================================
# 4. 허용 문서로 제한 (TODO D24-06-A03)
# ============================================================
# 서버가 확인한 허용 doc_id 집합(set)에 있는 청크만 남긴다.
# - allowed {'DOC-ATTEND'} -> DOC-FACILITY 청크 제외
# - list·문자열 등 set 이 아님 -> ValueError (출처가 불분명한 권한 값 거부)
# 주의: category 는 검색 편의 필터일 뿐 접근 권한이 아니다
def restrict_allowed(rows: list[dict], allowed_doc_ids: set[str]) -> list[dict]:
    # BEGIN D24-06-A03
    # 개념·실무 해설(mentor 원문): category는 검색 편의 필터이지 사용자 접근 권한이 아닙니다. 실제 허용 문서 ID는 서버 측 인증/정책 확인 결과를 사용해야 합니다.
    # TODO D24-06-A03 완료: 서버가 확인한 allowed_doc_ids 목록으로만 후보를 제한합니다.
    if not isinstance(allowed_doc_ids, set):
        raise ValueError('server-verified set required')
    return [r for r in rows if r.get('doc_id') in allowed_doc_ids]
    # END D24-06-A03

# ============================================================
# 5. 공개 버전·분류 필터 (TODO D24-06-A04)
# ============================================================
# category 가 같고 문서의 공개 포인터 (source_version, pipeline_version) 와 정확히 같은 청크만 고른다.
# - pointers {'DOC-ATTEND': ('v2', 'ingest-e5-v1')} -> v1 청크 제외, v2 청크 선택
def select_published(rows: list[dict], pointers: dict, category: str) -> list[dict]:
    # BEGIN D24-06-A04
    # 개념·실무 해설(mentor 원문): 현재 공개 포인터는 (source_version,pipeline_version) 쌍입니다. 둘 중 하나만 검사하면 다른 정제·모델 계약의 청크가 섞입니다.
    # TODO D24-06-A04 완료: 문서별 current source_version·pipeline_version 쌍과 category가 맞는 후보를 선택합니다.
    return [
        r
        for r in rows
        if r['category'] == category and pointers.get(r['doc_id']) == (
            r['source_version'],
            r['pipeline_version'],
        )
    ]
    # END D24-06-A04

# ============================================================
# 6. 정수 Citation 매핑 (TODO D24-06-A05)
# ============================================================
# 양의 정수 citation_id -> {chunk_id, source_locator, doc_id} 매핑을 만든다.
# - citation_id 가 0·음수·문자열·중복 : ValueError('invalid citation_id')
# - chunk_id 중복                     : ValueError('duplicate chunk_id')
# - 예: {202: {'chunk_id': '7530a2...', 'source_locator': 'DOC-ATTEND.md#sec-1', 'doc_id': 'DOC-ATTEND'}}
def citation_map(rows: list[dict]) -> dict:
    # BEGIN D24-06-A05
    # 개념·실무 해설(mentor 원문): 기존 KnowledgeAnswer.source_ids는 list[int]입니다. Chunk ID는 text이므로 DB에서 얻은 양의 정수 citation_id를 명시적으로 매핑합니다.
    # TODO D24-06-A05 완료: 양의 정수 citation_id와 문자열 chunk_id 사이의 중복 없는 매핑을 만듭니다.
    ids = [r['citation_id'] for r in rows]
    # type(n) is not int : True/False 같은 bool 도 거부 / len(ids) != len(set(ids)) : 중복 번호
    if any(type(n) is not int or n <= 0 for n in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid citation_id')
    chunks = [r['chunk_id'] for r in rows]
    if len(chunks) != len(set(chunks)):
        raise ValueError('duplicate chunk_id')
    return {
        r['citation_id']: {
            'chunk_id': r['chunk_id'],
            'source_locator': r['source_locator'],
            'doc_id': r['doc_id'],
        }
        for r in rows
    }
    # END D24-06-A05

# ============================================================
# 7. 감사 보고 (TODO D24-06-A06)
# ============================================================
# Metadata 검증 -> 안정 ID 확인 -> 허용 문서 -> 공개 버전·분류 -> Citation 순서로 거르고 단계별 제외 수를 기록한다.
def audit(rows: list[dict], pointers: dict, allowed: set[str], category: str) -> dict:
    # BEGIN D24-06-A06
    # 개념·실무 해설(mentor 원문): 문서 권한·공개 버전·필터·오류 여부를 순서대로 검사한 후 대상과 제외 사유를 분리해 보고합니다.
    # TODO D24-06-A06 완료: 정상·차단·미공개·Metadata 오류 상태를 구분해 감사 결과를 출력합니다.

    # ----------------------------------------------------
    # 7-1. Metadata 검증과 안정 ID 확인
    # ----------------------------------------------------
    good = []
    invalid = []
    for r in rows:
        try:
            check_meta(r)
        except ValueError as exc:
            invalid.append({'chunk_id': r.get('chunk_id'), 'reason': str(exc)})
        else:
            # 저장된 chunk_id 가 재계산 값과 다르면 'unstable_identity' 로 제외 (내용·판이 바뀐 청크 탐지)
            if stable_id(r) != r['chunk_id']:
                invalid.append({'chunk_id': r['chunk_id'], 'reason': 'unstable_identity'})
            else:
                good.append(r)

    # ----------------------------------------------------
    # 7-2. 허용 문서 -> 공개 버전·분류 -> Citation
    # ----------------------------------------------------
    accessible = restrict_allowed(good, allowed)
    selected = select_published(accessible, pointers, category)
    # 반환 dict 키 : 의미
    # - citations                    : citation_map 결과
    # - invalid                      : Metadata·ID 오류 목록
    # - excluded_permission          : 허용 문서가 아니라 제외된 수
    # - excluded_version_or_category : 공개 버전 또는 분류가 달라 제외된 수
    return {
        'citations': citation_map(selected),
        'invalid': invalid,
        'excluded_permission': len(good) - len(accessible),
        'excluded_version_or_category': len(accessible) - len(selected),
    }
    # END D24-06-A06

# ============================================================
# 8. 실행 흐름 (main)
# ============================================================
# 청크 3건(ATTEND v1·v2, FACILITY v1)에 v2 공개 포인터·허용 {DOC-ATTEND}·'학사' 조건을 적용한다.
def main():
    # make : 테스트 청크 1건 생성 (chunk_id 는 stable_id 로 계산해 정상 ID 로 만듦)
    def make(docid, ver, category, citation):
        r = {
            'doc_id': docid,
            'source_version': ver,
            'pipeline_version': 'ingest-e5-v1',
            'unit_id': 'sec-1',
            'source_locator': f'{docid}.md#sec-1',
            'category': category,
            'char_start': 0,
            'char_end': 12,
            'page_number': None,
            'citation_id': citation,
        }
        return {**r, 'chunk_id': stable_id(r)}
    # 201 ATTEND v1 학사 (공개 판 아님) / 202 ATTEND v2 학사 (선택) / 203 FACILITY v1 시설 (허용 문서 아님)
    items = [
        make('DOC-ATTEND', 'v1', '학사', 201),
        make('DOC-ATTEND', 'v2', '학사', 202),
        make('DOC-FACILITY', 'v1', '시설', 203),
    ]
    # audit 인자
    # - items                                : 청크 3건
    # - {'DOC-ATTEND': ('v2', 'ingest-e5-v1')} : 공개 포인터
    # - {'DOC-ATTEND'}                       : 서버가 확인했다고 가정한 허용 문서 set
    # - '학사'                               : category
    report = audit(items, {'DOC-ATTEND': ('v2', 'ingest-e5-v1')}, {'DOC-ATTEND'}, '학사')
    out = save_json(HERE, 'metadata_audit.json', report)
    show_success('06 Advanced', out)
