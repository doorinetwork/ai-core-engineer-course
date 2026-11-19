"""실습 06 Basic · 본문과 설명 정보를 나눕니다.

실습 06 Metadata · 기본필수
- 역할: 청크 본문에 출처 Metadata(문서 ID·원문 판·분류·위치)를 붙이고 분류(category)로 고른다
- 기능: attach_metadata()(D24-06-B01) 출처 복사, select_category()(D24-06-B02) 분류 필터
- 결과: outputs/metadata.json (전체 2건 중 '학사' 1건 선택)
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
# 2. 출처 Metadata 붙이기 (TODO D24-06-B01)
# ============================================================
# 청크 dict를 복사하고 문서의 출처 정보 네 가지를 덧붙인 새 dict를 돌려준다.
# - chunk : {'chunk_id': 'C1', 'body': '출석 안내'} 같은 본문 정보
# - doc   : doc_id·source_version·category·source_locator 를 가진 출처 정보
# - 원본 chunk dict는 바꾸지 않음 (**chunk 로 새 dict 생성)
def attach_metadata(chunk: dict, doc: dict) -> dict:
    # BEGIN D24-06-B01
    # 개념·실무 해설(mentor 원문): Metadata는 본문이 아닌 관리 정보입니다. 문서 ID·원문 판·처리 판·위치를 함께 가져야 검색 결과의 근거를 설명할 수 있습니다.
    # TODO D24-06-B01 완료: 원문 문서 ID·버전·원문 위치를 Chunk에 복사합니다.
    # 반환 dict 키 : 의미
    # - **chunk         : 기존 chunk_id·body 그대로
    # - doc_id          : 원문 문서 ID (예: 'DOC-ATTEND')
    # - source_version  : 원문 판 (예: 'v1')
    # - category        : 검색 필터용 분류 (예: '학사', '시설')
    # - source_locator  : 원문 위치 (예: 'attendance.md#sec-1')
    return {
        **chunk,
        'doc_id': doc['doc_id'],
        'source_version': doc['source_version'],
        'category': doc['category'],
        'source_locator': doc['source_locator'],
    }
    # END D24-06-B01

# ============================================================
# 3. 분류로 청크 고르기 (TODO D24-06-B02)
# ============================================================
# category가 같은 청크만 원래 순서대로 돌려준다.
# - [C1(학사), C2(시설)], '학사' -> [C1]
# - 다른 분류 청크는 결과에서 빠질 뿐 삭제·변경하지 않음
def select_category(chunks: list[dict], category: str) -> list[dict]:
    # BEGIN D24-06-B02
    # 개념·실무 해설(mentor 원문): Category 필터는 목록에서 해당 분류에 속한 행만 선택하는 처리입니다. 다른 분류의 청크를 삭제하거나 변경하지 않습니다.
    # TODO D24-06-B02 완료: 주어진 category에 속하는 Chunk만 원래 순서를 보존해서 반환합니다.
    return [chunk for chunk in chunks if chunk['category'] == category]
    # END D24-06-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 출석(학사)·시설(시설) 청크 2건에 Metadata를 붙이고 '학사'만 고른다.
def main():
    # a : 출석 안내 청크 C1 + DOC-ATTEND 출처 (category '학사')
    a = attach_metadata(
        {'chunk_id': 'C1', 'body': '출석 안내'},
        {
            'doc_id': 'DOC-ATTEND',
            'source_version': 'v1',
            'category': '학사',
            'source_locator': 'attendance.md#sec-1',
        },
    )
    # b : 시설 안내 청크 C2 + DOC-FACILITY 출처 (category '시설')
    b = attach_metadata(
        {'chunk_id': 'C2', 'body': '시설 안내'},
        {
            'doc_id': 'DOC-FACILITY',
            'source_version': 'v1',
            'category': '시설',
            'source_locator': 'facility.html#main',
        },
    )
    selected = select_category([a, b], '학사')
    # '학사' 선택 결과가 정확히 C1 하나인지 확인
    assert [r['chunk_id'] for r in selected] == ['C1']
    # 저장 dict 키 : 의미
    # - total    : 필터 전 청크 수 2
    # - selected : '학사' 청크 목록
    out = save_json(HERE, 'metadata.json', {'total': 2, 'selected': selected})
    show_success('06 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
