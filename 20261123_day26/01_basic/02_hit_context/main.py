"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다.

실습 02 Hit -> Context · 기본필수
- 역할: 검색 Hit를 "[ID] 위치 + 줄바꿈 + 본문" 블록으로 만들고, 앞의 최대 max_items건으로
        Context 문자열과 같은 선택의 included_ids를 함께 만든다
- 기능: format_hit()(D26-02-B01) 블록 문자열, make_context()(D26-02-B02) Context·포함 ID
- 결과: outputs/hit_context.json (included_ids [201], character_count = Context 글자 수)
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# ROLE_ROOT : HERE 기준 두 단계 위 = 역할 폴더(mentee). coursekit/·data/가 있는 곳
ROLE_ROOT = HERE.parents[1]
# 역할 폴더를 import 경로 맨 앞에 넣어 어느 위치에서 실행해도 coursekit을 찾는다
sys.path.insert(0, str(ROLE_ROOT))
# 제공 모듈 coursekit/io.py: data/ 경로·JSON 읽기·실습별 outputs/ 저장·완료 메시지
from coursekit.io import save_json, show_success
# 제공 모듈 coursekit/data.py: 강의 예시 Hit 3건(lecture_hits.json)·질문 5개 읽기
from coursekit.data import example_hit, load_hits
# 제공 모듈 coursekit/reporting.py: 실행 모드·AIMessage·bundle을 JSON 기록용 dict로 바꾸기
from coursekit.reporting import execution_record

# ============================================================
# 2. Hit 한 건을 출처 블록 문자열로
# ============================================================
# Hit의 출처 번호·원문 위치·본문을 모델이 읽을 한 블록으로 만든다.
# - 입력 : {'citation_id': 201, 'source_locator': 'attendance_v2.md#sec-1', 'body': '수업 시작 후 15분까지 …'}
# - 반환 : '[201] attendance_v2.md#sec-1\n수업 시작 후 15분까지 …'
def format_hit(hit: dict) -> str:
    # Hit 한 건의 ID·위치·본문을 한 문자열 블록으로 반환합니다.
    # BEGIN D26-02-B01
    # TODO D26-02-B01 완료: [ID] 원문위치, 줄바꿈, 본문 순서로 한 블록을 만듭니다.
    # 필요한 세 값을 이름 있는 변수로 꺼낸다 (키가 없으면 KeyError로 바로 드러남)
    # 모델에 보낼 본문 바로 위에 출처 ID와 원문 위치를 표시합니다.
    # f-string의 중괄호에는 변수 값이 들어가고, \n은 실제 줄바꿈입니다.
    # 이 함수는 표시 형식만 바꾸며 검색·생성·사실 검증을 수행하지 않습니다.
    citation_id = hit["citation_id"]
    locator = hit["source_locator"]
    body = hit["body"]
    # f-string : {변수} 자리에 값이 들어가고 \n 은 실제 줄바꿈 -> 첫 줄 = 출처, 둘째 줄부터 = 본문
    return f"[{citation_id}] {locator}\n{body}"
    # END D26-02-B01

# ============================================================
# 3. Context와 포함 ID 만들기 
# ============================================================
# 고른 Hit들의 블록을 빈 줄로 이어 Context를 만들고, 같은 반복에서 citation_id를 모은다.
# - [hit201]           : ('[201] attendance_v2.md#sec-1\n…', [201])
# - [hit201, hit202]   : ('[201] …\n\n[202] assignment.txt\n…', [201, 202])
# - []                 : ('', [])
def make_context(hits: list[dict], max_items: int = 2) -> tuple[str, list[int]]:
    # 앞에서 최대 max_items건을 골라 Context와 같은 선택의 ID 목록을 반환합니다.
    # BEGIN D26-02-B02
    # TODO D26-02-B02 완료: 앞의 최대 2건에서 블록과 ID를 함께 모아 반환합니다.
    # hits[:max_items] : 앞에서 최대 max_items건. 2건을 보장하지 않는다
    # 슬라이스는 실제 건수가 1건이면 1건만 고릅니다. 2건을 보장하는 명령이 아닙니다.
    # 같은 반복문에서 본문과 ID를 모아 둘의 출처가 어긋나지 않게 합니다.
    # 빈 목록에서는 join 결과가 빈 문자열이고 ID도 빈 목록입니다.
    selected_hits = hits[:max_items]
    blocks = []
    included_ids = []
    # 블록과 ID를 같은 반복에서 함께 추가 -> 두 목록의 순서·건수가 항상 같다
    for hit in selected_hits:
        blocks.append(format_hit(hit))
        included_ids.append(hit["citation_id"])
    # '\n\n'.join(blocks) : 블록 사이에 빈 줄 하나. 블록이 없으면 '' 반환
    return ("\n\n".join(blocks), included_ids)
    # END D26-02-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 예시 Hit 1건으로 Context를 만들어 outputs/hit_context.json 으로 저장한다.
def main():
    # example_hit() : lecture_hits.json 첫 Hit (201 출석 문서)
    hit = example_hit()
    context, included_ids = make_context([hit])
    # report 키 : 의미
    # - execution       : execution_record('local_python') -> db·embedding·llm 모두 False
    # - context         : make_context([hit]) 의 Context 문자열
    # - included_ids    : 같은 선택의 citation_id 목록 ([201])
    # - input_source    : 'lecture_fixture' (강의 고정 예시, 실제 검색 결과 아님)
    # - character_count : len(context). 문자 수이며 Token 수가 아니다
    report = {
        "execution": execution_record("local_python"),
        "context": context,
        "included_ids": included_ids,
        "input_source": "lecture_fixture",
        "character_count": len(context),
    }
    # HERE/outputs/hit_context.json 저장 후 경로 반환
    path = save_json(HERE, "hit_context.json", report)
    show_success("02 기본필수", path)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
