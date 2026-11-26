"""
03 · @tool·Pydantic·JSON Schema · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

보고서 ID 입력 검사 함수와, 실제 DB 상태 조회 함수를 LangChain @tool 로 등록하는 함수
"""

# ============================================================
# 보고서 ID 입력 검사 : validate_id() 
# ============================================================
# validate_id : 실제 reports.report_id인 양의 정수만 허용합니다.
# - 63, '63'               : 63
# - -1, 0, '063', 6.3, True : ValueError (반올림·추측 변환을 하지 않음)
# 주의: report_id 는 11/20 DB 의 보고서 번호이며 강의 예시의 task_id 가 아니다
def validate_id(report_id: int) -> int:
    # 해설 D29-03-A-01: 모델에게 노출될 인수 Schema와 실제 Python Tool의 실행을 별도 관찰합니다.
    # 검수 기준: report_id는 실제 DB의 양의 정수이며 task_id가 아닙니다.
    # 제공 모듈 coursekit/contracts.py : bool·소수·음수·선행 0 문자열을 거절하는 report_id_from()
    from coursekit.contracts import report_id_from
    return report_id_from(report_id)

# ============================================================
# 상태 조회 Tool 등록 : register_status_tool()
# ============================================================
# register_status_tool : @tool을 사용해 실제 DB 상태 조회 함수를 등록하고 반환합니다.
# 반환 Tool 객체
# - name        : 'lookup_status' (모델이 호출할 때 쓰는 이름)
# - description : 안쪽 함수의 docstring 이 그대로 Tool 설명이 됨
# - args_schema : 함수 타입 힌트(report_id: int)로 자동 생성된 Pydantic 입력 Schema
# - invoke({'report_id': 63}) : 실제 DB 조회 후 JSON 문자열 반환
def register_status_tool():
    # 해설 D29-03-A-02: 모델에게 노출될 인수 Schema와 실제 Python Tool의 실행을 별도 관찰합니다.
    # 검수 기준: report_id는 실제 DB의 양의 정수이며 task_id가 아닙니다.
    from langchain.tools import tool
    import json
    # 제공 모듈 coursekit/db.py : READ ONLY 트랜잭션으로 보고서 인덱싱 상태를 SELECT 하는 fetch_report_status()
    from coursekit.db import fetch_report_status

    # --------------------------------------------------------
    # @tool 로 Python 함수를 Tool 로 등록
    # --------------------------------------------------------
    # @tool('lookup_status') : 함수 이름 대신 'lookup_status' 를 Tool 이름으로 사용
    # 주의: 아래 docstring 은 Tool description 으로 쓰이는 실행 요소다. 지우면 @tool 이 ValueError 를 낸다
    @tool('lookup_status')
    def lookup_status(report_id: int) -> str:
        """기존 보고서 report_id의 인덱싱 상태를 읽기 전용 조회합니다. task_id가 아닙니다."""
        # 입력 검사 -> 실제 DB SELECT -> JSON 문자열 변환 순서
        # json.dumps 인자
        # - ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 저장
        # - default=str        : datetime 등 JSON 기본 타입이 아닌 값을 문자열로 변환
        return json.dumps(fetch_report_status(validate_id(report_id)), ensure_ascii = False, default = str)

    return lookup_status

# ============================================================
# 3. 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_03 import schema_fields, enforce_explicit_schema
