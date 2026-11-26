"""
02 · AIMessage.tool_calls / ToolMessage · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

LLM 이 내는 Tool 호출 요청(AIMessage.tool_calls 의 한 항목)과 Tool 실행 결과(ToolMessage)를 같은 호출 ID 로 묶는 과정을 dict 로 재현
"""

# ============================================================
# 1. Tool 호출 요청 만들기 : make_request() 
# ============================================================
# make_request : AIMessage.tool_calls의 항목 형태. 실제로 도구를 실행하지는 않습니다.
# 함수 호출 인자
# - tool_name='lookup_status'    : 호출할 Tool 이름 (등록된 rag_search·lookup_status 만 허용)
# - args={'report_id': 63}       : Tool 에 넘길 인수 dict
# - call_id='call_001'           : 요청과 결과를 잇는 호출 ID (빈 문자열 금지)
# 예: make_request('lookup_status', {'report_id': 63}, 'call_001')
#     -> {'name': 'lookup_status', 'args': {'report_id': 63}, 'id': 'call_001'}
# - 미등록 Tool 이름·dict 가 아닌 인수·빈 호출 ID : ValueError
def make_request(tool_name: str, args: dict, call_id: str) -> dict:
    # 해설 D29-02-A-01: Tool Call은 요청 dict이고 실제 DB 조회가 끝난 뒤 결과 메시지가 생성됩니다.
    # 검수 기준: 같은 ID로 요청과 결과를 연결하고, 다중 호출에는 중복·누락 검사를 적용합니다.
    # 제공 모듈 coursekit/contracts.py : Tool 이름·인수·호출 ID 검사 후 요청 dict 를 만드는 tool_call_request()
    from coursekit.contracts import tool_call_request
    return tool_call_request(tool_name, args, call_id)

# ============================================================
# 2. 결과를 같은 호출 ID 에 묶기 : match_result() 
# ============================================================
# match_result : 실제로 DB Tool을 호출한 뒤 결과를 같은 tool_call_id에 묶습니다.
# - call   : make_request() 가 만든 요청 dict ('id'·'name' 사용)
# - result : 실제 DB 조회 결과 dict (예: fetch_report_status(63) 반환값)
# 예: match_result({'name': 'lookup_status', 'args': {...}, 'id': 'call_001'}, {'status': 'ok', 'report_id': 63})
#     -> {'type': 'ToolMessage', 'tool_call_id': 'call_001', 'name': 'lookup_status', 'content': {...}}
# - result 가 dict 가 아님·call 에 id 없음 : ValueError
def match_result(call: dict, result: dict) -> dict:
    # 해설 D29-02-A-02: Tool Call은 요청 dict이고 실제 DB 조회가 끝난 뒤 결과 메시지가 생성됩니다.
    # 검수 기준: 같은 ID로 요청과 결과를 연결하고, 다중 호출에는 중복·누락 검사를 적용합니다.
    # 제공 모듈 coursekit/contracts.py : 요청 ID 를 tool_call_id 로 옮겨 ToolMessage 역할 dict 를 만드는 tool_result_message()
    from coursekit.contracts import tool_result_message
    return tool_result_message(call, result)

# ============================================================
# 3. 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
# 같은 폴더 policy_02.py : 경계값·오류/권한 정책 함수 validate_message_pair(), summarize_pairs()
# 이 import 로 tasks.validate_message_pair() 처럼 tasks 모듈 이름으로 호출할 수 있음
from policy_02 import validate_message_pair, summarize_pairs
