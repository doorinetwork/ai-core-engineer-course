"""
용도별 추가 검증: Intermediate · Day29.

Agent 메시지 목록에서 Tool 요청 수와 실제 ToolMessage 수를 따로 셈
"""

# ============================================================
# Tool 요청·실행 집계 : summarize_tool_messages() 
# ============================================================
# summarize_tool_messages : Model 요청 수와 실제 ToolMessage 수를 별도로 집계.
# 반환 dict (coursekit/contracts.py count_tool_calls())
# - requested         : AIMessage.tool_calls 에 들어 있던 호출 ID 개수 (모델이 요청한 수)
# - executed_messages : ToolMessage 개수 (실제로 실행되어 결과가 돌아온 수)
# - unmatched         : 요청은 있었는데 ToolMessage 가 없는 호출 ID 목록
# 예: 요청 2건·결과 2건 -> {'requested': 2, 'executed_messages': 2, 'unmatched': []}
def summarize_tool_messages(messages: list) -> dict:
    # 해설 D29-06-I-03: create_agent에 실제 rag_search와 lookup_status 두 도구를 등록합니다.
    # 검수 기준: 모델의 Tool 선택은 비결정적이므로 요청 건수와 실제 ToolMessage를 따로 셉니다.
    # 제공 모듈 coursekit/contracts.py : dict·LangChain 메시지 모두에서 요청·결과를 세는 count_tool_calls()
    from coursekit.contracts import count_tool_calls
    return count_tool_calls(messages)
