"""
경계값·오류/권한 정책: Advanced · Day29.

Agent 실행 결과 메시지 목록에서 Tool 요청 수·실제 ToolMessage 수를 따로 세고, 최종 AIMessage 텍스트를 꺼낸다
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
    # 해설 D29-06-A-03: create_agent에 실제 rag_search와 lookup_status 두 도구를 등록합니다.
    # 검수 기준: 모델의 Tool 선택은 비결정적이므로 요청 건수와 실제 ToolMessage를 따로 셉니다.
    # 제공 모듈 coursekit/contracts.py : dict·LangChain 메시지 모두에서 요청·결과를 세는 count_tool_calls()
    from coursekit.contracts import count_tool_calls
    return count_tool_calls(messages)

# ============================================================
# 최종 답변 추출 : final_text() 
# ============================================================
# final_text : ToolMessage가 아니라 최종 AIMessage의 텍스트를 추출합니다.
# - 뒤에서부터 보며 tool_calls 가 없는 AIMessage 를 찾음 (Tool 을 요청한 중간 AIMessage 는 건너뜀)
#   - content 가 str   : 그 문자열 반환
#   - content 가 list 등 : ValueError (Structured Output 파서 필요)
# - 끝까지 못 찾음 : ValueError('최종 AIMessage가 없습니다.')
# 주의: main.py 의 기본필수판처럼 messages[-1] 을 바로 쓰면 마지막이 ToolMessage 인 경우를 구분하지 못함
def final_text(messages: list) -> str:
    """최종 AIMessage에서 텍스트만 추출합니다."""
    from langchain_core.messages import AIMessage
    # 마지막 메시지부터 역순으로 검사
    for m in reversed(messages):
        # ToolMessage 등은 제외
        if not isinstance(m, AIMessage):
            continue
        # Tool 호출 요청 메시지는 제외
        if m.tool_calls:
            continue
        # 문자열 / Content Block 리스트 모두 처리
        text = str(m.text).strip()

        if text:
            return text
        raise ValueError(
            "최종 AIMessage에 추출 가능한 텍스트가 없습니다."
        )
    raise ValueError("최종 AIMessage가 없습니다.")