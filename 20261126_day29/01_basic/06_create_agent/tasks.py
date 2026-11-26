"""
06 · create_agent + 실제 Tool 2개 · 기본필수.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

내용 검색과 인덱싱 상태를 함께 묻는 질문을 만들고, rag_search·lookup_status 두 Tool 을 등록한 LangChain Agent 를 만든다
"""

# ============================================================
# Agent 질문 만들기 : question_for_agent() 
# ============================================================
# question_for_agent : 내용 검색 + 인덱싱 상태를 모두 묻는 자연어 질문을 만듭니다.
# - report_id=63 : '...보고서 내용을 설명하고, 63번 보고서의 인덱싱 상태도 알려줘.'
# - report_id=0  : ValueError (질문을 만들기 전에 ID 부터 검사)
def question_for_agent(report_id: int) -> str:
    # 해설 D29-06-B-01: create_agent에 실제 rag_search와 lookup_status 두 도구를 등록합니다.
    # 검수 기준: 모델의 Tool 선택은 비결정적이므로 요청 건수와 실제 ToolMessage를 따로 셉니다.
    # 제공 모듈 coursekit/contracts.py : 양의 정수 report_id 만 통과시키는 report_id_from()
    from coursekit.contracts import report_id_from
    rid = report_id_from(report_id)
    # 앞부분은 rag_search, 뒷부분은 lookup_status 가 필요하도록 두 요청을 한 문장에 담음
    return f"RAG 근거 인용 정확도 향상과 관련된 보고서 내용을 설명하고, {rid}번 보고서의 인덱싱 상태도 알려줘."

# ============================================================
# 2. Agent 만들기 : make_agent() 
# ============================================================
# make_agent : create_agent + 등록 Tool 2개 + LLM Model + Tool Call 상한을 설정.
# create_agent() 인자
# - model         : ChatOpenAI (Tool Calling 지원 LLM)
# - tools         : coursekit/agent.py build_tools() 의 [rag_search, lookup_status]
# - system_prompt : 두 Tool 의 용도·금지 사항을 적은 SYSTEM_PROMPT
# - middleware    : Tool 호출·모델 호출 횟수 상한 (넘으면 오류로 중단)
def make_agent():
    # 해설 D29-06-B-02: create_agent에 실제 rag_search와 lookup_status 두 도구를 등록합니다.
    # 검수 기준: 모델의 Tool 선택은 비결정적이므로 요청 건수와 실제 ToolMessage를 따로 셉니다.
    # 제공 모듈 coursekit/agent.py : Tool 2개를 만드는 build_tools(), Agent 시스템 지시문 SYSTEM_PROMPT
    from coursekit.agent import build_tools, SYSTEM_PROMPT
    # 제공 모듈 coursekit/settings.py : LLM 키·모델 확인 require_llm_settings(), .env 설정값 settings()
    from coursekit.settings import require_llm_settings, settings
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware
    from langchain_openai import ChatOpenAI

    # --------------------------------------------------------
    # LLM 설정 확인 (Agent 를 만들기 전에 키·모델 누락을 먼저 차단)
    # --------------------------------------------------------
    require_llm_settings()

    # --------------------------------------------------------
    # Agent 생성
    # --------------------------------------------------------
    # ChatOpenAI 인자
    # - model=settings().llm_model : .env 의 LLM_MODEL (Tool Calling 지원 모델)
    # - timeout=45                 : API 응답을 최대 45초 기다림
    # - max_retries=1              : 실패 시 1번만 재시도 (무한 재시도·비용 증가 방지)
    # middleware 인자
    # - ToolCallLimitMiddleware(run_limit=4, exit_behavior='error')  : 한 번 실행에서 Tool 호출 4번 초과 시 오류
    # - ModelCallLimitMiddleware(run_limit=6, exit_behavior='error') : 한 번 실행에서 LLM 호출 6번 초과 시 오류
    # 주의: 기본필수 정답은 Tool 상한을 4로 고정한다 (coursekit/agent.py 는 .env 의 MAX_TOOL_CALLS 를 사용)
    return create_agent(
        model = ChatOpenAI(model = settings().llm_model, timeout = 45, max_retries = 1),
        tools = build_tools(),
        system_prompt = SYSTEM_PROMPT,
        middleware = [
            ToolCallLimitMiddleware(run_limit = 4, exit_behavior = 'error'),
            ModelCallLimitMiddleware(run_limit = 6, exit_behavior = 'error'),
        ],
    )
