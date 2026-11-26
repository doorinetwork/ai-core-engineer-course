"""
01 · AI Agent vs 고정 RAG/Workflow · 기본필수 · TODO 완성본 (mentor 정답 기준).

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

사용자 질문을 보고 "문서 검색(rag) / DB 상태 조회(db) / 둘 다(agent)" 중 어떤 처리 전략이 필요한지 고르는 규칙 기반 라우터
"""

# ============================================================
# 1. 질문 성격 분류 : classify_request() 
# ============================================================
# classify_request : 질문 성격을 'rag'·'db'·'agent' 중 선택. 규정+상태 복합은 agent.
# - DB 상태 단어와 문서 검색 단어가 모두 있음 : 'agent' (두 Tool 이 모두 필요한 복합 요청)
# - DB 상태 단어만 있음                       : 'db'    (lookup_status 만 필요)
# - 그 외 (상태 단어 없음)                    : 'rag'   (rag_search 만 필요)
# 예: 'RAG 근거 보고서 내용을 찾아 63번 인덱싱 상태도 알려줘' -> 'agent'
# 예: '63번 인덱싱 상태'                                   -> 'db'
# 예: '근거 인용 정확도 향상 계획'                          -> 'rag'
def classify_request(question: str) -> str:
    # 해설 D29-01-B-01: 업무 질문에서 문서 근거 조회와 구조화 상태 조회를 구분합니다.
    # 검수 기준: 규칙 기반 라우팅을 LLM Agent의 자율 선택과 혼동하지 않습니다.
    # --------------------------------------------------------
    # 키워드 검사 준비
    # --------------------------------------------------------
    # lower() : 'Revision', 'READY' 처럼 대문자로 들어와도 같은 키워드로 판단하기 위해 소문자로 통일
    q = question.lower()
    # has_db  : reports.index_status·revision 같은 구조화 상태를 묻는 단어가 하나라도 있는지
    has_db = any(t in q for t in ('상태', '인덱싱', 'revision', 'ready', 'pending'))
    # has_doc : 보고서 본문 내용·근거를 찾아 달라는 단어가 하나라도 있는지
    has_doc = any(t in q for t in ('찾아', '내용', '설명', '근거', '검색'))
    # --------------------------------------------------------
    # 분류 결정 (복합 -> DB 단독 -> 나머지는 문서 검색)
    # --------------------------------------------------------
    # 주의: '상태'가 들어 있으면 문서 단어가 없어도 'db'. 상태 단어가 없으면 문서 단어가 없어도 'rag'
    if has_db and has_doc:
        return 'agent'
    if has_db:
        return 'db'
    return 'rag'

# ============================================================
# 2. 처리 계획 작성 : plan_for_request()
# ============================================================
# plan_for_request : 처리 전략과 사용할 Tool 목록을 반환. 조건문은 Agent 자체가 아닙니다.
# 반환 dict
# - strategy : classify_request() 결과 ('rag' / 'db' / 'agent')
# - tools    : 전략별 사용할 Tool 이름 목록 (coursekit/agent.py 에 등록된 이름과 같아야 함)
# - decision : 이 계획이 고정 조건으로 만든 실습용 계획이라는 설명 문장
# 예: 'agent' -> {'strategy': 'agent', 'tools': ['rag_search', 'lookup_status'], 'decision': '...'}
def plan_for_request(question: str) -> dict:
    # 해설 D29-01-B-02: 업무 질문에서 문서 근거 조회와 구조화 상태 조회를 구분합니다.
    # 검수 기준: 규칙 기반 라우팅을 LLM Agent의 자율 선택과 혼동하지 않습니다.
    mode = classify_request(question)
    # 전략 -> Tool 목록 대응표
    # - 'rag'   : 문서 근거 검색 Tool 하나
    # - 'db'    : 인덱싱 상태 조회 Tool 하나
    # - 'agent' : 두 Tool 모두 (순서는 계획일 뿐이며 실제 호출 순서를 보장하지 않음)
    tools = {
        'rag': ['rag_search'],
        'db': ['lookup_status'],
        'agent': ['rag_search', 'lookup_status'],
    }[mode]
    return {
        'strategy': mode,
        'tools': tools,
        'decision': '고정 조건에 따른 실습용 계획; 실제 Agent 선택은 06에서 LLM 수행',
    }
