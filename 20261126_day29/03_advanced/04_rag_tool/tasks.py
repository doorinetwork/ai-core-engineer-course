"""
04 · 기존 RAG를 rag_search Tool로 · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

검색 요청을 검사해 정리하고, 11/20 DB 에 이미 있는 overview 벡터로 실제 RAG 답변을 받는 두 단계
"""

# ============================================================
# 1. 검색 요청 검사·정리 : prepare_search() 
# ============================================================
# prepare_search : 실제 6개 카테고리만 허용하고 query/category를 반환.
# - query    : 앞뒤 공백 제거 후 1~500자여야 함 (빈 문자열·공백만 -> ValueError)
# - category : 대문자로 바꾼 뒤 'ALL' 또는 6개 Category 코드만 허용 (그 외 -> ValueError)
# 예: prepare_search('보고서 찾아줘', 'all') -> {'query': '보고서 찾아줘', 'category': 'ALL'}
# 예: prepare_search(' ', 'ALL')            -> ValueError
def prepare_search(query: str, category: str = 'ALL') -> dict:
    # 해설 D29-04-A-01: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    # 제공 모듈 coursekit/contracts.py : 검색어 길이·Category 허용 목록을 검사하는 search_request()
    from coursekit.contracts import search_request
    query, category = search_request(query, category)
    return {'query': query, 'category': category}


# ============================================================
# 2. 실제 RAG 호출 : invoke_rag() 
# ============================================================
# invoke_rag : 실제 기존 DB의 pgvector/CUDA RAG를 호출합니다. 가짜 검색 결과 금지.
# - request : prepare_search() 반환 dict ('query'·'category' 키 사용)
# 반환 dict (coursekit/rag.py rag_answer())
# - status    : 'answered' (근거 있음) / 'insufficient_context' (유사도 기준을 넘는 근거 없음)
# - answer    : LLM 답변 문자열 (근거 부족이면 고정 안내 문장, LLM 호출 안 함)
# - sources   : 실제 Context 에 들어간 보고서 출처 목록 [{source_id: 'RPT-63', ...}]
# - retrieval : candidate_ids / ranked_ids / included_ids (후보 -> 재정렬 -> Context 포함)
def invoke_rag(request: dict) -> dict:
    # 해설 D29-04-A-02: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    # 제공 모듈 coursekit/rag.py : 실제 E5·pgvector·CrossEncoder·LLM 으로 근거 답변을 만드는 rag_answer()
    from coursekit.rag import rag_answer
    return rag_answer(request['query'], request['category'])

# ============================================================
# 3. 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_04 import format_evidence, validate_evidence
