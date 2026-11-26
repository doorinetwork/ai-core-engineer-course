"""
용도별 추가 검증: Intermediate · Day29.

실습 01 처리 계획이 문서 검색·DB 조회 두 데이터 소스를 모두 쓰는지 판정
"""

# ============================================================
# 두 데이터 소스 필요 여부 : needs_two_sources()
# ============================================================
# needs_two_sources : 문서 검색 Tool과 DB Tool이 모두 필요한 계획인지 확인.
# - plan['tools'] 가 {'rag_search', 'lookup_status'} 와 같은 집합 : True
# - 하나만 있거나 다른 Tool 이 섞임                              : False
# 예: {'tools': ['rag_search', 'lookup_status']} -> True / {'tools': ['rag_search']} -> False
def needs_two_sources(plan: dict) -> bool:
    # 해설 D29-01-I-03: 업무 질문에서 문서 근거 조회와 구조화 상태 조회를 구분합니다.
    # 검수 기준: 규칙 기반 라우팅을 LLM Agent의 자율 선택과 혼동하지 않습니다.
    # set() 비교 : 순서와 무관하게 두 Tool 이 정확히 들어 있는지만 확인
    return set(plan['tools']) == {'rag_search', 'lookup_status'}
