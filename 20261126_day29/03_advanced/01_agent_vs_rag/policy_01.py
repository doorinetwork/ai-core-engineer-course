"""
경계값·오류/권한 정책: Advanced · Day29.

처리 계획 dict 가 두 데이터 소스를 모두 쓰는지, 등록된 Tool 만 중복 없이 담았는지 검사
"""


# ============================================================
# 1. 두 데이터 소스 필요 여부 : needs_two_sources() 
# ============================================================
# needs_two_sources : 문서 검색 Tool과 DB Tool이 모두 필요한 계획인지 확인.
# - plan['tools'] 가 {'rag_search', 'lookup_status'} 와 같은 집합 : True
# - 하나만 있거나 다른 Tool 이 섞임                              : False
# 예: {'tools': ['rag_search', 'lookup_status']} -> True / {'tools': ['rag_search']} -> False
def needs_two_sources(plan: dict) -> bool:
    # 해설 D29-01-A-03: 업무 질문에서 문서 근거 조회와 구조화 상태 조회를 구분합니다.
    # 검수 기준: 규칙 기반 라우팅을 LLM Agent의 자율 선택과 혼동하지 않습니다.
    # set() 비교 : 순서와 무관하게 두 Tool 이 정확히 들어 있는지만 확인
    return set(plan['tools']) == {'rag_search', 'lookup_status'}


# ============================================================
# 2. 계획 정책 검사 : validate_plan() 
# ============================================================
# validate_plan : 등록되지 않은 Tool·중복 Tool·상태 누락을 진단.
# 아래 중 하나라도 해당하면 ValueError (Fail Closed: 애매하면 통과시키지 않음)
# - tools 가 list 가 아님        : 예 'rag_search' 문자열 하나
# - tools 가 비어 있음           : 예 []
# - 같은 Tool 이 두 번 이상 있음 : 예 ['rag_search', 'rag_search']
# - 등록되지 않은 Tool 이 있음   : 예 ['delete_report']
# 통과 : {'valid': True, 'tools': plan['tools']}
def validate_plan(plan: dict) -> dict:
    # 해설 D29-01-A-04: 업무 질문에서 문서 근거 조회와 구조화 상태 조회를 구분합니다.
    # 검수 기준: 규칙 기반 라우팅을 LLM Agent의 자율 선택과 혼동하지 않습니다.
    # allowed : coursekit/agent.py build_tools() 에 실제 등록된 Tool 이름
    allowed = {'rag_search', 'lookup_status'}
    # get('tools', []) : tools 키가 없으면 빈 목록으로 보고 아래 검사에서 거절
    names = plan.get('tools', [])
    # 검사 순서: 타입 -> 빈 목록 -> 중복(len 비교) -> 허용 목록 부분집합(<=)
    if not isinstance(names, list) or not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError('Tool 계획이 등록 목록·중복 정책을 위반합니다.')
    return {'valid': True, 'tools': names}
