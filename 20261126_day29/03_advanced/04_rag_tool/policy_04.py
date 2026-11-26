"""
경계값·오류/권한 정책: Advanced · Day29.

RAG 결과에서 답변·출처 ID·상태를 분리해 보여 주고, 출처가 실제 Context 에 포함된 보고서에서만 나왔는지 검사
"""

# ============================================================
# 1. 근거 표시 정리 : format_evidence() 
# ============================================================
# format_evidence : RAG 답변·실제 Context 출처·상태를 구분해 결과로 반환.
# 반환 dict
# - status     : 'answered' / 'insufficient_context'
# - answer     : 답변 문자열
# - source_ids : sources 목록에서 source_id 만 꺼낸 목록 (예: ['RPT-63', 'RPT-66'])
def format_evidence(result: dict) -> dict:
    # 해설 D29-04-A-03: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    return {
        'status': result['status'],
        'answer': result['answer'],
        'source_ids': [x['source_id'] for x in result['sources']],
    }

# ============================================================
# 2. 출처 계약 검사 : validate_evidence() 
# ============================================================
# validate_evidence : 선택된 Context 내 출처만 인정하고 No-answer 출처를 거절합니다.
# - verified : retrieval['included_ids'] (실제 Context 에 들어간 보고서 번호)를 'RPT-번호' 로 바꾼 집합
# - validate_source_contract() 가 ValueError 를 내는 경우
#   - sources 가 list 가 아님 / source_id 누락·중복
#   - status 가 insufficient_context·not_found 인데 출처가 붙어 있음
#   - verified 집합 밖의 출처가 있음
# 예: included_ids [63, 66] -> verified {'RPT-63', 'RPT-66'}, sources 에 'RPT-999' 가 있으면 ValueError
def validate_evidence(result: dict) -> bool:
    # 해설 D29-04-A-04: 실제 CUDA E5 query: 인코딩으로 11/20의 overview Vector를 검색합니다.
    # 검수 기준: CUDA CrossEncoder/LLM과 출처 ID 검증의 책임을 분리합니다.
    # 제공 모듈 coursekit/contracts.py : 출처 목록이 검증된 ID 집합 안에 있는지 검사하는 validate_source_contract()
    from coursekit.contracts import validate_source_contract
    verified = {f"RPT-{rid}" for rid in result['retrieval']['included_ids']}
    return validate_source_contract(result, verified)
