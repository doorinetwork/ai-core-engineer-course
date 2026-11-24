"""
단계별 역할 분리: schemas/request.py / 독립 실행은 main.py.
"""

# ============================================================
# 1. Pipeline 요청 생성 : prepare_pipeline_request() 
# ============================================================
# prepare_pipeline_request : 질문 dict 를 RAG Pipeline 입력(RagRequest 형식) dict 로 만듦
# - 반환 dict
#   - query          : 원 질문 (검색용 질문은 Chain 안 Rewrite 단계가 만듦)
#   - history        : 이전 대화 Turn 목록
#   - category       : 검색 Category
#   - candidate_k    : 검색 후보 수 5
#   - context_k      : Context 근거 수 2
#   - use_rewrite    : True  (History 반영 Rewrite 사용)
#   - use_multiquery : False (추가 질문 검색 사용 안 함)
#   - use_rerank     : True  (CrossEncoder 재정렬 사용)
def prepare_pipeline_request(case):
    # BEGIN D27-06-A01
    # TODO D27-06-A01 완료: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 1. 입력: case (필요한 값만 받습니다.)
    # 2. 절차: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 주의: role(권한)은 요청 dict 에 넣지 않음 (서버 코드가 trusted_role 로 지정)
    return {'query': case['question'],
            'history': case['history'],
            'category': case['category'],
            'candidate_k': 5,
            'context_k': 2,
            'use_rewrite': True,
            'use_multiquery': False,
            'use_rerank': True}
    # END D27-06-A01
