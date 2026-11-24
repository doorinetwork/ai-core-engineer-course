"""
단계별 역할 분리: checks.py / 독립 실행은 main.py.
"""

# ============================================================
# 1. trace 기록 검사 : validate_trace()
# ============================================================
# validate_trace : Pipeline 결과 trace 에 원 질문·검색 질문·후보 ID·Context ID 가 제대로 남았는지 확인
# - 반환 dict
#   - questions_separate       : original_question 과 search_query 가 둘 다 비어 있지 않은가
#   - ids_recorded             : candidate_ids·included_ids 가 list 로 기록됐는가
#   - included_from_candidates : Context 에 넣은 ID 가 모두 후보 ID 안에 있는가
def validate_trace(result):
    # BEGIN D27-06-I03
    # TODO D27-06-I03 완료: 원 질문·검색용 질문·후보·Context ID를 각각 확인합니다.
    # 1. 입력: result (필요한 값만 받습니다.)
    # 2. 절차: 원 질문·검색용 질문·후보·Context ID를 각각 확인합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # trace : create_rag_pipeline() 의 make_trace() 가 만든 실행 기록
    t = result['trace']
    # issubset() : included_ids 집합이 candidate_ids 의 부분집합이면 True
    return {'questions_separate': bool(t['original_question'] and t['search_query']),
            'ids_recorded': isinstance(t['candidate_ids'], list) and isinstance(t['included_ids'], list),
            'included_from_candidates': set(t['included_ids']).issubset(t['candidate_ids'])}
    # END D27-06-I03
