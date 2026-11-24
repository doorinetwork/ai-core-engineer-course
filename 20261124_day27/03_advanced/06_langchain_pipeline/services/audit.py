"""
단계별 역할 분리: services/audit.py / 독립 실행은 main.py.
"""

# ============================================================
# 1. Pipeline 감사 기록 : audit_pipeline() 
# ============================================================
# audit_pipeline : 실행 모드·Stage 시간 목록·출력 출처 ID 를 감사 기록으로 남김
# - 반환 dict
#   - mode                    : trace.source_mode ('offline_fixture' 또는 'live')
#   - stage_times             : 시간이 기록된 Stage 이름 정렬 목록 (값이 아닌 이름만)
#   - output_source_ids       : 응답 sources 의 citation_id 목록
#   - manual_grounding_review : 항상 True (출처 ID 연결과 답변 진위는 별도로 사람이 검토)
def audit_pipeline(result):
    # BEGIN D27-06-A04
    # TODO D27-06-A04 완료: Stage 시간과 출처 연결을 기록하되 진위 판단은 별도로 수행합니다.
    # 1. 입력: result (필요한 값만 받습니다.)
    # 2. 절차: Stage 시간과 출처 연결을 기록하되 진위 판단은 별도로 수행합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    t = result['trace']
    # sorted(dict) : 키(Stage 이름)만 정렬 -> ['context', 'generation', 'multiquery', 'rerank', 'retrieval', 'rewrite']
    return {'mode': t['source_mode'],
            'stage_times': sorted(t['timing_ms']),
            'output_source_ids': [r['citation_id'] for r in result['sources']],
            'manual_grounding_review': True}
    # END D27-06-A04
