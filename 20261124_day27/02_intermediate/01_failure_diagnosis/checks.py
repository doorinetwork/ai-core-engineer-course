"""단계별 역할 분리: checks.py / 독립 실행은 main.py.
처리 결과를 점검하는 단계를 담는다.
"""

# ============================================================
# 판정 근거 기록 : reason_for_stage() 
# ============================================================
# reason_for_stage : 진단 결과와 함께 판단에 쓴 근거 플래그 3개를 기록
# - 반환 dict
#   - reason   : 진단 단계 이름 (예: 'reranking')
#   - evidence : ambiguous · gold_in_candidates · gold_in_context 원래 값
# 예: F03 -> {'reason': 'reranking', 'evidence': {'ambiguous': False, 'gold_in_candidates': True, 'gold_in_context': False}}
def reason_for_stage(case, stage):
    # BEGIN D27-01-I03
    # TODO D27-01-I03 완료: 진단할 때 사용한 근거 3개를 기록합니다.
    # 주의: 판정 근거를 다시 계산하지 않고 사례의 원래 값을 그대로 옮겨 적음
    return {'reason': stage,
            'evidence': {'ambiguous': case['ambiguous'],
                         'gold_in_candidates': case['gold_in_candidates'],
                         'gold_in_context': case['gold_in_context']}}
    # END D27-01-I03
