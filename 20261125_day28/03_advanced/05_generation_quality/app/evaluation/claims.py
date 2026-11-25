"""
Day28 실습 05 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Claim Faithfulness 계산·설명 함수
"""

# ============================================================
# 1. Claim Faithfulness 계산 : claim_faithfulness()
# ============================================================
# claim_faithfulness : Faithfulness = Context 가 지지하는 Claim 수 / 전체 Claim 수
# - claims 비어 있음 : None ('모두 지지'와 구별)
# - 그 외            : 0.0 ~ 1.0
# 예: G01 Claim 3개 중 supported 2개 -> 0.666..
def claim_faithfulness(claims: list[dict]) -> float | None:
    # 정답 D28-05-A01: Claim 목록에서 근거가 지지하는 Claim 비율을 계산하세요. Claim이 없으면 None.
    # 절차: 모든 Claim의 supported는 교육용 사람 레이블입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    if not claims:
        return None
    # bool(c['supported']) : True=1, False=0 으로 더해 지지 Claim 수를 셈
    return sum(bool(c['supported']) for c in claims) / len(claims)


# ============================================================
# 2. Claim 판정 설명 : describe_claim_result() 
# ============================================================
# describe_claim_result : Claim 판정 결과를 한 문장으로 설명
# - Claim 없음        : '검토할 Claim 없음: 별도 판정 필요'
# - 비지원 Claim 있음 : '근거 불충분: ' + 비지원 Claim 문장 목록
# - 모두 지원         : '모든 Claim이 제공된 Context와 일치'
# 예: G01 -> '근거 불충분: 지각은 5분부터'
def describe_claim_result(claims: list[dict]) -> str:
    # 정답 D28-05-A02: 비지원 Claim을 이름과 함께 기록하세요. “모든 Claim 지원”과 “Claim 없음”은 구별합니다.
    # 절차: Faithfulness는 Reference Answer와의 비교가 아닙니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    if not claims:
        return '검토할 Claim 없음: 별도 판정 필요'
    unsupported = [c['text'] for c in claims if not c['supported']]
    # 주의: 조건식이 + 보다 우선순위가 낮아 ('근거 불충분: ' + 목록) if unsupported else '모든 Claim…' 으로 해석됨
    return '근거 불충분: ' + ', '.join(unsupported) if unsupported else '모든 Claim이 제공된 Context와 일치'
