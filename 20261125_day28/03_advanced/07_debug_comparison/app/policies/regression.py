"""
Day28 실습 07 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

회귀·오류 기반 배포 보류 판정 함수
"""

# ============================================================
# 회귀·배포 보류 판정 : regression_decision() 
# ============================================================
# regression_decision : Recall 감소(회귀)나 오류가 있으면 배포 보류로 판정
# - 반환 dict
#   - regression   : before·after 가 모두 있고 after < before 이면 True
#   - error_count  : 실행 오류 수
#   - hold_release : 회귀 또는 오류가 있으면 True
#   - reason       : '회귀/오류 발견' 또는 '추가 사람 평가 필요'
# 주의: hold_release False 여도 배포 승인이 아니라 사람 평가가 남았다는 뜻
def regression_decision(before: float | None, after: float | None, error_count: int) -> dict:
    # 정답 D28-07-A04: Baseline보다 Recall이 감소하거나 오류가 있으면 배포 보류를 반환하고 이유를 기록하세요.
    # 절차: 이 단계의 Gate만 통과했다고 실제 배포 승인이 되는 것은 아닙니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    regression = (before is not None and after is not None and after < before)
    return {'regression': regression, 'error_count': error_count,
            'hold_release': regression or error_count > 0,
            'reason': '회귀/오류 발견' if regression or error_count > 0 else '추가 사람 평가 필요'}
