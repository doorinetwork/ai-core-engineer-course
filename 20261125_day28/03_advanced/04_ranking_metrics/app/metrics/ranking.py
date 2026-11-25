"""
Day28 실습 04 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 1. 첫 관련 문서 순위 : first_relevant_rank() 
# ============================================================
# first_relevant_rank : Top-k 안에서 첫 Gold 문서의 순위(1부터 시작)를 찾음
# - 찾음   : 1 ~ k 사이 정수
# - 못 찾음: None
# 예: Q01 ranked ['FAC-101-C1', 'FAC-107-C1', 'FAC-102-C1'], Gold ['FAC-102-C1'] -> 3
def first_relevant_rank(ranked: list[str], gold: list[str], k: int) -> int | None:
    # 정답 D28-04-A01: Top-k에서 첫 Gold 문서가 등장하는 1부터 시작하는 순위를 찾으세요. 없으면 None.
    # 절차: enumerate(..., start=1)을 활용하세요.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # enumerate(ranked[:k], 1) : (1, 첫 ID), (2, 둘째 ID) ... 순서로 순위와 ID 를 꺼냄
    for pos, cid in enumerate(ranked[:k], 1):
        if cid in gold:
            return pos
    return None

# ============================================================
# 2. Reciprocal Rank 계산 : calculate_rr() 
# ============================================================
# calculate_rr : Reciprocal Rank = 1 / 첫 관련 문서 순위
# - position 정수 : 1.0 / position (예: 3 -> 0.333..)
# - position None : 0.0 (Gold 는 있으나 Top-k 에서 못 찾음)
# 주의: Gold 자체가 없는 No-answer 는 이 함수를 호출하지 않고 coursekit 에서 None 으로 둠
def calculate_rr(position: int | None) -> float:
    # 정답 D28-04-A02: 관련 문서 첫 순위의 역수(1/rank)를 반환하세요. 문서가 없으면 0.0입니다.
    # 절차: Gold가 아예 없는 No-answer 사례는 집계 대상에서 제외합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    return 1.0 / position if position is not None else 0.0
