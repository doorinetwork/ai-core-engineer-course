"""
Day28 실습 04 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 1. nDCG@k 계산 : calculate_ndcg() 
# ============================================================
# calculate_ndcg : nDCG@k = DCG@k / IDCG@k (IDCG = 등급이 높은 순으로 정렬한 이상적 순서의 DCG)
# - grades 가 비어 있음      : None
# - Top-k 에 미판정 문서     : ValueError
# - 이상적 DCG 가 0 (관련 문서 없음) : None
# 예: Q01 DCG 3.5, IDCG 7.0 -> 0.5
def calculate_ndcg(ranked: list[str], grades: dict[str,int], k: int) -> float | None:
    # 정답 D28-04-A04: Gold 등급에서 이상적인 순서를 만들어 IDCG를 구한 다음 DCG/IDCG를 계산하세요. Gold가 비었으면 None.
    # 절차: 등급이 높을수록 앞순위에 오도록 정렬합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    from math import log2
    if not grades:
        return None
    if any(cid not in grades for cid in ranked[:k]):
        raise ValueError("미판정 후보의 등급을 먼저 확인하세요")
    # 이상적 순서: 등급 내림차순(-grades[cid]), 같은 등급은 ID 오름차순으로 고정
    ideal = sorted(grades, key = lambda cid: (-grades[cid], cid))

    # --------------------------------------------------------
    # 내부 함수 dcg : 주어진 순서의 DCG@k
    # --------------------------------------------------------
    # i 는 0부터 시작하므로 log2(i + 2) = log2(순위 + 1)
    def dcg(items):
        return sum((2 ** grades[cid] - 1) / log2(i + 2) for i, cid in enumerate(items[:k]))
    optimal = dcg(ideal)
    return dcg(ranked) / optimal if optimal > 0 else None
