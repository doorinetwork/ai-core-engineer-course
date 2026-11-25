"""
Day28 실습 03 · 02_intermediate · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 1. Precision@k·Hit@k 계산 : precision_and_hit() 
# ============================================================
# precision_and_hit : Precision@k 와 Hit@k 를 함께 계산
# - precision_at_k : Top-k 안의 Gold 개수 / k (결과가 k개보다 적어도 분모는 k)
# - hit_at_k       : Top-k 안에 Gold 가 하나라도 있으면 1, 없으면 0
# - gold 가 비어 있음 : 두 값 모두 None (No-answer)
# 예: Q01 hits 1, k=3 -> {'precision_at_k': 0.333.., 'hit_at_k': 1}
def precision_and_hit(ranked: list[str], gold: list[str], k: int) -> dict:
    # 정답 D28-03-I03: Precision@k는 Gold 포함 수/k, Hit@k는 Gold 포함 여부를 0 또는 1로 계산하세요. No-answer는 None.
    # 절차: k보다 결과가 적더라도 이 교안에서는 Precision 분모를 k로 정의합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    if not gold:
        return {'precision_at_k': None, 'hit_at_k': None}
    hits = len(set(ranked[:k]) & set(gold))
    # int(hits > 0) : True/False -> 1/0
    return {'precision_at_k': hits / k, 'hit_at_k': int(hits > 0)}
