"""
Day28 실습 03 · 02_intermediate · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 1. Recall@k 계산 : calculate_recall() 
# ============================================================
# calculate_recall : Recall@k = Top-k 안의 Gold 개수 / 전체 Gold 개수
# - k 가 정수가 아니거나 1 미만 : ValueError
# - gold 가 비어 있음           : None (No-answer, 0점과 다름)
# - 그 외                       : 0.0 ~ 1.0
# 예: Q07 Gold 2개 중 Top-3 에 1개 -> 1 / 2 = 0.5
def calculate_recall(ranked: list[str], gold: list[str], k: int) -> float | None:
    # 정답 D28-03-I01: Top-k에서 찾은 Gold 개수 / 전체 Gold 개수를 계산하세요. Gold가 빈 질문은 None을 반환하세요.
    # 절차: 예: Gold 2개 중 Top-3에 1개면 Recall@3=0.5입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # type(k) is not int : True·False(bool) 도 int 의 하위형이므로 isinstance 대신 type() 으로 엄격히 검사
    if type(k) is not int or k < 1:
        raise ValueError('k는 1 이상의 정수')
    # No-answer는 0점이 아님 : Gold 가 없으면 Recall 을 정의할 수 없어 None 반환
    if not gold:
        return None
    # ranked[:k]       : 상위 k개 검색 결과
    # set(...) & set() : 중복을 한 번만 세는 교집합
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


# ============================================================
# 2. Gold 적중 수 : count_gold_hits() 
# ============================================================
# count_gold_hits : Top-k 검색 결과와 Gold 의 교집합 크기 (Recall·Precision 의 분자)
# - 중복 ID 는 set 으로 한 번만 셈
# 예: Q01 ranked ['FAC-101-C1', 'FAC-107-C1', 'FAC-102-C1', ...], Gold ['FAC-102-C1'], k=3 -> 1
def count_gold_hits(ranked: list[str], gold: list[str], k: int) -> int:
    # 정답 D28-03-I02: Top-k 목록과 Gold의 교집합 크기를 반환하세요. 중복 ID는 한 번만 셉니다.
    # 절차: set 교집합(&)을 사용할 수 있습니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    return len(set(ranked[:k]) & set(gold))
