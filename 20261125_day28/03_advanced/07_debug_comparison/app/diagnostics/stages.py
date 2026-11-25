"""
Day28 실습 07 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

4개 변형(Baseline·Rewrite Only·Rerank Only·Both) 비교표 함수
"""

# ============================================================
# 4개 변형 비교표 : compare_four_variants() 
# ============================================================
# compare_four_variants : 변형별 실행 결과를 같은 Gold·k=3 으로 평가한 비교표 dict 를 만듦
# - runs   : 변형 이름 -> 실행 결과 (offline 은 baseline·rewrite_only·rerank_only·both 4개)
# - 반환   : 변형 이름 -> evaluate_retrieval() 지표 dict
# 예: Q01 rr_at_k baseline 0.333 / rewrite_only 0.5 / rerank_only 1.0 / both 1.0
def compare_four_variants(runs: dict, gold: list[str], grades: dict) -> dict:
    # 정답 D28-07-A03: Baseline, Rewrite Only, Rerank Only, Both를 같은 Gold와 k로 평가한 딕셔너리를 만드세요.
    # 절차: 한 번에 하나의 기능을 바꾼 비교군을 남기는 이유를 해설하세요.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_metrics.py : Recall·Precision·Hit·RR·DCG·nDCG 를 한 번에 계산
    from coursekit.eval_metrics import evaluate_retrieval
    # 주의: 모든 변형에 같은 gold·grades·k 를 써야 변화량을 기능 차이로 해석할 수 있음
    return {name: evaluate_retrieval(item['ranked_ids'], gold, grades, k = 3)
            for name, item in runs.items()}
