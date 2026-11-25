"""
Day28 실습 07 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Before/After 점수 변화량과 문제 단계 진단 함수
"""


# ============================================================
# 1. Before/After 변화량 : score_delta()
# ============================================================
# score_delta : 같은 질문의 Before -> After 점수 변화량 (After - Before)
# - 둘 중 하나라도 None(No-answer) : None
# - 그 외                          : 소수 6자리 반올림 값
# 예: Q07 baseline 0.5 -> rerank 1.0 -> 0.5
def score_delta(before: float | None, after: float | None) -> float | None:
    # 정답 D28-07-A01: 같은 질문의 Before/After 점수 변화량을 계산하세요. No-answer(None)는 수치 차이를 만들지 않습니다.
    # 절차: A/B 비교 전 두 실행의 Dataset 버전과 모드가 같아야 합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    if before is None or after is None:
        return None
    # round(..., 6) : 부동소수점 오차(예: 0.30000000000000004)를 정리
    return round(after - before, 6)


# ============================================================
# 2. 문제 단계 진단 : find_problem_stage() 
# ============================================================
# find_problem_stage : 실행 Trace 를 보고 Gold 가 어느 단계에서 사라졌는지 반환
# - 반환값 : 'system_failure'·'no_answer_case'·'candidate_miss'·'ranking_miss'·'context_miss'
#            ·'generation_or_validation'·'inspect_answer_grounding'
# 예: Q01 baseline (Gold 가 3위, 상위 2개에 없음) -> 'ranking_miss'
def find_problem_stage(run: dict, gold: list[str]) -> str:
    # 정답 D28-07-A02: 검색 후보 누락·Reranking 누락·Context 누락·생성 실패 중 어디인지 반환하세요.
    # 절차: Candidate에 Gold가 없다면 먼저 Retrieval 단계 문제입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_experiments.py : 후보 -> 상위 2개 -> Context -> 상태 순서로 진단
    from coursekit.eval_experiments import diagnose
    return diagnose(run, gold)
