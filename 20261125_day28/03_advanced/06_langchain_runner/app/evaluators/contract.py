"""
Day28 실습 06 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# Target 실행 평가 : evaluate_target_run() 
# ============================================================
# evaluate_target_run : Target 실행 결과와 Gold 를 Target 밖(Evaluator)에서 결합해 평가 레코드를 만듦
# - example : 원래 평가 예제 (answerable 포함)
# - run     : invoke_target() 결과
# - gold    : 질문의 Gold ID 목록
# - grades  : 질문의 문서별 관련성 등급
# - 반환    : Retrieval 지표·출처 유효성·No-answer 판정이 담긴 dict
def evaluate_target_run(example: dict, run: dict, gold: list[str], grades: dict) -> dict:
    # 정답 D28-06-A03: Target 출력과 Gold를 Target 외부에서 결합하여 Retrieval 점수·출처 상태를 계산하세요.
    # 절차: Gold는 Target 입력에 넣지 않고 Evaluator에만 전달합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 질문 ID·모드·Trace 필드 검사 후 지표·출처 검사
    from coursekit.eval_report import evaluate_record
    # evaluate_record() 인자
    # example, run, gold, grades : 위 입력 그대로
    # k=3                        : Recall@3·MRR@3 등 Top-3 기준
    return evaluate_record(example, run, gold, grades, k = 3)
