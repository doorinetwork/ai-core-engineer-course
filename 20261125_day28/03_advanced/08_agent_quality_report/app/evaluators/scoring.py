"""
Day28 실습 08 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

질문별 평가·전체 집계 함수
"""

# ============================================================
# 1. 질문별 평가 레코드 : evaluate_one() 
# ============================================================
# evaluate_one : 질문 1개의 Target 출력과 Gold 를 질문 ID 로 결합해 평가 레코드를 반환
# - Trace 필수 필드(ranked_ids·candidate_ids·included_ids·answer·status·sources)가 빠지면 ValueError
def evaluate_one(example: dict, run: dict, gold: list[str], grades: dict) -> dict:
    # 정답 D28-08-A01: 질문 ID로 Target 출력과 Gold를 결합해 평가 레코드를 반환하세요.
    # 절차: Trace 필드가 빠졌다면 빈 리스트로 바꾸지 말고 오류로 보고합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 질문 단위 평가 레코드 생성
    from coursekit.eval_report import evaluate_record
    # k=3 : Top-3 기준 지표
    return evaluate_record(example, run, gold, grades, k = 3)

# ============================================================
# 2. 전체 집계 : summarize_all() 
# ============================================================
# summarize_all : 질문별 평가 레코드를 집계 (유효 Gold 질문만 평균, No-answer·오류·출처 실패는 개수로)
# - rows 비어 있음 또는 모드·버전·변형이 섞임 : ValueError
# - 반환 예: {'mean_recall_at_k': 1.0, 'mrr_at_k': 1.0, 'no_answer_excluded': 1, 'source_integrity_failures': 0, ...}
def summarize_all(rows: list[dict]) -> dict:
    # 정답 D28-08-A02: 개별 질문 결과의 평균(유효 Gold만)과 오류·출처 상태를 집계하세요.
    # 절차: No-answer 질문은 Retrieval 평균에서 제외하고 별도 개수로 기록합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 평균·개수 집계
    from coursekit.eval_report import report_summary
    return report_summary(rows)
