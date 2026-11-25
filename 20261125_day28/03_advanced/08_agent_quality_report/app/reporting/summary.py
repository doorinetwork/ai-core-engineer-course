"""
Day28 실습 08 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Markdown 품질 보고서 작성 함수
"""

# ============================================================
# 1. Markdown 보고서 작성 : write_summary_report() 
# ============================================================
# write_summary_report : 집계 결과를 Markdown 보고서 문자열로 만듦 (main.py 가 outputs/report.md 로 저장)
# - 출처 실패 또는 상태 실패가 1건 이상 : 품질 판단 '점검 필요'
# - 그 외                                : '사람 검토 후 추가 판단'
def write_summary_report(summary: dict, rows: list[dict]) -> str:
    # 정답 D28-08-A03: 실행 모드, Recall/MRR, No-answer, 출처 실패, 개선 판단이 들어 있는 Markdown 보고서 문자열을 만드세요.
    # 절차: 개별 오류와 실제 근거 검토를 누락한 “평균 점수”만 제출하지 않습니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    review = '점검 필요' if summary['source_integrity_failures'] or summary['status_failures'] else '사람 검토 후 추가 판단'
    # 보고서 항목
    # - Run mode / Variant : 실행 모드·실험 변형 (서로 다른 실행을 섞지 않았음을 표시)
    # - Mean Recall@3, MRR@3 : 유효 Gold 질문 평균
    # - No-answer excluded : 평균에서 뺀 질문 수
    # - Source failures    : 출처 검증 실패 수
    # - Quality decision   : 위 review
    return ('# Day28 RAG Evaluation Report\n\n'
            f"- Run mode: {summary['mode']}\n"
            f"- Variant: {summary['variant']}\n"
            f"- Mean Recall@3: {summary['mean_recall_at_k']}\n"
            f"- MRR@3: {summary['mrr_at_k']}\n"
            f"- No-answer excluded: {summary['no_answer_excluded']}\n"
            f"- Source failures: {summary['source_integrity_failures']}\n"
            f"- Quality decision: {review}\n"
            '- 주의: 교육용 Fixture 점수는 모델 품질 실측값이 아닙니다.\n')
