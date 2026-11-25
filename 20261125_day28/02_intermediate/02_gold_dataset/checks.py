"""
Day28 실습 02 · 02_intermediate · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# No-answer 표시 : label_no_answer() 
# ============================================================
# label_no_answer : Gold 가 비어 있는 질문을 No-answer 로 표시
# - 반환 True  : Gold 0개 (예: Q06 '정확한 과제 마감 날짜는?')
# - 반환 False : Gold 1개 이상
def label_no_answer(question_id: str, gold: dict) -> bool:
    # 정답 D28-02-I03: Gold가 비어 있는 질문은 No-answer로 분리하는 bool을 반환하세요.
    # 절차: Gold가 없는 질문의 Recall을 0점으로 평균내면 안 됩니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 주의: No-answer 질문은 Recall 0 으로 평균에 넣지 않고 별도로 집계
    return len(gold[question_id]) == 0
