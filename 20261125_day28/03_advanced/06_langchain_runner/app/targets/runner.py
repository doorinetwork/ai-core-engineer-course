"""
Day28 실습 06 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""


# ============================================================
# 1. Target 입력 분리 : safe_target_input() 
# ============================================================
# safe_target_input : 평가 예제에서 Target(RAG) 입력 4개 필드만 새 dict 로 복사
# - 포함 : question_id, question, category, history
# - 제외 : answerable, reference_answer 등 정답 정보
# 예: Q01 예제 -> {'question_id': 'Q01', 'question': '회의실 예약을 취소하는 방법은?', 'category': '시설', 'history': []}
def safe_target_input(example: dict) -> dict:
    # 정답 D28-06-A01: question_id·question·category·history만 새 dict로 복사하고 Gold·Reference를 제외하세요.
    # 절차: Target Runner는 정답을 모르는 상태에서 실행해야 공정합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # Gold Label 및 Reference Answer는 모델 입력으로 전달하지 않습니다.
    # 허용 키 목록을 직접 지정해 새 dict 생성 (example 에서 빼는 방식은 새 정답 필드가 생기면 누출됨)
    return {key: example[key] for key in ('question_id', 'question', 'category', 'history')}


# ============================================================
# 2. Runnable.invoke 호출 : invoke_target() 
# ============================================================
# invoke_target : LangChain Runnable(Target)의 invoke() 를 호출하고 출력 계약을 확인
# - target  : coursekit 가 RunnableLambda 로 감싼 RAG 실행 함수
# - request : safe_target_input() 결과
# - Gold 관련 키가 request 에 있으면 ValueError (assert_no_gold_leakage)
# - 결과가 dict 가 아니거나 question_id 가 다르면 ValueError
def invoke_target(target, request: dict) -> dict:
    # 정답 D28-06-A02: 실제 Runnable.invoke(request)를 호출하고 결과가 dict이며 질문 ID가 같은지 확인하세요.
    # 절차: 단순 함수 직접 호출 대신 RunnableLambda를 감싼 객체의 invoke()입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_dataset.py : 금지 키(gold_chunk_ids·reference_answer 등) 포함 여부 검사
    from coursekit.eval_dataset import assert_no_gold_leakage
    assert_no_gold_leakage(request)
    # target.invoke(request) : RunnableLambda 안의 함수 실행 (offline 은 고정 Fixture, live 는 실제 RAG)
    result = target.invoke(request)
    if not isinstance(result, dict) or result.get('question_id') != request['question_id']:
        raise ValueError('Target 출력의 질문 ID 불일치')
    return result
