"""
Day28 실습 02 · 02_intermediate · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 1. Gold ID 선택 : select_gold() 
# ============================================================
# select_gold : Gold Label 파일에서 질문의 관련 청크 ID 목록을 읽어 중복을 없앤 list 로 반환
# - question_id : 'Q01' 같은 질문 ID
# - gold        : data/gold_labels.json 의 gold dict (질문 ID -> 청크 ID 목록)
# - 반환        : 처음 나온 순서를 유지한 중복 없는 ID list
# 예: gold['Q07'] = ['FAC-104-C1', 'FAC-102-C1'] -> ['FAC-104-C1', 'FAC-102-C1']
def select_gold(question_id: str, gold: dict) -> list[str]:
    # 정답 D28-02-I01: gold[question_id]에서 관련 청크 ID 목록을 읽어 중복 없는 list로 반환하세요. 모델 검색 결과를 입력으로 쓰지 마세요.
    # 절차: Gold는 Retriever의 정답이 아니라 평가용 레이블입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 주의: Retriever 검색 결과가 아닌 사람이 만든 평가 레이블만 입력으로 사용
    ids = list(gold[question_id])
    # dict.fromkeys(): 순서를 유지하면서 중복 키 제거 (set 은 순서를 보장하지 않음)
    return list(dict.fromkeys(ids))

# ============================================================
# 2. Gold ID 검증 : validate_gold_ids() 
# ============================================================
# validate_gold_ids : Gold ID 목록이 중복 없이 모두 Corpus 에 실제 존재하는지 검사
# - ids    : select_gold() 결과
# - corpus : 청크 ID -> 청크 dict (data/corpus.json)
# - 반환   : 두 조건을 모두 만족하면 True, 하나라도 어기면 False
# 예: [] (No-answer) -> 중복 없음·존재 검사 대상 없음 -> True
def validate_gold_ids(ids: list[str], corpus: dict) -> bool:
    # 정답 D28-02-I02: ID 중복이 없고 모든 ID가 Corpus에 실제 존재하는지 검사하세요.
    # 절차: set의 길이와 corpus.keys()를 활용하세요.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # len(ids) == len(set(ids))       : 중복 ID 없음
    # all(cid in corpus for cid in ids): 모든 ID 가 Corpus 키에 존재
    return len(ids) == len(set(ids)) and all(cid in corpus for cid in ids)
