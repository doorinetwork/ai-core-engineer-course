"""
Day28 실습 01 · 02_intermediate · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""

# ============================================================
# 우선 점검 모듈 추천 : recommend_fix() 
# ============================================================
# recommend_fix : stage 별로 개발팀이 가장 먼저 점검할 모듈을 반환
# - candidate_miss          : Query·Metadata Filter·Indexing·Candidate-k (후보 단계)
# - ranking_or_context_miss : Reranker·Context Top-N·길이 예산 (순위·Context 단계)
# - 없는 stage              : KeyError 발생
def recommend_fix(stage: str) -> str:
    # 정답 D28-01-I03: 각 stage마다 실제 개발팀이 우선 점검할 모듈을 반환하세요.
    # 절차: Candidate 밖의 문서는 Reranker가 되살릴 수 없습니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # stage : 우선 점검 대상
    # - system_failure          : GPU/DB/API 연결·Timeout
    # - candidate_miss          : Query·Metadata Filter·Indexing·Candidate-k
    # - ranking_or_context_miss : Reranker·Context Top-N·길이 예산
    # - generation_failure      : Prompt·근거 주장·출처 검증
    # - passed                  : 검증 결과 기록 후 유지
    return {
        'system_failure': 'GPU/DB/API 연결·Timeout을 먼저 진단',
        'candidate_miss': 'Query·Metadata Filter·Indexing·Candidate-k 점검',
        'ranking_or_context_miss': 'Reranker·Context Top-N·길이 예산 점검',
        'generation_failure': 'Prompt·근거 주장·출처 검증 점검',
        'passed': '검증 결과 기록 후 유지',
    }[stage]
