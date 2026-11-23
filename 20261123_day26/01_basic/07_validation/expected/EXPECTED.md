# 실습 07 · 기대 조건

이 문서는 **고정된 검증 사례의 기준**입니다. 실제 LLM 응답이나 검색 측정값이 아닙니다.

| 사례 | 기대 결과 | 확인 지점 |
|---|---|---|
| normal | accepted | status=answered, source_ids=[201], 원문 대조 필요 |
| insufficient | accepted | status=insufficient_context, source_ids=[] |
| unknown_source | rejected / answer_validation_error | 999는 포함 ID {201}에 없음 |

출력 최상위 `source_mode`는 `validation_fixtures`, `included_ids`는 `[201]`입니다. 각 사례에는 `case, outcome, error, result, human_review_required`가 있습니다. 통과 시 error는 null이고 result는 네 필드의 dict, human_review_required는 true입니다. 거절 시 result는 null이고 error는 위 오류 코드, human_review_required는 false입니다. `execution`에서 외부 DB·Embedding·LLM 실행을 성공으로 기록하지 않아야 합니다.

자동 테스트는 함수 계약과 의도된 반례를 검사합니다. 사람의 근거 대조를 대신 채점하지 않습니다.
