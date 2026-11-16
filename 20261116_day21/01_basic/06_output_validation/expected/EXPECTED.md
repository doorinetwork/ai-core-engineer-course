# 예상 결과와 한계

validation_fixtures.json의 각 expected_error와 자신의 분류가 같아야 합니다. valid_answer·valid_insufficient는 자동 통과, broken_json은 json_error, 문자열/boolean ID·추가 필드는 schema_error. semantic_counterexample·over_abstain은 자동 통과하므로 사람 검토가 필요합니다.

이 문서는 기준이지 실제 모델 실행 로그가 아닙니다.
