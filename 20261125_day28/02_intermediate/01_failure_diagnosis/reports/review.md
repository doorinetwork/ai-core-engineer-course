### Failure Stage·증거·개선 방향 표

| case_id | `Failure Stage` | 증거 | 설명 | 개선 방향 |
|:--|:--|:--|:--|:--|
| F01 | candidate_miss | Gold 문서 ID가 최초 검색 후보에 존재하지 않음 | 정답 문서가 후보 밖에 있음 | Query·Metadata Filter·Indexing·Candidate-k 점검 |
| F02 | ranking_or_context_miss | Gold는 후보에 있으나 LLM Context에 포함되지 않음 |  후보에는 있으나 Context Top-N에서 제외 | Reranker·Context Top-N·길이 예산 점검 |
| F03 | generation_failure | 근거가 Context에 있는데 답변 주장이 지지되지 않음 | 근거는 있지만 답변에 없는 내용을 주장 | Prompt·근거 주장·출처 검증 점검 |
| F04 | system_failure | 서비스 실행이 실패해 지표를 계산할 수 없음 | API 인증 오류 / GPU Out of Memory 예시 | GPU/DB/API 연결·Timeout을 먼저 진단 |
| F05 | passed | 검색 후보·Context·답변 검토 조건 충족 | 요청 처리 정상 사례 | 검증 결과 기록 후 유지 |