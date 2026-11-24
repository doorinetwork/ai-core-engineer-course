
## 실행 추적표 (live) — 질문부터 included_ids·답변·audit 까지
 
### 단계별 추적표
 
| 순서 | Stage (Runnable) | 입력 | 출력 (산출물 값) | 산출물 필드 | 처리 시간 ms (단일 실행 참고값) |
|:--|:--|:--|:--|:--|:--|
| 0 | start | `prepare_pipeline_request()` 가 만든 요청 dict | `query` = '그럼 취소는 어떻게 해?', `search_query` 초기값 = `query`, `trusted_role` = `learner` | `original_question` | 기록 없음 (`run()` 으로 감싸지 않음) |
| 1 | rewrite | `query` + `history` | `search_query` = '회의실 예약은 어떻게 취소하나요?' | `original_question`·`search_query` | 2288.214 |
| 2 | multiquery | `search_query` | `queries` = ['회의실 예약은 어떻게 취소하나요?'] (추가 질의 0개) | `queries`·`query_count` = 1 | 0.006 |
| 3 | retrieval | `queries` 1개 · Category 시설 · k 5 · role learner | 후보 5개: FAC-102 → FAC-107 → FAC-103 → FAC-109 → FAC-108 | `candidate_ids` | 410.148 |
| 4 | rerank | `search_query` + 후보 5개 (top_n = 5) | 재정렬 5개: FAC-102 → FAC-109 → FAC-107 → FAC-103 → FAC-108 | `ranked_ids` | 4968.836 |
| 5 | context | `ranked` 5개 · context_k 2 · max_chars 650 | 포함 2개: FAC-102 → FAC-109 | `included_ids` | 0.014 |
| 6 | generation | 원 질문 + 검색용 질문 + Context 2블록 | `answer` 2문장, `source_ids` = [2] → `sources` FAC-102-C1 1건 | `result`·`sources` | 1389.042 |
| 7 | validate_trace (Chain 밖) | `chain.invoke()` 결과 dict | 3개 항목 모두 true | `trace_check` | 기록 없음 |
| 8 | audit_pipeline (Chain 밖) | 같은 결과 dict | mode live · Stage 이름 6개 · 출처 ID [2] · manual_grounding_review true | `audit` | 기록 없음 |
 
- ID 목록은 모두 `-C1` 청크다. 표에서는 `FAC-102-C1` 을 `FAC-102` 처럼 줄여 쓴다.
- `validate_trace()`·`audit_pipeline()` 은 Chain 의 Stage 가 아니다. `run_06()` 이 `chain.invoke(req)` 가 끝난 뒤 `more[0](result)`·`more[1](result)` 로 차례로 호출해 `row['trace_check']`·`row['audit']` 에 넣는다. 두 함수 모두 Chain 실행 결과를 **읽기만** 하고 결과를 바꾸지 않는다 (원문 대조(코드) `workshops.py`).

### 청크별 추적표 (후보 → 재정렬 → Context → 인용)
 
| chunk_id | 제목 (원문 대조) | candidate 순위 | ranked 순위 | 순위 변화 | Context 포함 | answer 인용 | 취소 방법 답 여부 (3절) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| FAC-102-C1 | 회의실 예약 취소 | 1 | 1 | 유지 | 포함 (1번째 블록) | 인용 (`source_ids` 2) | 답한다 (Gold) |
| FAC-107-C1 | 회의실 키 수령 | 2 | 3 | ▼1 | 제외 (context_k 2 초과) | - | 답하지 않는다 |
| FAC-103-C1 | 회의실 예약 변경 | 3 | 4 | ▼1 | 제외 | - | 부분 관련 |
| FAC-109-C1 | 회의실 청소 안내 | 4 | 2 | ▲2 | 포함 (2번째 블록) | 인용 안 함 | 답하지 않는다 |
| FAC-108-C1 | 회의실 야간 사용 | 5 | 5 | 유지 | 제외 | - | 답하지 않는다 |
 
- 후보 5개와 재정렬 5개는 같은 ID 집합이다. Rerank 는 문서를 추가하거나 빼지 않고 순서만 바꿨다 (`rerank()` 가 `rank_fn(search_query, candidates, min(len(candidates), 20))` = top_n 5 로 호출, 원문 대조(코드)).
- 5개 모두 `source_version` 2026-11-v2 · `source_locator` guides/fac-1xx.md#section-1 · Category 시설 · public 문서다 (원문 대조). 산출물에는 인용된 FAC-102 의 출처만 기록되어 있다.