
## 실행 추적표 (live) — 질문부터 included_ids·답변까지

### 단계별 추적표

| 순서 | Stage (Runnable) | 입력 | 출력 (산출물 값) | 산출물 필드 | 처리 시간 ms (단일 실행 참고값) |
|:--|:--|:--|:--|:--|:--|
| 0 | start | `prepare_pipeline_request()` 가 만든 요청 dict | `query` = '그럼 취소는 어떻게 해?', `search_query` 초기값 = `query`, `trusted_role` = `learner` | `original_question` | 기록 없음 (`run()` 으로 감싸지 않음) |
| 1 | rewrite | `query` + `history` | `search_query` = '회의실 예약을 취소하는 방법은 무엇인가요?' (`status` ready) | `original_question`·`search_query` | 1842.801 |
| 2 | multiquery | `search_query` | `queries` = ['회의실 예약을 취소하는 방법은 무엇인가요?'] (추가 질의 0개) | `queries`·`query_count` = 1 | 0.003 |
| 3 | retrieval | `queries` 1개 · Category 시설 · k 5 · role learner | 후보 5개: FAC-102 → FAC-107 → FAC-103 → FAC-109 → FAC-101 | `candidate_ids` | 396.668 |
| 4 | rerank | `search_query` + 후보 5개 (top_n = 5) | 재정렬 5개: FAC-102 → FAC-109 → FAC-107 → FAC-103 → FAC-101 | `ranked_ids` | 4577.927 |
| 5 | context | `ranked` 5개 · context_k 2 · max_chars 650 | 포함 2개: FAC-102 → FAC-109 | `included_ids` | 0.018 |
| 6 | generation | 원 질문 + 검색용 질문 + Context 2블록 | `answer` 2문장, `source_ids` = [2] → `sources` FAC-102-C1 1건 | `result`·`sources` | 1386.888 |

- ID 목록은 모두 `-C1` 청크다. 표에서는 `FAC-102-C1` 을 `FAC-102` 처럼 줄여 쓴다.


### 청크별 추적표 (후보 → 재정렬 → Context → 인용)

| chunk_id | 제목 (원문 대조) | candidate 순위 | ranked 순위 | 순위 변화 | Context 포함 | answer 인용 | 취소 방법 답 여부 (3절) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| FAC-102-C1 | 회의실 예약 취소 | 1 | 1 | 유지 | 포함 (1번째 블록) | 인용 (`source_ids` 2) | 답한다 (Gold) |
| FAC-107-C1 | 회의실 키 수령 | 2 | 3 | ▼1 | 제외 (context_k 2 초과) | - | 답하지 않는다 |
| FAC-103-C1 | 회의실 예약 변경 | 3 | 4 | ▼1 | 제외 | - | 부분 관련 |
| FAC-109-C1 | 회의실 청소 안내 | 4 | 2 | ▲2 | 포함 (2번째 블록) | 인용 안 함 | 답하지 않는다 |
| FAC-101-C1 | 회의실 예약 신청 | 5 | 5 | 유지 | 제외 | - | 답하지 않는다 |
