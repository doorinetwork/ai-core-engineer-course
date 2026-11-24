# 실습 08 Enhanced RAG API

> **보고서 ID:** D27-B08-EHC-RAG  
> **작성 기준일:** 2026-11-24  

## 개선된 RAG API와 실행·출처·실패 비교표

### 실행 비교표

| question_id | 요청 질문 · Category | HTTP 상태 (`http_status`) | 응답 `status` | `result.status` | Chain 실행 여부 | answer |
|:--|:--|:--|:--|:--|:--|:--|
| Q01 | 회의실 예약을 취소하는 방법은? · 시설 | **200** | `answered` | `answered` | 실행<br>(6 Stage 모두 기록) | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택해 취소하세요. <br>시작 2시간 전까지 취소할 수 있습니다. |
| Q06 | 정확한 과제 마감 날짜는? · 지원 | **200** | `insufficient_context` | `insufficient_context` | 실행 (6 Stage 모두 기록) | 제공된 지식으로는 답할 수 없습니다. |
| INVALID_K | Q06 요청 + `candidate_k` 0 | **422** | 없음 (`detail` 만) | 없음 | **실행 안 됨** (요청 검증에서 거부, `trace` 없음) | 없음 |

### 출처 비교표

| question_id | `source_ids` | `sources[].citation_id` | `chunk_id` | `doc_id` | `source_version` | `source_locator` | `category` | included_ids (Context) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Q01 | [2] | 2 | FAC-102-C1 | FAC-102 | 2026-11-v2 | guides/fac-102.md#section-1 | 시설 | FAC-102-C1 · FAC-109-C1 |
| Q06 | [] | - (`sources` 빈 목록) | - | - | - | - | - | ASS-102-C1 · ASS-103-C1 |
| INVALID_K | - | - | - | - | - | - | - | - (Chain 미실행) |

- Q01 은 Context 2개 중 **FAC-102 1개만 인용**했다. FAC-109 는 Context 에 들어갔지만 `source_ids` 에 없다.
- Q06 은 Context 2개가 있었는데도 인용은 되지 않았다. LLM 이 '근거 부족' 을 판단한 결과다.

### 실패 비교표

| 구분 | 이번 live 행 | HTTP | 응답 모양 | 의미 |
|:--|:--|:--|:--|:--|
| 정상 답변 | Q01 | 200 | `status` answered + `sources` 1건 | 근거 문서를 인용한 답변 |
| 근거 부족 (`insufficient_context`) | Q06 | 200 | `answer` = 고정 문구 `NO_ANSWER`, `source_ids` [], `missing_information` 기재 | 실행 오류가 아닌, 검색된 자료로는 답할 수 없다는 정상 결과 |
| 요청 형식 오류 | INVALID_K | **422** | FastAPI 기본 검증 오류 `detail[]` | 사용자 입력 문제. Chain을 호출하기 전에 거부 |
| Stage 실행 오류 | **이번 live에는 없음** | 502 | `detail = {stage, code}` | 서버 쪽 처리 실패. 근거 부족으로 위장하지 않는다 |

## 질문→출처 포함 답변까지 순서 설명

### 단계별 순서표

| 순서 | 단계 | 입력 | 하는 일 | Q01 결과 (산출물) | ms |
|:--|:--|:--|:--|:--|:--|
| 0 | 준비 (요청 전) |  - | `require_cuda()` 확인, E5 로드(`make_live_search()`), Rewrite·Generation Chain 생성, LCEL Chain 조립, FastAPI app 에 연결 | 기록 없음 | 기록 없음 |
| 1 | FastAPI 요청 |  `POST /api/v1/rag/answer` JSON | 경로 함수 `answer(req: RagRequest)` 호출 | HTTP 200 | - |
| 2 | 검증 (Pydantic) |  요청 JSON | `extra = 'forbid'`·`strict = True`, query 1~800자·공백 불가, history ≤ 8, Category 3종, candidate_k 1~20, context_k 1~3 | 통과 (INVALID_K 는 여기서 422) | - |
| 3 | 서버 역할 지정 |  검증된 요청 | `body = req.model_dump()` 후 `body['trusted_role'] = 'learner'` (요청 JSON 에는 role 필드 자체가 없다) | 검색 role = learner | - |
| 4 | Chain 시작 |  body | `start()` 가 `RagRequest` 로 한 번 더 검증하고 `search_query = query`, 빈 `queries`·`candidates` 등 상태 dict 생성, `source_mode = 'live'` | `original_question` = `search_query` 초기값 | 기록 없음 |
| 5 | Rewrite |  query + history | `use_rewrite` True 면 `rewrite_fn()`(LLM Structured Output) 결과로 `search_query` 만 바꾼다. 원 질문 `query` 는 그대로 | `search_query` = '회의실 예약을 취소하는 방법은?' (원 질문과 같음, History 없음) | 2336.687 |
| 6 | Multi-query |  search_query | `use_multiquery` False → 추가 질의 `[]`, `unique_queries()` 가 `[search_query]` 만 반환 | `queries` 1개, `query_count` 1 | 0.006 |
| 7 | Retrieval |  queries · 시설 · k 5 · learner | E5 질의 Embedding → pgvector `WHERE category · ready · is_current · 현재 버전 · (staff OR public)` `ORDER BY cosine_distance, chunk_id LIMIT 5` | 후보 FAC-102 · 107 · 109 · 103 · 101 | 337.737 |
| 8 | Rerank |  search_query + 후보 5 | CrossEncoder 로 (질문, 본문) 5쌍 점수화, `(-rerank_score, pre_rank)` 정렬, top_n = min(5, 20) = 5 | ranked FAC-102 · 109 · 107 · 103 · 101 | 4346.404 |
| 9 | Context |  ranked 5 · context_k 2 · max_chars 650 | 순위대로 `[citation_id] source_locator\n본문` 블록을 붙이고 2개가 되면 멈춘다 | included FAC-102 · FAC-109 | 0.013 |
| 10 | Generation |  원 질문 + '[검색용 독립 질문] ' + search_query, Context 2블록 | LLM 이 `WireAnswer`(status·answer·source_ids·missing_information) JSON Schema 로 답한다 | answer 2문장, source_ids [2] | 1852.036 |
| 11 | Validation |  LLM 출력 + included 의 citation_id 집합 | 파싱 오류·거부·양의 정수·중복·`source_ids ⊆ Context`·상태 규칙 검사, 다시 `source_index` 기준으로 확인 | 통과 | (generation 에 포함) |
| 12 | 응답 (sources 포함) |  `sources = [index[cid] for cid in source_ids]` 로 Metadata 6개 복원, `make_trace()` 로 ID·시간 기록 | `sources` 1건 (citation_id 2 = FAC-102-C1) | - |

### 값의 전달 과정 (Q01)

| 전달 구간 | 넘어가는 값 | Q01 값 |
|:--|:--|:--|
| 요청 → 검증 | JSON 8필드 | query · history [] · category 시설 · candidate_k 5 · context_k 2 · use_rewrite True · use_multiquery False · use_rerank True |
| 검증 → Chain | `body` + `trusted_role` | trusted_role learner |
| Rewrite → Multi-query | `search_query` | 회의실 예약을 취소하는 방법은? |
| Multi-query → Retrieval | `queries` | [회의실 예약을 취소하는 방법은?] |
| Retrieval → Rerank | `candidates` (Document 5개) | 102 · 107 · 109 · 103 · 101 |
| Rerank → Context | `ranked` (같은 5개, 순서만 변경) | 102 · 109 · 107 · 103 · 101 |
| Context → Generation | `context` 문자열 + `included` | '[2] guides/fac-102.md#section-1 …' + '[FAC-109 의 citation_id] guides/fac-109.md#section-1 …' |
| Generation → 응답 | `result` + `sources` | source_ids [2] → FAC-102-C1 출처 1건 |

- 표에서 `FAC-102-C1` 은 `FAC-102` 처럼 줄였다. (가독성 확보)

### 같은 과정에서 Q06이 다른 점

| 단계 | Q06 결과 | 설명 |
|:--|:--|:--|
| Rewrite | `search_query` = 원 질문 그대로, 0.001 ms |  |
| Retrieval | 후보 ASS-102 · ASS-103 · ASS-104 · ASS-101 · IT-102 (Category 지원) | 지원 Category 7개 청크 중 5개. IT-101·IT-103 은 빠졌다 (원문 대조) |
| Rerank | ranked 순서가 후보와 같다 | CrossEncoder 점수 순서가 E5 거리 순서와 같았다는 뜻이다 (점수 미기재) |
| Context | included ASS-102 · ASS-103 | 블록 2개 이은 길이 약 183자 (citation_id 두 자리로 가정해 계산, 원문 대조) < 650 이라 max_chars 는 영향 없음 |
| Generation | `insufficient_context`, source_ids [], missing_information '과제별 안내에 기재된 정확한 마감 날짜가 제공되지 않았습니다.' | |
| 응답 | `sources` 빈 목록 | `source_ids` 가 비어 복원할 출처가 없다 |