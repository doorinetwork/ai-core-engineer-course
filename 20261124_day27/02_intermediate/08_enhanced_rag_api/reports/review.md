# 실습 08 Enhanced RAG API

> **보고서 ID:** D27-I08-EHC-RAG  
> **작성 기준일:** 2026-11-24  

## 개선된 RAG API와 실행·출처·실패 비교표
 
### 요청 조건 비교
 
| 항목 | Q01 | Q06 | INVALID_K | 근거 구분 |
|:--|:--|:--|:--|:--|
| 질문 (`query`) | 회의실 예약을 취소하는 방법은? | 정확한 과제 마감 날짜는? | Q06 과 같은 질문 | Q01·Q06: output `original_question`,<br>INVALID_K 는 코드: `invalid = {**payload, 'candidate_k': 0}` 의 `payload` 가 반복문 마지막 값(Q06) |
| Category | 시설 | 지원 | 지원 | `data/questions.json` |
| History | 없음 | 없음 | 없음 |  `data/questions.json` |
| `use_rewrite` | **True** | **False** | False |  `'use_rewrite': False if case['id'] == 'Q06' else True` |
| `use_rerank` · `use_multiquery` | True · False | True · False | True · False | 코드 |
| `candidate_k` | 5 | 5 | **0** | INVALID_K 는 산출물 `detail.input` 0 과 일치 |
| `context_k` (Top-N) | 2 | 2 | 2 | 코드 |
 
### 실행 비교표 (Stage 흐름)
 
| 항목 | Q01<br> (정상 질문) | Q06<br> (근거 부족) | INVALID_K<br> (입력 오류) |
|:--|:--|:--|:--|
| HTTP 상태 | **200** | **200** | **422** |
| `body.status` | `answered` | `insufficient_context` | 없음 (`detail` 만 있음) |
| `original_question` | 회의실 예약을 취소하는 방법은? | 정확한 과제 마감 날짜는? | - |
| `search_query` | 회의실 예약을 취소하는 방법은? (원 질문과 같음) | 정확한 과제 마감 날짜는? (Rewrite 꺼짐) | - |
| `queries` · `query_count` | 1개 · 1 | 1개 · 1 | - |
| `candidate_ids` (5) | FAC-102 · FAC-107 · FAC-109 · FAC-103 · FAC-101 | ASS-102 · ASS-103 · ASS-104 · ASS-101 · IT-102 | - |
| `ranked_ids` (5) | FAC-102 · **FAC-109 · FAC-107** · FAC-103 · FAC-101 | ASS-102 · ASS-103 · ASS-104 · ASS-101 · IT-102 (순서 변화 없음) | - |
| `included_ids` (Context, 2) | FAC-102 · FAC-109 | ASS-102 · ASS-103 | - |
| Generation 호출 | 호출함 (Context 있음) | 호출함 (Context 있음) | 호출 안 함 |
| `trace` | 있음 | 있음 | 없음 |
 
- 표에서는 `FAC-102-C1` 을 `FAC-102` 처럼 줄여썼다(칸 확보). ID 목록은 모두 `-C1` 청크다.
- Q01 의 순위 변화는 2·3위(FAC-107 ↔ FAC-109) 뿐이다. Q06 은 Rerank 앞뒤 순서가 같다.
- Q06 도 Context 가 비지 않았으므로 `generate()` 의 `if not s['context']` 분기가 아니라 **LLM 호출 + `validate_generation()`** 경로를 지났다. 즉 Q06 의 정보 부족 판단은 Context 가 없어서가 아니라, Context 를 읽은 LLM 이 `insufficient_context` 를 고르고 그 결과가 검증 규칙을 통과한 것이다 (`coursekit/pipeline.py`).
- INVALID_K 는 FastAPI 가 `RagRequest` 로 요청 본문을 검증하는 단계에서 막혀, `answer()` 안의 `chain.invoke()` 가 실행되지 않았다. 그래서 `trace` 가 없다 (`coursekit/api.py`).

### 출처 비교표
 
| 항목 | Q01 | Q06 | INVALID_K |
|:--|:--|:--|:--|
| `result.source_ids` | `[2]` | `[]` | - |
| `sources` 개수 | 1 | 0 (빈 목록) | - |
| `sources[0]` | citation_id: 2<br>chunk_id: FAC-102-C1<br>doc_id: FAC-102 <br>source_version: 2026-11-v2 <br> source_locator: guides/fac-102.md#section-1 <br>category: 시설 | - | - |
| Context 에 넣은 청크 (`included_ids`) | FAC-102 · FAC-109 | ASS-102 · ASS-103 | - |
| Context 에 있었지만 인용하지 않은 청크 | **FAC-109** | ASS-102 · ASS-103 (전부) | - |
| `missing_information` | '' (빈 문자열) | 과제별 정확한 마감 날짜가 제공되지 않았습니다. 과제별 안내를 확인해야 합니다. | - |
| `answer` | 예약 시스템의 ‘나의 예약’ 메뉴에서 취소를 선택하면 됩니다. 예약 시작 2시간 전까지 취소할 수 있습니다. | 제공된 지식으로는 답할 수 없습니다. | - |
 

### 실패 비교표
 
| 실패 종류 (교안 S78·S81) | 이번 live 행 | HTTP | 응답 내용 | 처리 위치 |
|:--|:--|:--|:--|:--|
| 근거 부족 | Q06 | 200 | `status` insufficient_context, `source_ids` [], `sources` [] | `validate_generation()` 이 상태 규칙 확인 |
| 입력 오류 | INVALID_K | **422** | `detail[0]`: type `greater_than_equal`, loc `["body", "candidate_k"]`, msg 'Input should be greater than or equal to 1', input 0, ctx `{"ge": 1}` | FastAPI·Pydantic `candidate_k: int = Field(default = 5, ge = 1, le = 20)`, Chain 호출 전 |
| 실행 오류 <br>(Stage 실패) | **행 없음** | (502) | 이번 live 에서는 발생하지 않았고, 강제 오류 요청도 보내지 않았다 | `create_app()` 이 `StageError` 를 502 + `{'stage', 'code'}` 로 바꾼다 (원문 대조(코드)). 동작 확인은 10절 offline 참고값만 있다 |
 

## 조건별 결과와 같은 출처 정책 확인
 
### 확인할 출처 정책
 
| ID | 정책 | 코드 근거 | 
|:--|:--|:--|
| P1 | `sources` 는 **서버가** Context 에 넣은 Document Metadata 로 만든다. 모델이 출처 경로를 만들지 않는다 | `generate()`: `index = source_index(s['included'])`, `sources = [index[cid] for cid in result['source_ids']]` | 
| P2 | 인용 ID(`source_ids`)는 Context 에 넣은 `citation_id` 안에서만 고른다 | `validate_generation(..., {d.metadata['citation_id'] for d in s['included']})` 의 `source_not_in_context`, `generate()` 의 `StageError('generation', 'source_not_in_context')` | 
| P3 | **답변에 실제로 쓴 근거만** `sources` 에 담는다 (Context 전체가 아님) | P1 의 `sources` 는 `included` 전체가 아니라 `result['source_ids']` 로만 만든다 | 
| P4 | `answered` 이면 `source_ids` 가 1개 이상, 답변이 비어 있지 않고, `missing_information` 은 빈 문자열 | `validate_generation()` 의 `invalid_answered` | 
| P5 | `insufficient_context` 이면 `source_ids` [], `answer` 는 고정 문장 `NO_ANSWER`, `missing_information` 은 비어 있지 않음 → `sources` 도 [] | `validate_generation()` 의 `invalid_insufficient_context`, Context 가 비면 `generate()` 가 같은 형태를 직접 만든다 | 
| P6 | 실행·입력 오류는 정보 부족으로 위장하지 않고 HTTP 오류(422·502)로 구분한다 | `create_app()` 의 `HTTPException(status_code = 502, ...)`, Pydantic 422. `check_http_response()` 주석 '정보 부족 응답으로 바꾸지 않음' | 
| P7 | `sources` 항목은 조건과 관계없이 **같은 6개 필드**(citation_id · chunk_id · doc_id · source_version · source_locator · category)다 | `source_index()` 가 이 6개 키만 복사 | 
| P8 | 검색 범위는 서버가 정한 learner 역할·현재 버전·ready 자료로 같다 | `create_app(trusted_role = 'learner')`, `search_db()` WHERE `ready`·`is_current`·`current_source_version`·`public` | 
 
### 조건별 정책 준수표
 
| 정책 | Q01 정상 답변 <br>(Rewrite 켬) | Q06 근거 부족 <br>(Rewrite 끔) | INVALID_K <br>입력 오류 |
|:--|:--|:--|:--|
| HTTP·status | 200 · answered | 200 · insufficient_context | 422 · (status 없음) |
| P1 <br>서버가 sources 생성 | 따름: `sources[0]` 6개 값이 FAC-102-C1 Metadata (corpus 의 doc_id·source_version·source_locator·category 와 같음) | 따름: `source_ids` 가 비어 `sources` 도 빈 목록 | 해당 없음 (Chain 미실행) |
| P2 <br>인용 ⊆ Context | 따름: 인용 FAC-102-C1 ∈ included {FAC-102, FAC-109} | 따름: 빈 집합 (인용 없음) | 해당 없음 |
| P3 <br>쓴 근거만 sources | 따름: Context 2개 중 인용한 1개만 `sources` 에 있다. FAC-109 는 Context 에만 있고 `sources` 에 없다 | 따름: Context 2개가 있어도 인용이 없어 `sources` [] | 해당 없음 |
| P4 <br>answered 규칙 | 따름: `source_ids` [2] (1개), 답변 있음, `missing_information` '' | 해당 없음 | 해당 없음 |
| P5 <br>insufficient 규칙 | 해당 없음 | 따름: `source_ids` [], `answer` = '제공된 지식으로는 답할 수 없습니다.' (NO_ANSWER 와 같음), `missing_information` 있음, `sources` [] | 해당 없음 |
| P6 <br>오류 구분 | 해당 없음 | 해당 없음 (오류가 아니라 정상 처리된 정보 부족) | 따름: 422 + `detail` 만 반환, `result`·`sources` 를 만들지 않음 |
| P7 <br>같은 필드 구조 | 6개 필드 | 빈 목록 (필드 구조 비교 대상 없음) | 해당 없음 |
| P8 <br>같은 검색 범위 | learner·시설·버전 2026-11-v2 (corpus 상 현재 버전) | learner·지원. 후보 5개 모두 corpus 상 public·ready·is_current | 해당 없음 |
| `response_check` | http_ok true · citation_consistent true | http_ok true · citation_consistent true | http_ok **false** · error_detail 기록 |
 