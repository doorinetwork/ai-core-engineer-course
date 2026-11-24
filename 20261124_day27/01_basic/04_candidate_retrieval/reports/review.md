
## 후보 ID·출처·필터·Context 선택 수의 확인표 (live)
 
### 후보 ID와 출처
 
| 순위 | chunk_id (live 산출물) | doc_id | 제목 | source_locator | source_version | 근거 구분 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | FAC-102-C1 | FAC-102 | 회의실 예약 취소 | guides/fac-102.md#section-1 | 2026-11-v2 | ID 는 산출물, 나머지는 원문 대조 |
| 2 | FAC-107-C1 | FAC-107 | 회의실 키 수령 | guides/fac-107.md#section-1 | 2026-11-v2 | ID 는 산출물, 나머지는 원문 대조 |
| 3 | FAC-109-C1 | FAC-109 | 회의실 청소 안내 | guides/fac-109.md#section-1 | 2026-11-v2 | ID 는 산출물, 나머지는 원문 대조 |
| 4 | FAC-103-C1 | FAC-103 | 회의실 예약 변경 | guides/fac-103.md#section-1 | 2026-11-v2 | ID 는 산출물, 나머지는 원문 대조 |
| 5 | FAC-101-C1 | FAC-101 | 회의실 예약 신청 | guides/fac-101.md#section-1 | 2026-11-v2 | ID 는 산출물, 나머지는 원문 대조 |
 
- 순위는 `candidate_ids` 순서다. `search_db()` 가 `ORDER BY cosine_distance ASC, chunk_id ASC` 로 정렬하므로 1위가 질문 벡터와 가장 가깝다. 거리값은 산출물에 없어 확인할 수 없다.
- `doc_id`·제목·`source_locator`·`source_version` 은 `data/corpus.json` 의 같은 `chunk_id` 항목에서 옮겼다. 실제 DB 행의 값은 산출물에 기록되지 않았으므로 **원문 대조값**이다. DB 가 같은 corpus 로 색인되었다면 같은 값이다.
- `citation_id` 는 실제 DB 가 만든 값이라 corpus 의 `fixture_citation_id`(1001 등)와 다를 수 있다. 산출물에 없어 미기재로 둔다.
- `hit_to_document()` 는 `body`·`citation_id`·`chunk_id`·`doc_id`·`source_version`·`source_locator`·`category` 중 하나라도 없으면 `ValueError` 를 낸다. 실행이 오류 없이 끝나 후보 5개 모두 Document 로 변환되었으므로, 5개 모두 `chunk_id` 와 `source_locator` 를 가진 채 Retriever 를 통과했다 (코드 원문 대조로 판단, 개별 값은 산출물에 없음).

### 검색 필터
 
| 필터 | 이번 요청 값 | 적용 위치 | 후보 5개 확인 (원문 대조) |
|:--|:--|:--|:--|
| category | `시설` (산출물 `category`) | `search_db()` `WHERE d.category = %s`, `validate_args()` 는 학사·시설·지원만 허용 | 5개 모두 `FAC-` 시설 문서 |
| role / 권한 | `learner` (`run_04()` 고정값) | `(%s = 'staff' OR d.access_level = 'public')` | 5개 모두 `access_level` public. staff 전용 FAC-110-C1 없음 |
| status / 공개 준비 | `ready = TRUE` | `c.ready = TRUE` | 5개 모두 ready. 준비중 FAC-112-C1 없음 |
| version / 현재 버전 | `is_current = TRUE`, `source_version = current_source_version` | `c.is_current = TRUE AND c.source_version = d.current_source_version` | 5개 모두 현재 버전. 옛 규정 FAC-111-C1(`is_current` false) 없음 |
| Embedding 계약 | `embedding_revision` = 모델 revision | `c.embedding_revision = %s`, `check_meta()` 가 DB 표식과 E5 계약을 비교 | 실행이 `RuntimeError` 없이 끝나 계약 일치로 판단 (값은 미기재) |
| 후보 수 | `candidate_k` = 5 (산출물) | `LIMIT %s`, `validate_args()` 1~20 | 5개 = candidate_k, 초과 없음 |
 
- 필터 값은 산출물에 없다. 표의 값은 `coursekit/db.py` `search_db()` SQL 과 `data/corpus.json` 메타데이터를 대조한 결과다.
- 제외 확인은 corpus 의 시설 문서 12개 중 필터에 걸리는 3개(FAC-110 staff, FAC-111 옛 버전, FAC-112 준비중)가 후보에 없다는 관찰이다. 후보가 5개뿐이라 거리 순위 밖이어서 빠졌을 가능성과 필터로 빠졌을 가능성을 이 산출물만으로 나눌 수 없다. 필터 자체의 동작은 SQL 원문으로 확인했다.

### Context 선택 수
 
| 항목 | 값 | 근거 구분 |
|:--|:--|:--|
| candidate_k (검색 후보 수) | 5 | 산출물 |
| 실제 후보 수 | 5 | 산출물 `candidate_ids` 길이 |
| context_k (LLM Context 에 넣을 수) | 2 | 산출물 |
| 허용 범위 | candidate_k 1~20 (`validate_args()`, `fixture_search()`), context_k 1~3 (`context_from_documents()`, `api.py` `Field(ge = 1, le = 3)`) | 코드 원문 대조 |
| 이 실습에서 Context 선택 실행 여부 | 실행하지 않음. `run_04()` 는 `context_k` 를 기록만 하고 `context_from_documents()` 를 부르지 않는다 | 코드 원문 대조 |
| context_k = 2 를 이 후보에 적용하면 | FAC-102-C1, FAC-107-C1 (앞에서부터 2개, 650자 이내) |  |


## 3. 원문 대조
 
| chunk_id | 본문 요지 (`data/corpus.json`) | 질문(취소 방법)에 대한 답 여부 |
|:--|:--|:--|
| FAC-102-C1 | 나의 예약 메뉴에서 취소 선택, 시작 2시간 전까지 취소 가능 | 답한다 (Q01 Gold) |
| FAC-107-C1 | 안내 데스크에서 키 수령·반납, 예약 취소 규정은 별도 문서 참조 | 답하지 않는다 (취소 언급만) |
| FAC-109-C1 | 사용 후 전원·좌석 정리, 예약·취소 절차는 시스템 문서 확인 | 답하지 않는다 (취소 언급만) |
| FAC-103-C1 | 날짜·시간 변경은 기존 예약을 취소한 후 새로 신청 | 부분 관련 (변경 절차 안의 취소) |
| FAC-101-C1 | 예약 시스템에서 날짜·시간·목적 입력해 신청 | 답하지 않는다 (신청 절차) |
 