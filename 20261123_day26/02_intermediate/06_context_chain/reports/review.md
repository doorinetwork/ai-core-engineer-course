# 실습 06 · 원문 대조 기록

API Key나 비밀번호는 적지 않습니다. 실행 결과와 원문을 보고 빈칸을 채웁니다.

| 확인 항목 | 나의 기록 |
|---|---|
| 실행 명령 / 실행 종류 | `python main.py` / live (실제 openai) |
| 실제 모델 호출 여부와 확인 위치 | 호출 함. `real_model_calls`: 1, `execution.llm` = true. |
| 질문 / 사용한 원문 위치 | '15분 늦으면 어떻게 기록되나요?' / 학사 문서 201 attendance_v2.md#sec-1, 202 assignment.txt |
| 원문이 지지하는 답변 주장 | parsed.answer "수업 시작 후 15분까지는 지각으로 기록합니다. 사전 승인을 받은 경우에는 운영실에 확인합니다." · source_ids [201] -> 201 원문과 일치 |
| 원문에서 확인할 수 없는 정보 | 202(과제) 블록은 Context에 포함됐지만 출석 질문과 관계없어 인용되지 않았다 |
| 검색 후보 ID와 실제 포함 ID의 차이 | 학사 후보 [201, 202] 모두 600자 안에 들어가 포함 ID도 [201, 202]. 답변 인용 ID는 [201] |
| raw·parsed·parsing_error 각각의 의미 | `raw`: 모델의 원 AIMessage(status 'completed'), `parsed`: WireAnswer로 파싱된 값, `parsing_error`: 파싱 실패 예외(이번 None) |
| 중복 추가 / max_items 변경 전후 포함 ID | 기준 [201, 202] / 첫 청크 중복 추가 [201, 202] (chunk_id 중복 제외) / max_items 1 -> [201]. LLM 호출 0 |


## Context·포함 ID·Chain 중간값·bundle 세 필드 확인표
 
| 구분 | 필드 | live 값 | 확인 내용 |
|---|---|---|---|
| 실행 종류 | execution · real_model_calls | {"mode": "live", "db": false, "embedding": false, "llm": true} · 1 | live 는 llm true 로 실제 모델 1회 호출, DB·Embedding 은 쓰지 않음 |
| Context | context | 블록 2개 ([201] attendance_v2.md#sec-1 / [202] assignment.txt), 233자 | max_chars 600 안에 두 블록 전체가 들어감. 블록 형식 '[citation_id] source_locator\n본문' |
| 포함 ID | included_ids | [201, 202] | 실제로 Context 에 넣은 문서의 ID |
| 포함 ID | included_sources | 201 DOC-ATTEND v2 attendance_v2.md#sec-1 (demo-attend-v2-sec1); 202 DOC-ASSIGN v1 assignment.txt (demo-assign-v1-sec1) | 출처 정보는 생성 Chain 밖에서 따로 보관  |
| Chain 중간값 | prompt_input | Key ['question', 'context'] · question '15분 늦으면 어떻게 기록되나요?' · context = 위 context 와 같음 | prompt_input → Prompt → Structured Model → bundle 순서로 전달  |
| bundle 세 필드 | raw | AIMessage · status 'completed' · model 'gpt-6-luna' · finish 'completed' · usage 422 tokens | 모델이 돌려준 원래 응답 (raw.text 는 4필드 JSON 문자열) |
| bundle 세 필드 | parsed | status 'answered' · answer "수업 시작 후 15분까지 입실하면 지각으로 기록됩니다. 단, 사전 승인을 받은 경우에는 운영실에 확인해야 합니다." · source_ids [201] · missing_information "" | raw.text 를 파싱한 값과 같음. parsed 가 있어도 승인 완료가 아님 |
| bundle 세 필드 | parsing_error | null | 파싱 실패 없음 |
 
## 아직 확인하지 못한 것
 
- 승인 검사(자료형·상태·출처: check_raw_message → parsing_error → parsed → 로컬 규칙)는 실습 07 에서 한다. 
- 사람의 근거 검토는 `human_evidence_review` = 'pending' 그대로다. 

