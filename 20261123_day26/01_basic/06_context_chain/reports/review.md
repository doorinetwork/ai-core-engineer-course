# 실습 06 · 원문 대조 기록

API Key나 비밀번호는 적지 않습니다. 실행 결과와 원문을 보고 빈칸을 채웁니다.

| 확인 항목 | 나의 기록 |
|---|---|
| 실행 명령 / 실행 종류 |  |
| 실제 모델 호출 여부와 확인 위치 |  |
| 질문 / 사용한 원문 위치 |  |
| 원문이 지지하는 답변 주장 |  |
| 원문에서 확인할 수 없는 정보 |  |
| 검색 후보 ID와 실제 포함 ID의 차이 |  |
| raw·parsed·parsing_error 각각의 의미 |  |

| 확인 항목 | 나의 기록 |
|---|---|
| 실행 명령 / 실행 종류 | `python main.py` / live (실제 opanai 호출) |
| 실제 모델 호출 여부와 확인 위치 | 호출 함. `real_model_calls`: 1, `execution.llm` = true. |
| 질문 / 사용한 원문 위치 | '15분 늦으면 어떻게 기록되나요?' / 학사 문서 201 attendance_v2.md#sec-1, 202 assignment.txt |
| 원문이 지지하는 답변 주장 | parsed.answer "수업 시작 후 15분까지는 지각으로 기록합니다. 사전 승인을 받은 경우에는 운영실에 확인합니다." · source_ids [201] -> 201 원문과 일치 |
| 원문에서 확인할 수 없는 정보 | 202(과제) 블록은 Context에 포함됐지만 출석 질문과 관계없어 인용되지 않았다 |
| 검색 후보 ID와 실제 포함 ID의 차이 | 학사 후보 [201, 202] 모두 600자 안에 들어가 포함 ID도 [201, 202]. 답변 인용 ID는 [201] |
| raw·parsed·parsing_error 각각의 의미 | `raw` = 모델의 원 AIMessage(status 'completed'), `parsed` = WireAnswer로 파싱된 값, `parsing_error` = 파싱 실패 예외(이번 None) |


## 아직 확인하지 못한 것

- `--offline` 모드로 수행해보지 못함
