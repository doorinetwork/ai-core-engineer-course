# 실습 05 · 원문 대조 기록

API Key나 비밀번호는 적지 않습니다. 실행 결과와 원문을 보고 빈칸을 채웁니다.

| 확인 항목 | 나의 기록 |
|---|---|
| 실행 명령 / 실행 종류 | `python main.py` / live (openai 호출) |
| 실제 모델 호출 여부와 확인 위치 | 호출 함. `real_model_calls`: 1, `execution.llm` = true, `fixed_example_calls` = 0. 실제 `python main.py`로 실행 |
| 질문 / 사용한 원문 위치 | '15분 늦으면 어떻게 기록되나요?' / 201 · DOC-ATTEND · attendance_v2.md#sec-1 |
| 원문이 지지하는 답변 주장 | 고정 응답 "수업 시작 후 15분까지는 지각으로 기록합니다. 사전 승인을 받은 경우에는 운영실에 확인합니다." 는 원문 1문장·3문장과 일치한다 |
| 원문에서 확인할 수 없는 정보 | 없음. |
| Prompt 입력과 AIMessage의 차이 | Prompt 입력은 system(RULES)·human('질문:\n…\n\n자료:\n…') 메시지 2개의 ChatPromptValue,<br> AIMessage는 JSON 문자열 text와 response_metadata(status 'completed', source_mode 'fixed_fixture')를 가진 응답 |

## 아직 확인하지 못한 것

- `--offline`으로 실행해서 고정 예시와 비교
