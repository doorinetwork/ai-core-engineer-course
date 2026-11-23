# 실습 01 확인 기록

- 선택한 수준: 01_basic
- 실행일: 2026-11-23
- 실행 명령: `python main.py`
- 결과 파일: outputs/rag_flow.json
- 입력 출처와 실행한 구성요소:
`data/lecture_hits.json` Hit 201 (출석) 1건
`mode` concept_review
`db` false
`embedding` false
`llm` false

## 1. Context와 Prompt는 어떻게 다른가요?

`"Context Builder": {"input": "후보 목록", "output": "선택한 근거·포함 ID"}`  
`"Prompt": {"input": "규칙·질문·Context", "output": "모델이 읽을 메시지"}`  
Context는 검색 후보 중 이번 답변에 넣기로 결정한 근거이고, Prompt에서 context와 질문, 규칙을 합해 모델이 읽을 메시지로 만든다.

## 2. 15분과 15분 초과가 다른 이유를 원문에서 찾으세요.  

`"evidence": "수업 시작 후 15분까지 입실하면 지각으로 기록합니다. 15분을 초과하면 결석으로 기록합니다. 단, 사전 승인을 받은 경우에는 운영실에 확인합니다."`  
evidence 원문에 따라 15분을 초과하면 결석으로 기록해야 한다.

## 3. 이 JSON은 실제 LLM의 응답인가요?

`"llm": false`  
실제 LLM의 응답이 아니다.

