# 실습 03 확인 기록

- 선택한 수준: 01_basic
- 실행일: 2026-11-23
- 실행 명령: `python main.py`
- 결과 파일: outputs/object_contracts.json
- 입력 출처와 실행한 구성요소: 
`data/lecture_hits.json` 구성요소 4개 대응표  
`mode` concept_review  
`db` false  
`embedding` false  
`llm` false

## 1. 같은 invoke인데 Prompt와 Chat Model은 무엇이 다른가요?

```
"Prompt": "ChatPromptValue",
"Chat Model": "AIMessage"
```
Prompt는 질문과 근거를 모아 ChatPromptValue를 만들 뿐 아직 API를 호출하지 않는다.  
Chat Model은 프롬프트를 가지고 AI model에 전달해 AIMessage를 받는다.  

## 2. Document에서 본문을 읽는 표현은 무엇인가요?

`"Document": "page_content와 metadata를 담아 생성"`  
`page_content`이다.

## 3. Document 목록을 question·context가 필요한 Prompt에 그대로 전달해도 되나요?

안 된다.  
`"Prompt": "question·context가 있는 dict"`
Prompt는 {'question': 문자열, 'context': 문자열} dict를 입력으로 받는다.  