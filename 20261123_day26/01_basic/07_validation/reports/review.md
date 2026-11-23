# 실습 07 · 원문 대조 기록

- 선택한 수준: basic
- 실행한 파일: `python main.py`
- 실제 실행한 범위: 고정 입력 Validation / DB·Embedding·LLM 미실행

## 공통 확인

1. Context에 들어간 ID와 허용 ID는 각각 무엇인가요?  
Context는 출석 Hit 1건으로 만든 [201] 블록이고, 허용 ID는 포함 문서에서 얻은 {201}이다 (`included_ids = [201]`).  
2. normal 답변의 숫자·조건·예외를 지지하는 원문 문장을 그대로 적고 위치를 밝히세요.  
attendance_v2.md#sec-1 (201): "수업 시작 후 15분까지 입실하면 지각으로 기록합니다. 15분을 초과하면 결석으로 기록합니다. 단, 사전 승인을 받은 경우에는 운영실에 확인합니다."  
숫자: '15분까지'  
조건: '지각'  
예외: '단, 사전 승인…운영실에 확인'  
3. unknown_source는 어떤 단계에서 거절됐나요? 검색 후보에만 있던 ID도 허용해도 되나요?  
`WireAnswer` 파싱은 통과하고  
`validate_answer`의 허용 ID 검사(source_not_provided)에서 거절되어 `answer_validation_error`가 기록됐다.  
검색 후보에만 있고 Context에 넣지 않은 ID는 이 모델이 확인한 근거가 아니므로 허용해서는 안 된다.
4. insufficient_context가 시스템 오류와 다른 이유를 적으세요.  
`insufficient_context`는 응답이 정상 완료되고 규칙을 지킨 '근거 부족' 답변이라 채택되었다.  
`시스템 오류`(생성 실패·미완료·파싱 실패)는 RagServiceError로 거절된다.

자동 검사 통과 여부와 원문 대조 결과를 별도로 기록합니다.
