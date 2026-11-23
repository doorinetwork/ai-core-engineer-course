# 실습 03 완료 기준

이 문서는 실행 로그가 아니라 확인해야 할 조건입니다. 기본 실행의 예시 데이터에는 고정 ID가 있지만 실제 DB ID와 동일하다고 가정하지 않습니다.

- 네 구성요소 각각에 input·output·access가 있는 표가 생성됩니다.
- Prompt의 출력은 ChatPromptValue이며 Chat Model의 출력은 AIMessage입니다.
- Document는 본문·출처를 담는 data로 표시하고 실제 API는 호출하지 않습니다.

## 제출 전 확인

- outputs/object_contracts.json가 현재 폴더 실행으로 새로 만들어졌는가?
- 각 조건에 대응하는 필드를 직접 가리킬 수 있는가?
- reports/review.md에 결과의 이유와 실제 실행 여부를 적었는가?

정상적인 mentor 기본 실행에서 생성한 local_reference.json이 제공되는 경우, 이는 고정 예시 데이터와 로컬 코드의 결과입니다. DB·LLM 실측 결과가 아닙니다. 실제 연동은 해당 실습의 별도 실행 명령으로 확인합니다.
