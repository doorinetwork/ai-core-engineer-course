# 실습 06 · 기대 조건

이 문서는 **완료 기준**입니다. 실제 모델 측정 결과나 정답 응답 파일이 아닙니다.

## 공통 실행 표기

- `python main.py`: 실제 모델의 응답을 받은 경우에만 LLM 실행을 true로 기록합니다.
- `python main.py --offline`: 고정 예시입니다. `real_model_calls=0`이며 실제 LLM은 미실행입니다.
- 모델 오류로 실행이 중단되면 이전 outputs 파일을 이번 결과로 제출하지 않습니다. 파일 수정 시각과 터미널 오류를 확인합니다.

## 확인 조건

- 기본 Context에는 출석 201·과제 202의 전체 블록이 들어가고 `included_ids`는 `[201, 202]`입니다. 이는 제공된 교육용 Document의 ID입니다.
- Context 본문과 출처 목록은 같은 `included_docs`에서 나옵니다.
- `bundle`의 Key는 `raw`, `parsed`, `parsing_error`입니다. 정상 고정 예시에서는 raw가 AIMessage, parsed가 WireAnswer에서 변환된 네 필드 객체, parsing_error는 null입니다.
- 실제 실행에서 오류 또는 거절이 발생했다면 고정 정상 예시와 다르게 기록될 수 있습니다. parsed를 정답으로 임의 교체하지 않습니다.
- ID 포함 여부를 관찰한 값은 Grounding 검사 결과가 아닙니다. 답변 주장을 원문과 대조해야 합니다.
- 중복 첫 청크를 넣어도 포함 ID는 `[201, 202]`이고, `max_items=1`이면 `[201]`입니다. 다른 조건은 같게 유지합니다. 이 비교의 모델 호출은 0회입니다.

## 제출할 확인

`outputs/`의 실행 기록과 `reports/review.md`의 원문 대조를 함께 확인합니다. API Key나 DB 비밀번호는 제출하지 않습니다.

## 고정 예시 파일

`offline_reference_context_chain.json`는 강사용 완성 코드를 `--offline`으로 실행해 만든 비교용 기록입니다. 실제 LLM 응답이나 성능 측정값이 아닙니다.
