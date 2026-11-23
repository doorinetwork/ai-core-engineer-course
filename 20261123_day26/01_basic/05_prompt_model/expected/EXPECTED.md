# 실습 05 · 기대 조건

이 문서는 **완료 기준**입니다. 실제 모델 측정 결과나 정답 응답 파일이 아닙니다.

## 공통 실행 표기

- `python main.py`: 실제 모델의 응답을 받은 경우에만 LLM 실행을 true로 기록합니다.
- `python main.py --offline`: 고정 예시입니다. `real_model_calls=0`이며 실제 LLM은 미실행입니다.
- 모델 오류로 실행이 중단되면 이전 outputs 파일을 이번 결과로 제출하지 않습니다. 파일 수정 시각과 터미널 오류를 확인합니다.

## 확인 조건

- `prompt_value_type`은 `ChatPromptValue`, 메시지 역할은 system·human입니다.
- `response.type`은 `AIMessage`이며 텍스트는 모델 출력의 관찰값입니다. 실제 문장의 완전 일치를 요구하지 않습니다.
- 출석 원문에는 15분까지·15분 초과·사전 승인 조건이 함께 있어야 합니다.
- 모델에 보내기 전의 Prompt와 생성 후의 AIMessage를 구분하여 설명합니다.
- `human_evidence_review`는 원문 대조 전 `pending`입니다. 자동 정답 점수가 아닙니다.

## 제출할 확인

`outputs/`의 실행 기록과 `reports/review.md`의 원문 대조를 함께 확인합니다. API Key나 DB 비밀번호는 제출하지 않습니다.

## 고정 예시 파일

`offline_reference_prompt_model.json`는 강사용 완성 코드를 `--offline`으로 실행해 만든 비교용 기록입니다. 실제 LLM 응답이나 성능 측정값이 아닙니다.
