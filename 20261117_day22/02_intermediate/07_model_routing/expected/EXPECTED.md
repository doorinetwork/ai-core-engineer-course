# 명시적 모델 라우팅 — 예상과 실측 구분

잘못된mode전송0. compare는명시적으로만2모델을비교합니다.

## 실행 유형
- `api_live`: 본인 키로 실제 공급자에 전송한 결과. 실패도 같은 출처로 기록합니다.
- `simulated`: 제공 응답·이벤트·예외. 모의 토큰과 시간은 실측 성능이 아닙니다.
- `app_cache`: 승인한 원래 답변을 읽음. 새 전송0, 원래실행ID와origin_source_mode를확인합니다.
- `not_run`: 사전 조건·권한·예산 때문에 요청하지 않음. 성공 응답을 채우지 않습니다.

## 핵심 점검
status·source_mode·answer_object·attempts·usage·evidence_ok를구분하세요. `evidence_ok=None`은아직사람이검토하지않았다는뜻입니다. 사용량을받지못한None과캐시hit의0은다릅니다.

정상 raw는단정된문장보다선택한근거와계약에맞는지평가합니다. 실패 raw는수정해덮어쓰지않습니다.
