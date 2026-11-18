# 실습 03 · 검증할 결과

정규화된 문서 Embedding (3,384) 또는 (9,384), 질문 (1,384). fixture는 실제 E5 모델의 벡터가 아닙니다.

실습 난이도: 02_intermediate / TODO 5개

- `python main.py --mode fixture`는 로컬 모의 검증이며 E5/DB 실측이 아닙니다.
- `python main.py --mode live`의 실제 성공은 설치·DB·모델 다운로드가 준비된 Windows PC에서만 확인할 수 있습니다.
- 오류 또는 미실행은 성공·빈 결과·비용 0으로 임의 변환하지 않습니다.
- JSON 출력은 `outputs/exercise_03.json`에 저장됩니다.
