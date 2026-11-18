# 실습 04 · 검증할 결과

현재 DB와 vector 확장 상태를 확인합니다. pgAdmin 수치 probe에서 [1,0,0]과 [0,1,0]의 Cosine distance 기대값은 1. fixture는 설치 미검증(not_run)입니다.

실습 난이도: 01_basic / TODO 3개

- `python main.py --mode fixture`는 로컬 모의 검증이며 E5/DB 실측이 아닙니다.
- `python main.py --mode live`의 실제 성공은 설치·DB·모델 다운로드가 준비된 Windows PC에서만 확인할 수 있습니다.
- 오류 또는 미실행은 성공·빈 결과·비용 0으로 임의 변환하지 않습니다.
- JSON 출력은 `outputs/exercise_04.json`에 저장됩니다.
