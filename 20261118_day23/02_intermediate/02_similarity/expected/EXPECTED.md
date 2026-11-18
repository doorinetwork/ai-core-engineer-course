# 실습 02 · 검증할 결과

q=(1,0), A=(0.8,0.6): Dot=0.8, Cosine=0.8, L2≈0.632456, Cosine distance=0.2. Cosine은 클수록, 거리는 작을수록 가깝습니다.

실습 난이도: 02_intermediate / TODO 5개

- `python main.py --mode fixture`는 로컬 모의 검증이며 E5/DB 실측이 아닙니다.
- `python main.py --mode live`의 실제 성공은 설치·DB·모델 다운로드가 준비된 Windows PC에서만 확인할 수 있습니다.
- 오류 또는 미실행은 성공·빈 결과·비용 0으로 임의 변환하지 않습니다.
- JSON 출력은 `outputs/exercise_02.json`에 저장됩니다.
