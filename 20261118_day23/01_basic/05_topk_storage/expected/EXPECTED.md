# 실습 05 · 검증할 결과

day23_search_docs에 9개의 ID·원문·벡터가 대응하고, 쿼리는 embedding <=> query ASC LIMIT k로 검색합니다. fixture는 실제 pgvector SQL 결과가 아닙니다.

실습 난이도: 01_basic / TODO 3개

- `python main.py --mode fixture`는 로컬 모의 검증이며 E5/DB 실측이 아닙니다.
- `python main.py --mode live`의 실제 성공은 설치·DB·모델 다운로드가 준비된 Windows PC에서만 확인할 수 있습니다.
- 오류 또는 미실행은 성공·빈 결과·비용 0으로 임의 변환하지 않습니다.
- JSON 출력은 `outputs/exercise_05.json`에 저장됩니다.
