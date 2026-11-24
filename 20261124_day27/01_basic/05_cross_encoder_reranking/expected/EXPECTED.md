# 실습 05 · Cross-Encoder Reranking · 완료 기준

- 출처: 11월 24일 HTML **54페이지** 실습 안내.
- 실행: 이 폴더의 `python main.py --offline`에서 기능·오류 처리를 검증한 뒤 기본 실습 실행을 수행합니다.
- 모든 TODO를 완성하고 `outputs/day27_05_fixture.json`의 입력·출력 필드를 확인합니다.
- 실제 실행은 환경 준비 후 `python main.py`입니다. 01번은 고정 진단 사례만 사용합니다.
- 실측과 고정 Fixture 결과를 구분해 보고서에 기록합니다.
- 핵심 점검: CUDA Cross-Encoder가 질문–문서 쌍에 부여한 점수로 후보만 재정렬합니다.
