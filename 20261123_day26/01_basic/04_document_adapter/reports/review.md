# 실습 04 확인 기록

- 선택한 수준: 01_basic
- 실행일: 2026-11-23
- 실행 명령: `python main.py`
- 결과 파일: outputs/document_adapter.json
- 입력 출처와 실행한 구성요소: 
`data/lecture_hits.json` Hit 201 1건  
`mode` local_adapter  
`db` false  
`embedding` false  
`llm` false  

## 1. Document 생성 성공이 실제 검색 성공을 뜻하나요?

아니다. 이 실습에서는 example_hit() 한 건을 불러와 입력했다. 실제 검색은 `--live-db` 모드로 실행해야 한다.

## 2. citation_id를 항상 201로 넣으면 어떤 문제가 생기나요?

모든 문서의 근거가 201이 될 수 없으므로 근거 불일치가 생기게 된다.

## 3. 빈 Hit 목록과 source_locator 누락은 왜 다르게 다루나요?

목록이 비어있는 정상 상태와 소스 출처가 누락된 오류 상태를 구분해야 한다. 빈 목록은 검색 결과가 없는 정상 상태이고, source_locator 누락은 소스 출처가 없는 오류 상태이다.

