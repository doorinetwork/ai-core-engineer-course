# 실습 02 확인 기록

- 선택한 수준: 01_basic
- 실행일: 2026-11-23
- 실행 명령: `python main.py`
- 결과 파일: outputs/hit_context.json
- 입력 출처와 실행한 구성요소:  
`data/lecture_hits.json` Hit 201 1건  
`mode` local_python  
`db` false  
`embedding` false  
`llm` false

## 1. 본문과 ID를 같은 반복문에서 모으는 이유는 무엇인가요?

context에 반영하는 근거와 ID가 서로 어긋나지 않도록 함께 모으기 위해서이다.

## 2. hits[:2]인데 한 건만 있는 경우 어떻게 되나요?

슬라이스를 통해 건 수가 2보다 적더라도 존재하는 1건만 선택된다. 이 실습에서는 `included_ids = [201]` 한 건이 선택된다.

## 3. len(context)는 Token 수인가요?

아니다. context 문자열의 글자 수이다.

