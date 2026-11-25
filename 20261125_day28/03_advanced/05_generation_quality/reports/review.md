
## Claim-level Faithfulness·질문 적합성 비교표
 
### G01 · Faithfulness·적합성·정답성 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 15분 지각 기준은? |
| Context | 15분까지 지각, 15분 초과 결석 |
| Claim (사람 레이블) | 15분까지는 지각(지원), 15분을 초과하면 결석(지원), 벌금이 5만원(비지원) |
| faithfulness | 2/3 = 0.6667 |
| reason | 근거 불충분: 벌금이 5만원 |
| answer_relevance | 높음 |
| answer_correctness | 부분정답 |
| label_source | human_fixture |
| human_review | judge=human_fixture · reviewer=교육용 검수자 · disagreement=추가 사람 검토 필요 · publishable=False |
| 해석 | Claim 3개 중 2개만 Context 가 지지 -> 2/3. 질문에는 답했으나(관련성 높음) 근거 없는 벌금 주장이 섞여 부분정답 |
 
### G02 · Faithfulness·적합성·정답성 
| 항목 | 값 |
|:--|:--|
| 질문 | 정확한 과제 마감일은? |
| Context | 과제 제출일은 담당자의 공지에 따름 |
| Claim (사람 레이블) | 내일 오후 6시 마감(비지원) |
| faithfulness | 0/1 = 0.0000 |
| reason | 근거 불충분: 내일 오후 6시 마감 |
| answer_relevance | 높음 |
| answer_correctness | 오답 |
| label_source | human_fixture |
| human_review | judge=human_fixture · reviewer=교육용 검수자 · disagreement=추가 사람 검토 필요 · publishable=False |
| 해석 | Context 는 "담당자 공지에 따름" 뿐인데 구체 날짜를 지어냄: faithfulness = 0/1. 질문에 직접 답해 관련성은 높지만 오답  |
 
### G03 · Faithfulness·적합성·정답성 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 과제 제출기한은? |
| Context | 회의실은 운영실 승인 후 사용 |
| Claim (사람 레이블) | 회의실 사용은 승인 필요(지원) |
| faithfulness | 1/1 = 1.0000 |
| reason | 모든 Claim이 제공된 Context와 일치 |
| answer_relevance | 낮음 |
| answer_correctness | 오답 |
| label_source | human_fixture |
| human_review | judge=human_fixture · reviewer=교육용 검수자 · disagreement=추가 사람 검토 필요 · publishable=False |
| 해석 | Claim 1개가 Context 와 일치해 Faithfulness 1.0. 그러나 과제 기한 질문에 회의실 승인 문장으로 답해 관련성 낮음·오답. 근거 충실도와 질문 적합성·정답성은 별개 |
 
### G04 · Faithfulness·적합성·정답성 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 정확한 과제 마감일은? |
| Context | 정확한 날짜는 별도 공지 예정 |
| Claim (사람 레이블) | [] (Claim 없음) |
| faithfulness | None (Claim 0개, 분모 0) |
| reason | 검토할 Claim 없음: 별도 판정 필요 |
| answer_relevance | 높음 |
| answer_correctness | 정답 |
| label_source | human_fixture |
| human_review | judge=human_fixture · reviewer=교육용 검수자 · disagreement=추가 사람 검토 필요 · publishable=False |
| 해석 | Claim 이 0개라 지지 비율의 분모가 0 -> None (정의 불가). 0.0 은 "주장이 모두 틀림"이라는 뜻이라 다르다. 정보 부족을 밝힌 답이라 정답이며, Faithfulness 대신 별도 판정 필요 |