
## 질문별 Recall@k·Precision@k·Hit@k 표
 
### Q01 · Recall@3·Precision@3·Hit@3 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 회의실 예약을 취소하는 방법은? |
| Top-3 (ranked_ids[:3]) | FAC-101-C1, FAC-107-C1, FAC-102-C1 |
| Gold | FAC-102-C1 |
| hits_at_3 | 1 |
| gold_count | 1 |
| recall_at_3 (교집합/Gold 수) | 1/1 = 1.0000 |
| precision_at_k (교집합/k) | 1/3 = 0.3333 |
| hit_at_k | 1 |
| 해석 | Top-3 안 Gold 1개 · 전체 Gold 1개 |
 
### Q07 · Recall@3·Precision@3·Hit@3 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 시설 예약 승인과 취소의 차이는? |
| Top-3 (ranked_ids[:3]) | FAC-101-C1, FAC-102-C1, FAC-107-C1 |
| Gold | FAC-104-C1, FAC-102-C1 |
| hits_at_3 | 1 |
| gold_count | 2 |
| recall_at_3 (교집합/Gold 수) | 1/2 = 0.5000 |
| precision_at_k (교집합/k) | 1/3 = 0.3333 |
| hit_at_k | 1 |
| 해석 | Top-3 안 Gold 1개 · 전체 Gold 2개 |
 
### Q06 · Recall@3·Precision@3·Hit@3 
 
| 항목 | 값 |
|:--|:--|
| 질문 | 정확한 과제 마감 날짜는? |
| Top-3 (ranked_ids[:3]) | [] |
| Gold | [] |
| hits_at_3 | 0 |
| gold_count | 0 |
| recall_at_3 (교집합/Gold 수) | None (분모 0) |
| precision_at_k (교집합/k) | None |
| hit_at_k | None |
| 해석 | Gold 0개(No-answer) -> Recall 은 분모(Gold 수)가 0 이라 정의 불가. <br>Precision·Hit 은 맞힐 Gold 가 없어 늘 0 이 되므로 검색 품질을 뜻하지 않아 precision_and_hit() 가 None 으로 둔다. 0점(검색 실패)이 아니며 평균에서 제외하고 정보 부족 응답(insufficient_context)으로 따로 평가. |