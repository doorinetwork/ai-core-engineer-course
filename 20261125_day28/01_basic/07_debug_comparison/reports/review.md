
## Variant별 지표 비교·실패 사례·개선 결정

### Q01 · live Baseline vs Rerank

| 항목 | 값 |
|:--|:--|
| 질문<br> (data/questions.json 참고) | 회의실 예약을 취소하는 방법은? |
| mode | live |
| dataset_version | d28-eval-v1 |
| comparison_kind | offline_version_comparison_not_online_AB |
| 비교 변형 | baseline vs rerank_only (live 는 Q01 1개 질문만, 코드 고정) |
| baseline_recall (Recall@3) | 1.0 |
| rerank_recall (Recall@3) | 1.0 |
| delta_recall (rerank - baseline) | 0.0 |
| delta 해석 | Rerank 로 Top-3 안의 Gold 개수는 변하지 않았다 |
| diagnosis (baseline 실행 기준) | inspect_answer_grounding |
| diagnosis 조건 (diagnose() 분기) | baseline 실행이 status answered 이고 Gold 가 후보·상위 2개·Context 에 모두 있음 (출처 ID 만으로 사실 충실성 인증 불가) |
| RR@3 · nDCG@3 | output에 없음 |
| ranked_ids · included_ids · status | output에 없음 |
| timing_ms | output에 없음 |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |

### Before/After 와 개선 결정

| 항목 | 값 |
|:--|:--|
| Before (baseline) Recall@3 | 1.0 |
| After (rerank_only) Recall@3 | 1.0 |
| 변화량 | 0.0 |
| 실패 사례 | 이 output 에는 없다. Q01 은 diagnosis inspect_answer_grounding 으로 검색 단계 실패가 아니다 |
| 개선 결정 | 보류. Recall@3 가 같아 Rerank 를 켤 근거가 이 output 에 없다. 순위 지표(RR·nDCG)·시간이 output에 없어 Rerank 효과를 판단할 수 없다 |
| 이유 | Gold FAC-102-C1 은 baseline 에서도 이미 상위 2개·Context 에 있었다 (diagnose 통과). 남은 확인은 답변 근거(생성 단계)다 |
| 판단 범위 | Q01 1건·1회 실행이라 다른 질문이나 평균 효과로 일반화하지 않는다 |