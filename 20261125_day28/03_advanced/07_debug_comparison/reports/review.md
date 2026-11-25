
## Variant별 지표 비교·실패 사례·개선 결정

### Q01 · live Baseline vs Rerank

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 취소하는 방법은? |
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
| RR@3 · nDCG@3 | baseline 1.0 · 1.0 / rerank_only 1.0 · 1.0 (matrix) |
| ranked_ids · included_ids · status | output에 없음 |
| timing_ms | output에 없음 |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| matrix 변형 | baseline, rerank_only (live 는 코드상 baseline·rerank_only 2개만, rewrite_only·both 없음) |
| matrix 두 변형 지표 동일 | True |
| regression.regression | False |
| regression.error_count | 0 (execute(7) 가 regression_decision(…, 0) 으로 항상 0 을 넘김: 실제 오류를 센 값 아님) |
| regression.hold_release | False (regression or error_count>0 으로 계산) |
| regression.reason | 추가 사람 평가 필요 (게이트 통과 시 나오는 고정 문구, 사람 평가가 실행된 기록 아님) |

### Q01 · matrix 항목 대비 (baseline vs rerank_only)

| 지표 | baseline | rerank_only | 차이 |
|:--|:--|:--|:--|
| recall_at_k | 1.0 | 1.0 | 0.0 |
| precision_at_k | 0.3333333333333333 | 0.3333333333333333 | 0.0 |
| hit_at_k | 1 | 1 | 0 |
| rr_at_k | 1.0 | 1.0 | 0.0 |
| dcg_at_k | 7.0 | 7.0 | 0.0 |
| ndcg_at_k | 1.0 | 1.0 | 0.0 |
| k | 3 | 3 | 0 |
| gold_count | 1 | 1 | 0 |
| retrieved_count | 3 | 3 | 0 |

- 두 변형의 지표 9개가 모두 같다. baseline 에서 이미 RR@3 1.0(Gold 1위)이라 Rerank 가 올릴 순위가 없었다. 이 output 으로는 Rerank 의 변경 효과가 **없음**만 말할 수 있고, ranked_ids 가 없어 2·3위 순서가 바뀌었는지는 알 수 없다.
- matrix 에 rewrite_only·both 가 없다. live 는 비용 때문에 코드가 2개 변형만 실행하므로, 변경 단계 분리(Rewrite·Rerank 각각)는 이번 live 로 확인되지 않는다.

### Before/After 와 개선 결정

| 항목 | 값 |
|:--|:--|
| Before (baseline) Recall@3 · RR@3 · nDCG@3 | 1.0 · 1.0 · 1.0 |
| After (rerank_only) Recall@3 · RR@3 · nDCG@3 | 1.0 · 1.0 · 1.0 |
| 변화량 (delta_recall) | 0.0 |
| 회귀 게이트 | regression False · hold_release False · reason '추가 사람 평가 필요' |
| 개선 결정 | 보류. 지표가 같아 Rerank 를 켤 근거가 없다. hold_release False 는 배포 승인이 아니라 '회귀가 없었다'는 뜻이다 |
| 이유 | Gold FAC-102-C1 이 baseline 에서 이미 1위·Context 에 있었다 (diagnosis inspect_answer_grounding). 남은 확인은 답변 근거(생성 단계)다 |
| 판단 범위 | Q01 1건·1회 실행이라 다른 질문이나 평균 효과로 일반화하지 않는다 |