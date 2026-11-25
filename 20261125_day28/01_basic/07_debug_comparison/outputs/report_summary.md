# Day28 실습 07 · Baseline vs Rerank 비교 보고서

## 1. 실행 정보

| 항목 | 값 |
|:--|:--|
| mode | live |
| dataset_version | d28-eval-v1 |
| comparison_kind | offline_version_comparison_not_online_AB |
| 비교 질문 수 | 1 (Q01) |
| 입력 파일 | result.json |

- 주의: mode 가 live 다. 실제 RAG 를 한 번 실행한 값이라 반복 실행 전에는 결론을 확정하지 않는다.
- 주의: comparison_kind 는 같은 질문셋으로 두 버전(baseline·rerank_only)을 비교한 오프라인 버전 비교다. 실사용자 온라인 A/B 실험이 아니다.

## 2. 질문별 비교

### Q01

| 항목 | 값 |
|:--|:--|
| baseline_recall (Recall@3) | 1.0 |
| rerank_recall (Recall@3) | 1.0 |
| delta_recall (rerank - baseline) | 0.0 |
| diagnosis (baseline 실행 기준) | inspect_answer_grounding |
| diagnosis 조건 | 위 검사를 모두 통과 (출처 ID 만으로 사실 충실성은 인증 불가) |
| 먼저 볼 곳 | 답변 근거(Claim) 직접 확인 |

## 3. 평균 (해당 없음 제외)

| 항목 | 값 |
|:--|:--|
| baseline_recall 평균 | 1.0 (1/1건) |
| rerank_recall 평균 | 1.0 (1/1건) |
| delta_recall 평균 | 0.0 (1/1건) |
| 평균에서 뺀 질문 | 없음 |

## 4. diagnosis 범례 (coursekit/eval_experiments.py diagnose(), 위에서부터 검사)

| 값 | 조건 | 먼저 볼 곳 |
|:--|:--|:--|
| system_failure | status 가 answered·insufficient_context 가 아님 | 실행 오류 (Target·LLM 호출) |
| no_answer_case | Gold 가 비어 있음 (답이 없는 질문) | 정보 부족 응답을 별도로 평가 |
| candidate_miss | candidate_ids 에 Gold 가 없음 | Query Rewrite·검색 단계 |
| ranking_miss | ranked_ids 상위 2개에 Gold 가 없음 | Reranker·Top-N (검색 순위) |
| context_miss | included_ids(LLM Context) 에 Gold 가 없음 | Context 조립·context_k |
| generation_or_validation | Gold 는 Context 에 있으나 status 가 answered 가 아님 | 생성·출력 검증 |
| inspect_answer_grounding | 위 검사를 모두 통과 (출처 ID 만으로 사실 충실성은 인증 불가) | 답변 근거(Claim) 직접 확인 |
