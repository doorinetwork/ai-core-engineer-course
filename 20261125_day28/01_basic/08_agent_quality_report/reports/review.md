## Day28 평가 리포트와 Day29 Agent Tool 준비 체크리스트

### Day28 평가 리포트 · summary (live)

| 항목 | 값 |
|:--|:--|
| 평가 질문 | Q01, Q02, Q03 (live 는 코드상 questions[:3]) |
| mean_recall_at_k | 1.0 |
| mean_precision_at_k | 0.3333 |
| mean_hit_at_k | 1.0 |
| mean_rr_at_k | 1.0 |
| mean_ndcg_at_k | 1.0 |
| mrr_at_k | 1.0 |
| evaluated_questions | 3 |
| no_answer_excluded | 0 (No-answer 질문 Q06 이 live 에 없음: 검증된 0 이 아니라 해당 질문 없음) |
| source_integrity_failures | 0 |
| insufficient_context | 0 |
| status_failures | 0 |
| no_answer_correct_abstentions | 0 (No-answer 질문 Q06 이 live 에 없음: 검증된 0 이 아니라 해당 질문 없음) |
| no_answer_unsupported_answers | 0 (No-answer 질문 Q06 이 live 에 없음: 검증된 0 이 아니라 해당 질문 없음) |
| mode | live |
| data_version | d28-eval-v1 |
| variant | both |

- 세 질문 모두 지표가 같다. 3질문 모두 Gold 가 FAC-102-C1 하나이고(등급 3), live 에서 3건 모두 FAC-102-C1 이 1위라 Recall·Hit·RR·nDCG 가 1.0 이다. Precision@3 는 Gold 가 1개라 상위 3개 중 최대 1개만 맞아 1/3 = 0.3333 이 최댓값이다.
- no_answer_excluded·no_answer_correct_abstentions·no_answer_unsupported_answers 의 0 은 **검증된 0 이 아니라 해당 질문 없음** 이다. live 는 Q01~Q03 만 실행해 No-answer 질문(Q06)이 없다.

### Q01 · live 평가 레코드

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 취소하는 방법은? |
| mode · variant · data_version | live · both · d28-eval-v1 |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| candidate_ids | output에 없음 |
| sources (인용 출처 목록) | output에 없음 (sources_valid 만 있음) |
| status | answered |
| sources_valid | True |
| no_answer_result | not_no_answer |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택하고 취소를 누르세요. 시작 2시간 전까지 취소할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 16216.181 |
| timing_ms.rewrite | 1693.199 |
| timing_ms.multiquery | 0.006 |
| timing_ms.retrieval | 382.347 |
| timing_ms.rerank | 4515.864 |
| timing_ms.context | 0.014 |
| timing_ms.generation | 1153.281 |
| Stage 합 / 전체와 차이 (계산) | 7744.711 / 8471.47 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### Q02 · live 평가 레코드

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 그럼 취소는 어떻게 해? |
| mode · variant · data_version | live · both · d28-eval-v1 |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-108-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| candidate_ids | output에 없음 |
| sources (인용 출처 목록) | output에 없음 (sources_valid 만 있음) |
| status | answered |
| sources_valid | True |
| no_answer_result | not_no_answer |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 취소를 선택하면 됩니다. 취소는 예약 시작 2시간 전까지 할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 3690.959 |
| timing_ms.rewrite | 1922.05 |
| timing_ms.multiquery | 0.005 |
| timing_ms.retrieval | 50.373 |
| timing_ms.rerank | 102.614 |
| timing_ms.context | 0.012 |
| timing_ms.generation | 1614.489 |
| Stage 합 / 전체와 차이 (계산) | 3689.543 / 1.416 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### Q03 · live 평가 레코드

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 철회하려면? |
| mode · variant · data_version | live · both · d28-eval-v1 |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| candidate_ids | output에 없음 |
| sources (인용 출처 목록) | output에 없음 (sources_valid 만 있음) |
| status | answered |
| sources_valid | True |
| no_answer_result | not_no_answer |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택해 취소하세요. 시작 2시간 전까지 취소할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 3386.001 |
| timing_ms.rewrite | 1768.738 |
| timing_ms.multiquery | 0.005 |
| timing_ms.retrieval | 47.72 |
| timing_ms.rerank | 191.272 |
| timing_ms.context | 0.014 |
| timing_ms.generation | 1376.783 |
| Stage 합 / 전체와 차이 (계산) | 3384.532 / 1.469 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### Day29 Agent Tool 준비 체크리스트 (live summary 로 계산, 제공 모듈 release_gate 참고)

| 검사 | 통과 | 근거 |
|:--|:--|:--|
| retrieval_recall (≥0.70) | True | mean_recall_at_k 1.0 |
| retrieval_mrr (≥0.60) | True | mrr_at_k 1.0 |
| source_integrity | True | source_integrity_failures 0 |
| execution_status | True | status_failures 0 |
| human_review | False | summary 에 human_review_passed 키가 없어 기본값 False (사람 검토 기록 output에 없음) |
| access_control_tests | False | summary 에 access_control_passed 키가 없어 기본값 False (권한 테스트 output에 없음) |
| ready_for_agent_review | False | 6개 모두 통과해야 True |
| agent_gate (output) | output에 없음 | 기본필수는 agent_release_gate hook 이 없다 (심화 TODO) |
