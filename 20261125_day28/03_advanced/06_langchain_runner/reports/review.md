
## 질문별 검색/생성 결과 + 로컬 평가 리포트

### Q01 · live 검색/생성 결과
 
| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 취소하는 방법은? |
| mode / source_mode | live / live |
| dataset_version | d28-eval-v1 |
| langchain_runnable | True |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| included = ranked 상위 2개 | True |
| candidate_ids (검색 직후 후보) | output에 없음 |
| status | answered |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택한 뒤 ‘취소’를 누르세요. 예약 시작 2시간 전까지 취소할 수 있습니다. |
| sources (출처 목록) | output에 없음 (sources_valid 만 있음) |
| sources_valid | True |
| no_answer_result | not_no_answer |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 | 1.0 · 1 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| RR@3 | 1.0 |
| DCG@3 · nDCG@3 | 7.0 · 1.0 (1위 등급 3 -> (2^3-1)/log2(2) = 7.0, IDCG 7.0) |
| latency_ms (전체) | 21042.209 |
| timing_ms.rewrite | 5137.596 |
| timing_ms.multiquery | 0.005 |
| timing_ms.retrieval | 375.534 |
| timing_ms.rerank | 4113.315 |
| timing_ms.context | 0.012 |
| timing_ms.generation | 1414.025 |
| Stage 합 / 전체와 차이 (계산) | 11040.487 / 10001.722 ms |
| audit.run_mode · variant · dataset_version | live · both · d28-eval-v1 |
| audit.source_ids | FAC-102-C1, FAC-109-C1 |
| audit.source_ids = included_ids | True |
| audit.secrets_logged · trace_fields_checked | False · True (audit_runner() 고정값) |
| Gold (data/gold_labels.json, Evaluator 용 참고) | FAC-102-C1 |
| Gold 순위 (참고 계산) | 1 |
| included 중 Gold 아닌 문서 (등급) | FAC-109-C1(0) |
 
### Q02 · live 검색/생성 결과
 
| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 그럼 취소는 어떻게 해? |
| mode / source_mode | live / live |
| dataset_version | d28-eval-v1 |
| langchain_runnable | True |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-108-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| included = ranked 상위 2개 | True |
| candidate_ids (검색 직후 후보) | output에 없음 |
| status | answered |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택해 취소하면 됩니다. 예약 시작 2시간 전까지 취소할 수 있습니다. |
| sources (출처 목록) | output에 없음 (sources_valid 만 있음) |
| sources_valid | True |
| no_answer_result | not_no_answer |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 | 1.0 · 1 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| RR@3 | 1.0 |
| DCG@3 · nDCG@3 | 7.0 · 1.0 (1위 등급 3 -> (2^3-1)/log2(2) = 7.0, IDCG 7.0) |
| latency_ms (전체) | 3639.205 |
| timing_ms.rewrite | 2160.203 |
| timing_ms.multiquery | 0.006 |
| timing_ms.retrieval | 51.248 |
| timing_ms.rerank | 208.1 |
| timing_ms.context | 0.012 |
| timing_ms.generation | 1218.312 |
| Stage 합 / 전체와 차이 (계산) | 3637.881 / 1.324 ms |
| audit.run_mode · variant · dataset_version | live · both · d28-eval-v1 |
| audit.source_ids | FAC-102-C1, FAC-109-C1 |
| audit.source_ids = included_ids | True |
| audit.secrets_logged · trace_fields_checked | False · True (audit_runner() 고정값) |
| Gold (data/gold_labels.json, Evaluator 용 참고) | FAC-102-C1 |
| Gold 순위 (참고 계산) | 1 |
| included 중 Gold 아닌 문서 (등급) | FAC-109-C1(0) |
 
### Q03 · live 검색/생성 결과
 
| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 철회하려면? |
| mode / source_mode | live / live |
| dataset_version | d28-eval-v1 |
| langchain_runnable | True |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-103-C1 > 4.FAC-107-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| included = ranked 상위 2개 | True |
| candidate_ids (검색 직후 후보) | output에 없음 |
| status | answered |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 회의실 예약을 선택한 뒤 ‘취소’를 누르세요. 예약 시작 2시간 전까지만 취소할 수 있습니다. |
| sources (출처 목록) | output에 없음 (sources_valid 만 있음) |
| sources_valid | True |
| no_answer_result | not_no_answer |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 | 1.0 · 1 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| RR@3 | 1.0 |
| DCG@3 · nDCG@3 | 7.0 · 1.0 (1위 등급 3 -> (2^3-1)/log2(2) = 7.0, IDCG 7.0) |
| latency_ms (전체) | 5284.322 |
| timing_ms.rewrite | 3641.465 |
| timing_ms.multiquery | 0.002 |
| timing_ms.retrieval | 52.631 |
| timing_ms.rerank | 175.375 |
| timing_ms.context | 0.01 |
| timing_ms.generation | 1413.716 |
| Stage 합 / 전체와 차이 (계산) | 5283.199 / 1.123 ms |
| audit.run_mode · variant · dataset_version | live · both · d28-eval-v1 |
| audit.source_ids | FAC-102-C1, FAC-109-C1 |
| audit.source_ids = included_ids | True |
| audit.secrets_logged · trace_fields_checked | False · True (audit_runner() 고정값) |
| Gold (data/gold_labels.json, Evaluator 용 참고) | FAC-102-C1 |
| Gold 순위 (참고 계산) | 1 |
| included 중 Gold 아닌 문서 (등급) | FAC-109-C1(0) |
 
### 로컬 평가 리포트 (live evaluation 3건을 report_summary() 로 요약, 계산)
 
| 항목 | 값 |
|:--|:--|
| mean_recall_at_k | 1.0 |
| mean_precision_at_k | 0.3333333333333333 |
| mean_hit_at_k | 1.0 |
| mean_rr_at_k | 1.0 |
| mean_ndcg_at_k | 1.0 |
| mrr_at_k | 1.0 |
| evaluated_questions | 3 |
| no_answer_excluded | 0 |
| source_integrity_failures | 0 |
| insufficient_context | 0 |
| status_failures | 0 |
| no_answer_correct_abstentions | 0 |
| no_answer_unsupported_answers | 0 |
| mode | live |
| data_version | d28-eval-v1 |
| variant | both |