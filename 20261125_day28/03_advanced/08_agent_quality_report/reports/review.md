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

- 세 질문 모두 지표가 같다. 3질문 모두 Gold 가 FAC-102-C1 하나(등급 3)이고 3건 모두 1위라 Recall·Hit·RR·nDCG 가 1.0 이다. Precision@3 는 Gold 1개·k=3 이라 1/3 = 0.3333 이 최댓값이다.
- no_answer 관련 카운트 0 은 **검증된 0 이 아니라 해당 질문 없음** 이다. live 는 Q01~Q03 만 실행해 No-answer 질문(Q06)이 없다.

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
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택해 취소하면 됩니다. 시작 2시간 전까지 취소할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 16149.485 |
| timing_ms.rewrite | 1910.423 |
| timing_ms.multiquery | 0.005 |
| timing_ms.retrieval | 335.527 |
| timing_ms.rerank | 4220.051 |
| timing_ms.context | 0.02 |
| timing_ms.generation | 1304.053 |
| Stage 합 / 전체와 차이 (계산) | 7770.079 / 8379.406 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### Q02 · live 평가 레코드

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 그럼 취소는 어떻게 해? |
| mode · variant · data_version | live · both · d28-eval-v1 |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-107-C1 > 4.FAC-103-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| candidate_ids | output에 없음 |
| sources (인용 출처 목록) | output에 없음 (sources_valid 만 있음) |
| status | answered |
| sources_valid | True |
| no_answer_result | not_no_answer |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약을 선택한 뒤 ‘취소’를 누르세요. 예약 시작 2시간 전까지 취소할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 3221.448 |
| timing_ms.rewrite | 1837.588 |
| timing_ms.multiquery | 0.006 |
| timing_ms.retrieval | 50.48 |
| timing_ms.rerank | 100.102 |
| timing_ms.context | 0.013 |
| timing_ms.generation | 1231.859 |
| Stage 합 / 전체와 차이 (계산) | 3220.048 / 1.4 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### Q03 · live 평가 레코드

| 항목 | 값 |
|:--|:--|
| 질문 (data/questions.json, 참고) | 회의실 예약을 철회하려면? |
| mode · variant · data_version | live · both · d28-eval-v1 |
| ranked_ids (rerank 후 순위) | 1.FAC-102-C1 > 2.FAC-109-C1 > 3.FAC-103-C1 > 4.FAC-107-C1 > 5.FAC-101-C1 |
| included_ids (LLM Context, context_k=2) | FAC-102-C1, FAC-109-C1 |
| candidate_ids | output에 없음 |
| sources (인용 출처 목록) | output에 없음 (sources_valid 만 있음) |
| status | answered |
| sources_valid | True |
| no_answer_result | not_no_answer |
| answer_preview | 예약 시스템의 ‘나의 예약’ 메뉴에서 해당 예약의 취소를 선택하세요. 예약 시작 2시간 전까지 취소할 수 있습니다. |
| k · gold_count · retrieved_count | 3 · 1 · 3 |
| Recall@3 · Hit@3 · RR@3 | 1.0 · 1 · 1.0 |
| Precision@3 | 0.3333 (상위 3개 중 Gold 1개 = 1/3) |
| DCG@3 · nDCG@3 | 7.0 · 1.0 |
| latency_ms (전체) | 3062.901 |
| timing_ms.rewrite | 1670.954 |
| timing_ms.multiquery | 0.006 |
| timing_ms.retrieval | 55.434 |
| timing_ms.rerank | 176.504 |
| timing_ms.context | 0.014 |
| timing_ms.generation | 1158.352 |
| Stage 합 / 전체와 차이 (계산) | 3061.264 / 1.637 ms |
| Gold (data/gold_labels.json, 참고) | FAC-102-C1 |
| included 중 Gold 아닌 문서 (등급, 참고) | FAC-109-C1(0) |

### markdown_report = outputs/report.md (live)

| 항목 | 값 |
|:--|:--|
| Run mode | live |
| Variant | both |
| Mean Recall@3 | 1.0 |
| MRR@3 | 1.0 |
| No-answer excluded | 0 |
| Source failures | 0 |
| Quality decision | 사람 검토 후 추가 판단 |
| 주의 | 교육용 Fixture 점수는 모델 품질 실측값이 아닙니다. |
| report.md 와 markdown_report 일치 | True |
| 질문별 결과 | 없음 (요약 6줄 + 판단·주의 2줄뿐) |

- `Quality decision` 은 write_summary_report() 의 조건 문구다. source_integrity_failures·status_failures 가 있으면 '점검 필요', 없으면 '사람 검토 후 추가 판단'. 실패가 0 이라 후자가 나왔고, 사람 검토가 실행된 기록은 아니다.
- `주의: 교육용 Fixture 점수는 모델 품질 실측값이 아닙니다.` 는 mode 와 상관없이 항상 붙는 고정 문구다. 이번 실행은 Run mode live 라 **문구와 mode 가 맞지 않는다**. live 보고서에서는 이 줄을 '실측 1회' 같은 mode 별 문구로 바꿔야 한다 (코드는 고치지 않음).

### Day29 Agent Tool 준비 체크리스트 (output agent_gate)

| 검사 | 통과 | 근거 구분 |
|:--|:--|:--|
| retrieval_recall (≥0.70) | True | 실제 계산: mean_recall_at_k 1.0 |
| retrieval_mrr (≥0.60) | True | 실제 계산: mrr_at_k 1.0 |
| source_integrity | True | 실제 계산: source_integrity_failures 0 |
| execution_status | True | 실제 계산: status_failures 0 |
| human_review | False | 사람 검토를 한 결과가 아니다 |
| access_control_tests | False | 권한 테스트를 한 결과가 아니다 |
| ready_for_agent_review | False | 6개 모두 True 여야 True. 이번 False 는 계산된 4개 검사(모두 True) 때문이 아니라 고정 False 2개 때문이다 |
| scope | 교육용 1차 Gate. Agent의 Tool 권한·안전성·운영 배포 승인 별도 필요 | output |
