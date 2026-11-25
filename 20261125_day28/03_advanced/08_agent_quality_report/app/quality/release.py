"""
Day28 실습 08 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Agent 1차 준비 Gate 함수
"""

# ============================================================
# Agent 1차 준비 Gate : agent_release_gate() 
# ============================================================
# agent_release_gate : Agent Tool 로 쓰기 전 1차 준비 Gate (지표 + 사람 검토 + 접근제어)
# - summary   : summarize_all() 결과
# - human_ok  : 사람 검토 통과 여부 (실행 골격은 False 전달)
# - access_ok : 접근제어 테스트 통과 여부 (실행 골격은 False 전달)
# - 반환      : {'ready_for_agent_review': 모든 검사 통과 여부, 'checks': 항목별 결과, 'scope': 적용 범위}
# 예: Recall 1.0·MRR 1.0 이어도 human_ok=False 이면 ready_for_agent_review False
def agent_release_gate(summary: dict, human_ok: bool, access_ok: bool) -> dict:
    # 정답 D28-08-A04: Recall·MRR·출처·실행 상태와 사람 검토·접근제어를 함께 검사하는 1차 Agent 준비 Gate를 반환하세요.
    # 절차: 1차 Gate는 실서비스 배포 전 모든 보안·운영 점검을 대체하지 않습니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 6개 검사(Recall·MRR·출처·실행상태·사람검토·접근제어)
    from coursekit.eval_report import release_gate
    # release_gate() 인자
    # {**summary, ...} : summary 를 복사하고 사람 검토·접근제어 결과 키를 추가
    # min_recall=.70   : 평균 Recall@3 최소 기준
    # min_mrr=.60      : MRR@3 최소 기준
    return release_gate({**summary, 'human_review_passed': bool(human_ok),
                         'access_control_passed': bool(access_ok)},
                        min_recall = .70, min_mrr = .60)
