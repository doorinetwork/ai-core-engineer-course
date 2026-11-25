"""
Day28 실습 06 · 03_advanced · 함수 역할별 구현.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.
"""


# ============================================================
# Runner 감사 기록 : audit_runner() 
# ============================================================
# audit_runner : Runner 실행 감사 기록 (비밀값·Prompt 원문 없이 재현에 필요한 값만)
# - 반환 dict
#   - question_id          : 질문 ID
#   - dataset_version      : 데이터셋 버전
#   - run_mode             : 'offline_fixture' 또는 'live'
#   - variant              : 실험 변형 (예: 'both')
#   - source_ids           : Context 에 포함된 출처 ID 목록
#   - secrets_logged       : 비밀값 기록 여부 (항상 False)
#   - trace_fields_checked : Trace 필드 검사 여부 (항상 True)
def audit_runner(request: dict, run: dict, version: str) -> dict:
    # 정답 D28-06-A04: 실행 모드·질문 ID·버전·출처 ID를 기록하고 비밀값/Prompt 원문을 로그로 남기지 마세요.
    # 절차: LangSmith 연동은 선택적·명시적 Opt-in이어야 합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # run.get('included_ids', []) : 필드가 없을 때 KeyError 대신 빈 목록
    return {'question_id': request['question_id'], 'dataset_version': version,
            'run_mode': run['mode'], 'variant': run['variant'],
            'source_ids': list(run.get('included_ids', [])),
            'secrets_logged': False, 'trace_fields_checked': True}
