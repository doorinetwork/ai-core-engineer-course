"""
경계값·오류/권한 정책: Advanced · Day29.

Tool 호출 횟수 상한을 검사하고, 감사 로그에 남길 이벤트에서 기밀값·질문 원문을 제거
"""

# ============================================================
# 호출 상한 검사 : tool_budget_check() 
# ============================================================
# tool_budget_check : 최대 호출 수를 넘기면 False(실패). 무한 반복 금지.
# - maximum < 1 또는 calls < 0 : ValueError (상한 0·음수 호출 수는 잘못된 카운터)
# - calls <= maximum          : True
# - calls >  maximum          : False
# 예: tool_budget_check(5, 4) -> False / tool_budget_check(4, 4) -> True
def tool_budget_check(calls: int, maximum: int) -> bool:
    # 해설 D29-07-A-03: 입력 오류, 권한 정책 위반, DB/모델 장애는 서로 다른 오류 경계입니다.
    # 검수 기준: 호출 제한과 최소 감사로그로 비정상 반복·기밀값 노출을 차단합니다.
    if maximum < 1 or calls < 0:
        raise ValueError('invalid limit counters')
    return calls <= maximum

# ============================================================
# 감사 로그 마스킹 : audit_log() 
# ============================================================
# audit_log : 기밀 값·질문 원문·API Key를 제거한 감사 메타데이터만 반환.
# 남기는 키 (coursekit/contracts.py sanitize_event() 의 허용 목록)
# - event · stage · status · tool_name · report_id · elapsed_ms · error_code · tool_call_id
# 예: {'event': 'tool_call', 'report_id': 63, 'api_key': 'SHOULD_NOT_LOG', 'question': 'secret'}
#     -> {'event': 'tool_call', 'report_id': 63}
# 주의: 허용 목록 방식이라 새 키가 추가돼도 기본적으로 기록되지 않음 (차단 목록 방식보다 안전)
def audit_log(event: dict) -> dict:
    # 해설 D29-07-A-04: 입력 오류, 권한 정책 위반, DB/모델 장애는 서로 다른 오류 경계입니다.
    # 검수 기준: 호출 제한과 최소 감사로그로 비정상 반복·기밀값 노출을 차단합니다.
    # 제공 모듈 coursekit/contracts.py : 허용 키만 남기는 sanitize_event()
    from coursekit.contracts import sanitize_event
    return sanitize_event(event)
