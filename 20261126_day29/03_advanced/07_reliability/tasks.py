"""
07 · Tool 실패·권한·반복 상한 · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

잘못된 인수를 DB 조회 전에 막고, 발생한 예외를 사용자에게 보여 줄 오류 코드로 바꾸는 함수
"""

# ============================================================
# 잘못된 인수 차단 : validate_report_id()
# ============================================================
# validate_report_id : invalid args는 DB 쿼리 전에 차단합니다.
# - '63'  : 63
# - -1    : ValueError -> main.py 에서 classify_error() 로 'INVALID_ARGS' 변환
def validate_report_id(value) -> int:
    # 해설 D29-07-A-01: 입력 오류, 권한 정책 위반, DB/모델 장애는 서로 다른 오류 경계입니다.
    # 검수 기준: 호출 제한과 최소 감사로그로 비정상 반복·기밀값 노출을 차단합니다.
    # 제공 모듈 coursekit/contracts.py : 양의 정수 report_id 만 통과시키는 report_id_from()
    from coursekit.contracts import report_id_from
    return report_id_from(value)

# ============================================================
# 오류 분류 : classify_error()
# ============================================================
# classify_error : 사용자 입력 오류와 연결 오류를 구분. 비밀정보·SQL 원문 로그 노출 금지.
# - ValueError (입력 검사 실패)        : {'status': 'error', 'error_code': 'INVALID_ARGS'}
# - 그 밖의 예외 (DB·GPU·LLM 장애 등) : {'status': 'error', 'error_code': 'TOOL_FAILURE'}
# 예: classify_error(ValueError('report_id는 1 이상이어야 합니다.')) -> {'status': 'error', 'error_code': 'INVALID_ARGS'}
# 주의: str(exc) 를 결과에 넣지 않으므로 예외 메시지 속 접속 정보가 밖으로 나가지 않음
def classify_error(exc: Exception) -> dict:
    # 해설 D29-07-A-02: 입력 오류, 권한 정책 위반, DB/모델 장애는 서로 다른 오류 경계입니다.
    # 검수 기준: 호출 제한과 최소 감사로그로 비정상 반복·기밀값 노출을 차단합니다.
    return {
        'status': 'error',
        'error_code': 'INVALID_ARGS' if isinstance(exc, ValueError) else 'TOOL_FAILURE',
    }

# ============================================================
# 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_07 import tool_budget_check, audit_log
