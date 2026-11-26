"""
05 · PostgreSQL Read-only lookup_status · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

보고서 ID 를 검사하고, 실제 DB 조회 결과(fetch_report_status)에서 사용자에게 보여 줄 상태 필드만 골라내는 함수
"""

# ============================================================
# 보고서 ID 해석 : parse_report_id()
# ============================================================
# parse_report_id : task_id가 아니라 reports.report_id. 임의로 문자열을 SQL에 붙이지 않습니다.
# - '63', 63          : 63
# - '063', '6.3', -1 : ValueError
# 주의: 검사한 정수는 coursekit/db.py 에서 :rid 바인딩 파라미터로 전달되며, SQL 문자열 연결에 쓰이지 않는다
def parse_report_id(value) -> int:
    # 해설 D29-05-A-01: 보고서 상태는 pending/ready/failed이고 제출 여부가 아닙니다.
    # 검수 기준: ready 문자열뿐 아니라 Model·Revision·Overview Hash가 일치하는지 검증합니다.
    # 제공 모듈 coursekit/contracts.py : 양의 정수 report_id 만 통과시키는 report_id_from()
    from coursekit.contracts import report_id_from
    return report_id_from(value)

# ============================================================
# 상태 표시용 dict 만들기 : status_view()
# ============================================================
# status_view : pending/ready/failed와 실제 벡터 정합성을 구분해 표시.
# - raw['status'] 가 'ok' 아님 (보고서 없음) : {'status': 'not_found', 'report_id': ...}
# - raw['status'] == 'ok'                   : 아래 4개 키만 남긴 dict
#   - report_id    : 보고서 번호
#   - index_status : reports.index_status 문자열 (pending / ready / failed)
#   - revision     : reports.revision (보고서 개정 번호)
#   - vector_fresh : 저장 벡터가 현재 revision·모델·overview 해시와 모두 맞는지 (True/False)
# 예: {'status': 'ok', 'report_id': 63, 'index_status': 'pending', 'revision': 1, 'vector_fresh': False, ...}
#     -> {'report_id': 63, 'index_status': 'pending', 'revision': 1, 'vector_fresh': False}
def status_view(raw: dict) -> dict:
    # 해설 D29-05-A-02: 보고서 상태는 pending/ready/failed이고 제출 여부가 아닙니다.
    # 검수 기준: ready 문자열뿐 아니라 Model·Revision·Overview Hash가 일치하는지 검증합니다.
    if raw['status'] != 'ok':
        return {'status': 'not_found', 'report_id': raw['report_id']}
    # 주의: index_status 와 vector_fresh 를 함께 보여 줘야 'ready 인데 벡터가 낡은' 경우를 구분할 수 있음
    return {
        'report_id': raw['report_id'],
        'index_status': raw['index_status'],
        'revision': raw['revision'],
        'vector_fresh': raw['vector_fresh'],
    }

# ============================================================
# 검색에 쓸 수 있는 상태인지 판정 : status_is_usable() 
# ============================================================
# status_is_usable : ready 문자열 하나가 아닌 vector_fresh까지 확인합니다.
# 세 조건이 모두 참일 때만 True
# - status == 'ok'            : 보고서가 DB 에 있음
# - index_status == 'ready'   : 인덱싱 작업이 끝났다고 표시됨
# - vector_fresh is True      : 저장 벡터의 revision·모델·overview 해시가 현재 보고서와 일치
# 예: {'status': 'ok', 'index_status': 'ready', 'vector_fresh': False} -> False (ready 표시만 있고 벡터가 낡음)
# 주의: 'is True' 비교라 vector_fresh 키가 없거나 None 이면 False
def status_is_usable(raw: dict) -> bool:
    # 해설 D29-05-A-03: 보고서 상태는 pending/ready/failed이고 제출 여부가 아닙니다.
    # 검수 기준: ready 문자열뿐 아니라 Model·Revision·Overview Hash가 일치하는지 검증합니다.
    return raw.get('status') == 'ok' and raw.get('index_status') == 'ready' and raw.get('vector_fresh') is True

# ============================================================
# 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_05 import enforce_read_policy
