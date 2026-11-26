"""
경계값·오류/권한 정책: Advanced · Day29.

서버가 허용한 Category 범위 밖의 보고서는 '없음 또는 금지'로만 답하는 앱 수준 읽기 정책
"""

# ============================================================
# Category 읽기 정책 : enforce_read_policy() 
# ============================================================
# enforce_read_policy : 앱 Category 정책. 실제 사용자 인증/소유권 ACL로 표현하지 마세요.
# - raw['status'] 가 'ok' 아님 (보고서 없음)      : {'status': 'not_found'}
# - category_code 가 allowed 에 없음              : {'status': 'not_found_or_forbidden'}
#   (없는 보고서인지 금지된 보고서인지 밖에서 구분할 수 없게 같은 계열 응답 사용)
# - 허용 범위 안 : {report_id, index_status, revision, vector_fresh}
# 예: category_code 'RESEARCH_PLAN', allowed {'MEETING'} -> {'status': 'not_found_or_forbidden'}
def enforce_read_policy(raw: dict, allowed: set[str]) -> dict:
    # 해설 D29-05-A-04: 보고서 상태는 pending/ready/failed이고 제출 여부가 아닙니다.
    # 검수 기준: ready 문자열뿐 아니라 Model·Revision·Overview Hash가 일치하는지 검증합니다.
    if raw.get('status') != 'ok':
        return {'status': 'not_found'}
    if raw['category_code'] not in allowed:
        return {'status': 'not_found_or_forbidden'}
    # 주의: 허용된 경우에도 category_code·embedding_model 등은 응답에서 제외하고 4개 키만 반환
    return {
        'report_id': raw['report_id'],
        'index_status': raw['index_status'],
        'revision': raw['revision'],
        'vector_fresh': raw['vector_fresh'],
    }
