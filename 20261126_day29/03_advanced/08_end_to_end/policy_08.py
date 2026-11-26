"""
경계값·오류/권한 정책: Advanced · Day29.

Agent 실행 결과가 완료됐고 인용 출처가 검증됐을 때만 '검증됨' 기록을 남기는 Evidence Gate
"""

# ============================================================
# 응답 감사 게이트 : audit_response()
# ============================================================
# audit_response : 검증된 출처만 기록하고 성공으로 위장된 오류는 거절합니다.
# - raw['status'] != 'completed'            : ValueError('agent incomplete')
# - raw['citation_ids_valid'] 가 True 아님 : ValueError('invalid citations')
# - 통과 : {'status': 'verified', 'tools': len(tool_trace)}
# 예: tool_trace 2건·citation_ids_valid True -> {'status': 'verified', 'tools': 2}
# 주의: api_response() 는 RuntimeError, 이 함수는 ValueError 를 낸다 (예외 종류가 다름)
def audit_response(raw: dict) -> dict:
    # 해설 D29-08-A-04: FastAPI는 사용자의 입력을 검증해 Agent에 전달하고 최종 결과를 반환합니다.
    # 검수 기준: 인용 ID가 실제 Tool 근거 집합 밖이면 성공 응답으로 위장하지 않습니다.
    if raw.get('status') != 'completed':
        raise ValueError('agent incomplete')
    if raw.get('citation_ids_valid') is not True:
        raise ValueError('invalid citations')
    # tools : 실제 실행된 ToolMessage 기록 개수 (tool_trace 가 없으면 0)
    return {'status': 'verified', 'tools': len(raw.get('tool_trace', []))}
