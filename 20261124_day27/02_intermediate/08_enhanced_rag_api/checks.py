"""
단계별 역할 분리: checks.py / 독립 실행은 main.py.
"""

# ============================================================
# HTTP 응답·출처 ID 검사 : check_http_response() 
# ============================================================
# check_http_response : HTTP 응답에서 성공 여부와 출처 ID 일치를 검사
# - 200 이 아님 : {'http_ok': False, 'error_detail': 응답 detail}
# - 200        : {'http_ok': True, 'citation_consistent': sources 의 ID 목록 == result.source_ids}
# 주의: ID 가 일치해도 답변 내용이 근거에 맞는지(Grounding)는 별도 검사
def check_http_response(response):
    # BEGIN D27-08-I03
    # TODO D27-08-I03 완료: 성공 응답의 source_ids와 실제 연결된 sources의 ID가 같은지 검사합니다.
    # 1. 입력: response (필요한 값만 받습니다.)
    # 2. 절차: 성공 응답의 source_ids와 실제 연결된 sources의 ID가 같은지 검사합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 응답 본문 JSON -> dict
    body = response.json()
    # 422·502 등 실패 응답은 오류 내용만 기록 (정보 부족 응답으로 바꾸지 않음)
    if response.status_code != 200:
        return {'http_ok': False, 'error_detail': body.get('detail')}
    # sources 목록의 citation_id 를 순서대로 모아 result.source_ids 와 비교
    # 예: sources [{citation_id: 1002, ...}] , source_ids [1002] -> True
    ids = [src['citation_id'] for src in body['sources']]
    return {'http_ok': True, 'citation_consistent': ids == body['result']['source_ids']}
    # END D27-08-I03
