"""
경계값·오류/권한 정책: Advanced · Day29.

Tool 호출 요청과 결과 메시지가 같은 호출 ID·Tool 이름으로 짝지어졌는지, 여러 호출에서 누락·중복이 없는지 검사
"""

# ============================================================
# 1. 요청·결과 한 쌍 검사 : validate_message_pair() 
# ============================================================
# validate_message_pair : 두 메시지에 같은 호출 ID·Tool 이름이 들어 있는지 확인.
# - call['id'] == message['tool_call_id'] 이고 call['name'] == message['name'] : True
# - 둘 중 하나라도 다름                                                     : False
# 예: call {'id': 'call_001', 'name': 'lookup_status'} + message {'tool_call_id': 'call_001', 'name': 'lookup_status'} -> True
def validate_message_pair(call: dict, message: dict) -> bool:
    # 해설 D29-02-A-03: Tool Call은 요청 dict이고 실제 DB 조회가 끝난 뒤 결과 메시지가 생성됩니다.
    # 검수 기준: 같은 ID로 요청과 결과를 연결하고, 다중 호출에는 중복·누락 검사를 적용합니다.
    return call['id'] == message['tool_call_id'] and call['name'] == message['name']

# ============================================================
# 2. 여러 호출의 짝 검사 : summarize_pairs() 
# ============================================================
# summarize_pairs : 복수 Tool Call의 ID 매칭 누락/중복을 보고합니다.
# - 요청 ID 또는 결과 ID 에 중복이 있음 : ValueError('중복된 tool_call_id')
# - 정상 : 아래 3개 키 dict
#   - requested : 요청(Tool Call) 개수
#   - returned  : 결과(ToolMessage) 개수
#   - unmatched : 요청은 있는데 결과가 없는 호출 ID 목록 (정렬)
# 예: calls [call_001, call_002], messages [call_001] -> {'requested': 2, 'returned': 1, 'unmatched': ['call_002']}
def summarize_pairs(calls: list[dict], messages: list[dict]) -> dict:
    # 해설 D29-02-A-04: Tool Call은 요청 dict이고 실제 DB 조회가 끝난 뒤 결과 메시지가 생성됩니다.
    # 검수 기준: 같은 ID로 요청과 결과를 연결하고, 다중 호출에는 중복·누락 검사를 적용합니다.
    # ids  : 요청 쪽 호출 ID 목록, done : 결과 쪽 tool_call_id 목록
    ids = [c['id'] for c in calls]
    done = [m['tool_call_id'] for m in messages]
    # 목록 길이와 집합 크기가 다르면 같은 ID 가 두 번 이상 있다는 뜻
    if len(ids) != len(set(ids)) or len(done) != len(set(done)):
        raise ValueError('중복된 tool_call_id')
    # unmatched : 집합 차이 (요청 ID - 결과 ID)
    # 주의: 결과에만 있고 요청에 없는 ID 는 이 집합 차이에 나타나지 않는다
    return {
        'requested': len(ids),
        'returned': len(done),
        'unmatched': sorted(set(ids) - set(done)),
    }
