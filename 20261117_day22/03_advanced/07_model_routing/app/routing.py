"""Day22 · 명시적 모델 선택 | PPT 53~59 | 03_advanced
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 07 심화 app/routing.py: 명시적 mode 선택·공통 비교 필드·잘못된 mode 차단에 더해
quality 미설정의 자동 대체 금지(check_unconfigured_quality)와 사람 근거 None 유지(check_evidence_view)를 검사한다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
# 제공 모듈 coursekit/config.py: mode -> 모델 ID 선택기(checkpoint_choose). quality 미설정이면 SetupError
from coursekit.config import choose_model as checkpoint_choose

# ============================================================
# 2. mode -> 모델 선택
# ============================================================
def route(mode, models):
    # 명시한 mode로 모델 ID와 선택 이유를 고른다.
    # - "fast"                  : ("gpt-5.4-mini-2026-03-17", "explicit_mode:fast")
    # - "quality" (설정됨)      : ("gpt-5.4-2026-03-05", "explicit_mode:quality")
    # - "quality" (미설정 None) : SetupError -> not_run (fast로 자동 대체하지 않음)
    # - 그 외 mode              : ValueError -> 전송 0회
    
    # TODO D22-07-A01: fast·quality만허용하고실제모델과선택이유를반환하세요.
    # 입력·완료 기준: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    # BEGIN D22-07-A01
    # 해설: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    if mode not in {"fast", "quality"}:
        raise ValueError("지원하지 않는 실행 모드")
    return checkpoint_choose(mode, models)
    # END D22-07-A01

# ============================================================
# 3. 모델 공통 비교 필드
# ============================================================
def record_view(result):
    # 실행 결과에서 비교에 쓸 10개 필드만 같은 이름으로 뽑는다.
    # - 반환 dict 키 : case_id·source_mode·model·route_reason·run_status·answer_object·evidence_ok·usage·request_ms·estimated_usd
    # - 없는 키는 None (예: not_run이면 answer_object=None)
    
    # TODO D22-07-A02: 어떤모델이든동일한결과필드로비교하세요.
    # 입력·완료 기준: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # BEGIN D22-07-A02
    # 해설: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # 주의: "더 좋은 모델" 같은 판정 필드는 넣지 않는다. evidence_ok는 사람 검토 전이면 None
    return {key: result.get(key) for key in ("case_id", "source_mode", "model", "route_reason", "run_status",
            "answer_object", "evidence_ok", "usage", "request_ms", "estimated_usd")}
    # END D22-07-A02

# ============================================================
# 4. 명시적 비교 모드 목록
# ============================================================
def comparison_modes():
    # --action compare일 때만 사용할 mode 목록을 돌려준다.
    # - 반환값 : ["fast", "quality"] -> 같은 질문 H01을 각 한 번씩 실행
    # TODO D22-07-A03: 명시적비교에서만fast·quality를한번씩선택하세요.
    # 입력·완료 기준: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    # BEGIN D22-07-A03
    # 해설: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    return ["fast", "quality"]
    # END D22-07-A03

# ============================================================
# 5. 잘못된 mode 사전 차단 검사
# ============================================================
def reject_unknown_mode():
    # route("unknown", ...)이 전송 전에 ValueError를 내는지 확인한다.
    # - ValueError 발생 : True 반환 (정상)
    # - 오류 없이 통과  : AssertionError("잘못된모드를허용했습니다")
    # - 주의: models 인자 {"fast": "example"}는 검사용 가짜 값이며 실제 호출은 없다
    # TODO D22-07-A04: 잘못된mode가호출전에오류인지모의검사하세요.
    # 입력·완료 기준: 인증오류를다른모델이나계정으로우회하지않습니다.
    # BEGIN D22-07-A04
    # 해설: 인증오류를다른모델이나계정으로우회하지않습니다.
    try:
        route("unknown", {"fast": "example"})
    except ValueError:
        return True
    raise AssertionError("잘못된모드를허용했습니다")
    # END D22-07-A04

# ============================================================
# 6. quality 미설정 자동 대체 금지 검사
# ============================================================
def check_unconfigured_quality():
    # quality 모델이 None일 때 route가 fast로 바꾸지 않고 SetupError를 내는지 확인한다.
    # - models 인자 {"fast": "example", "quality": None} : 검사용 가짜 설정 (실제 호출 없음)
    # - SetupError 발생 : True
    # - 통과            : AssertionError("quality미설정을통과했습니다")
    # TODO D22-07-A05: quality가미설정이면다른모델로자동대체하지않는지확인하세요.
    # 입력·완료 기준: 선택정책검사는simulated이고두모델실제비교와구분합니다.
    # BEGIN D22-07-A05
    # 해설: 선택정책검사는simulated이고두모델실제비교와구분합니다.
    from coursekit.config import SetupError
    try:
        route("quality", {"fast": "example", "quality": None})
    except SetupError:
        return True
    raise AssertionError("quality미설정을통과했습니다")
    # END D22-07-A05

# ============================================================
# 7. 사람 근거 None 유지 검사
# ============================================================
def check_evidence_view():
    # 자동 검증 통과(validated) 결과라도 record_view가 evidence_ok를 None 그대로 두는지 확인한다.
    # - 입력 : {"run_status": "validated", "evidence_ok": None}
    # - 반환값 True
    # TODO D22-07-A06: 자동검증결과를사람의근거승인으로만들지않는지확인하세요.
    # 입력·완료 기준: 공통결과구조의None을임의True로바꾸지않습니다.
    # BEGIN D22-07-A06
    # 해설: 공통결과구조의None을임의True로바꾸지않습니다.
    row = record_view({"run_status": "validated", "evidence_ok": None})
    assert row["evidence_ok"] is None
    return True
    # END D22-07-A06
