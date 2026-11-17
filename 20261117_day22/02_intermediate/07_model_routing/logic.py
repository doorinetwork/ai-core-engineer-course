"""Day22 · 명시적 모델 선택 | PPT 53~59 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 07 응용 TODO 모듈: fast/quality 명시 선택·공통 비교 필드·compare 모드 목록에 더해
잘못된 mode가 호출 전에 ValueError로 막히는지 전송 0회로 확인한다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단
from coursekit.support import parse_args
# 제공 모듈 coursekit/config.py: mode -> 모델 ID 선택기(checkpoint_choose). quality 미설정이면 SetupError
from coursekit.config import choose_model as checkpoint_choose
# 제공 모듈 coursekit/labs.py: mode별 실행·model_comparison.json 저장 도우미(routing_lab)
from coursekit.labs import routing_lab

# ============================================================
# 2. mode -> 모델 선택
# ============================================================
def route(mode, models):
    # 명시한 mode로 모델 ID와 선택 이유를 고른다.
    # - "fast"                  : ("gpt-5.4-mini-2026-03-17", "explicit_mode:fast")
    # - "quality" (설정됨)      : ("gpt-5.4-2026-03-05", "explicit_mode:quality")
    # - "quality" (미설정 None) : SetupError -> not_run (fast로 자동 대체하지 않음)
    # - 그 외 mode              : ValueError -> 전송 0회

    # TODO D22-07-I01: fast·quality만허용하고실제모델과선택이유를반환하세요.
    # 입력·완료 기준: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    # BEGIN D22-07-I01
    # 해설: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    if mode not in {"fast", "quality"}:
        raise ValueError("지원하지 않는 실행 모드")
    return checkpoint_choose(mode, models)
    # END D22-07-I01

# ============================================================
# 3. 모델 공통 비교 필드
# ============================================================
def record_view(result):
    # 실행 결과에서 비교에 쓸 10개 필드만 같은 이름으로 뽑는다.
    # - 반환 dict 키 : case_id·source_mode·model·route_reason·run_status·answer_object·evidence_ok·usage·request_ms·estimated_usd
    # - 없는 키는 None (예: not_run이면 answer_object=None)

    # TODO D22-07-I02: 어떤모델이든동일한결과필드로비교하세요.
    # 입력·완료 기준: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # BEGIN D22-07-I02
    # 해설: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # 주의: "더 좋은 모델" 같은 판정 필드는 넣지 않는다. evidence_ok는 사람 검토 전이면 None
    return {key: result.get(key) for key in ("case_id", "source_mode", "model", "route_reason", "run_status",
            "answer_object", "evidence_ok", "usage", "request_ms", "estimated_usd")}
    # END D22-07-I02

# ============================================================
# 4. 명시적 비교 모드 목록
# ============================================================
def comparison_modes():
    # --action compare일 때만 사용할 mode 목록을 돌려준다.
    # - 반환값 : ["fast", "quality"] -> 같은 질문 H01을 각 한 번씩 실행
    # TODO D22-07-I03: 명시적비교에서만fast·quality를한번씩선택하세요.
    # 입력·완료 기준: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    # BEGIN D22-07-I03
    # 해설: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    return ["fast", "quality"]
    # END D22-07-I03

# ============================================================
# 5. 잘못된 mode 사전 차단 검사
# ============================================================
def reject_unknown_mode():
    # route("unknown", ...)이 전송 전에 ValueError를 내는지 확인한다.
    # - ValueError 발생 : True 반환 (정상)
    # - 오류 없이 통과  : AssertionError("잘못된모드를허용했습니다")
    # TODO D22-07-I04: 잘못된mode가호출전에오류인지모의검사하세요.
    # 입력·완료 기준: 인증오류를다른모델이나계정으로우회하지않습니다.
    # BEGIN D22-07-I04
    # 해설: 인증오류를다른모델이나계정으로우회하지않습니다.
    try:
        route("unknown", {"fast": "example"})
    except ValueError:
        return True
    raise AssertionError("잘못된모드를허용했습니다")
    # END D22-07-I04

# HERE : 이 logic.py가 있는 교시 폴더
HERE = Path(__file__).resolve().parent

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
def main():
    # --mode 또는 --action compare에 따라 H01을 실행해 비교표를 저장한 뒤 잘못된 mode 차단을 검사한다.
    # - outputs/runs/<run_id>/run.json : mode별 실행 기록
    # - outputs/model_comparison.json  : record_view 행 목록 (run 1행, compare 2행)
    args = parse_args(HERE)
    # routing_lab 인자
    # - router=route                   : 위 모델 선택 함수
    # - record_view=record_view        : 위 공통 필드 함수
    # - compare_modes=comparison_modes : compare 때 쓸 mode 목록
    routing_lab(args, router = route, record_view = record_view, compare_modes = comparison_modes)
    reject_unknown_mode()
