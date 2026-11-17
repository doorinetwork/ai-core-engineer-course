"""Day22 · 명시적 모델 선택 | PPT 53~59 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 07 기본필수: --mode fast/quality를 호출 전에 명시적으로 골라 실제 모델 ID와 선택 이유를 기록하고,
어떤 모델이든 같은 결과 필드로 model_comparison.json에 남긴다. 알 수 없는 mode는 전송 0회로 중단한다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee 폴더 (config/models.json 위치)
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))

# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·오류 안내 래퍼
from coursekit.support import parse_args, guarded
# 제공 모듈 coursekit/config.py: mode -> 모델 ID 선택기(checkpoint_choose). quality 미설정이면 SetupError
from coursekit.config import choose_model as checkpoint_choose
# 제공 모듈 coursekit/labs.py: mode별 실행·model_comparison.json 저장 도우미(routing_lab)
from coursekit.labs import routing_lab

# ============================================================
# 2. mode -> 모델 선택
# ============================================================
def route(mode, models):
    # 명시한 mode로 모델 ID와 선택 이유를 고른다.
    # - mode   : "fast" 또는 "quality"
    # - models : {"fast": 모델 ID, "quality": 모델 ID 또는 None}
    # - "fast"                    : ("gpt-5.4-mini-2026-03-17", "explicit_mode:fast")
    # - "quality" (설정됨)        : ("gpt-5.4-2026-03-05", "explicit_mode:quality")
    # - "quality" (미설정 None)   : SetupError -> not_run (fast로 자동 대체하지 않음)
    # - 그 외 mode (예: "cheap")  : ValueError -> 전송 0회

    # TODO D22-07-B01: fast·quality만허용하고실제모델과선택이유를반환하세요.
    # 입력·완료 기준: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    # BEGIN D22-07-B01
    # 해설: 알수없는모드를fast로조용히바꾸지않습니다. quality권한없으면not_run입니다.
    if mode not in {"fast", "quality"}:
        raise ValueError("지원하지 않는 실행 모드")
    return checkpoint_choose(mode, models)
    # END D22-07-B01

# ============================================================
# 3. 모델 공통 비교 필드
# ============================================================
def record_view(result):
    # 실행 결과에서 비교에 쓸 10개 필드만 같은 이름으로 뽑는다.
    # - 반환 dict 키 : case_id·source_mode·model·route_reason·run_status·answer_object·evidence_ok·usage·request_ms·estimated_usd
    # - 없는 키는 None (예: not_run이면 answer_object=None)

    # TODO D22-07-B02: 어떤모델이든동일한결과필드로비교하세요.
    # 입력·완료 기준: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # BEGIN D22-07-B02
    # 해설: 고가모델이항상더좋다는결론을미리넣지않습니다.
    # 비교 필드 : 의미
    # - route_reason  : 선택 이유 (예: "explicit_mode:fast")
    # - run_status    : validated·schema_error·not_run 등
    # - evidence_ok   : 사람 근거 검토 결과. 검토 전에는 None 그대로
    # - estimated_usd : live 실측 추정 비용. simulated·미실행이면 None
    # 주의: "더 좋은 모델" 같은 판정 필드는 넣지 않는다
    return {key: result.get(key) for key in ("case_id", "source_mode", "model", "route_reason", "run_status",
            "answer_object", "evidence_ok", "usage", "request_ms", "estimated_usd")}
    # END D22-07-B02

# ============================================================
# 4. 명시적 비교 모드 목록
# ============================================================
def comparison_modes():
    # --action compare일 때만 사용할 mode 목록을 돌려준다.
    # - 반환값 : ["fast", "quality"] -> 각 한 번씩 같은 질문 H01로 실행
    # - 기본 --action run : 이 함수를 쓰지 않고 --mode 하나만 실행
    # TODO D22-07-B03: 명시적비교에서만fast·quality를한번씩선택하세요.
    # 입력·완료 기준: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    # BEGIN D22-07-B03
    # 해설: 기본run은한모델만호출합니다. compare는계정권한·예산확인후선택합니다.
    return ["fast", "quality"]
    # END D22-07-B03

# 강사용 핵심 해설
# fast/quality는 수업 설정 이름입니다. quality가 항상 정확하다는 뜻이 아닙니다.
# 모델 선택은 호출 전에 명시적으로 하며, 권한 오류를 숨기기 위해 다른 모델을 자동 호출하지 않습니다.
# 비교할 때는 질문·문맥·출력 계약을 같게 하고 검증 및 사용량 기록 형식을 유지합니다.

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # --mode 또는 --action compare에 따라 모델을 골라 H01을 실행하고 비교표를 저장한다.
    # - outputs/runs/<run_id>/run.json  : mode별 실행 기록
    # - outputs/model_comparison.json   : record_view 행 목록 (run 1행, compare 2행)
    # - 인증·할당량·권한·예산 소진 : 다음 모델을 호출하지 않고 중단
    args = parse_args(HERE)
    # routing_lab 인자
    # - router=route                       : 위 모델 선택 함수
    # - record_view=record_view            : 위 공통 필드 함수
    # - compare_modes=comparison_modes     : compare 때 쓸 mode 목록
    routing_lab(args, router = route, record_view = record_view, compare_modes = comparison_modes)

# ============================================================
# 6. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    guarded(main)
