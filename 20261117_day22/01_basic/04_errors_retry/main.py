"""Day22 · 오류와 제한 재시도 | PPT 32~38 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 04 기본필수: 인증·권한·할당량·속도 제한·서버 오류를 분류하고,
생성 2회 × 생성별 전송 2회 · 전체 전송 4회 예산 안에서만 재시도하는지 모의 사례 11개로 검사한다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee 폴더
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))

# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·오류 안내 래퍼
from coursekit.support import parse_args, guarded
# 제공 모듈 coursekit/errors.py: 기본 오류 분류기(checkpoint_classify)와 기본 대기 계산기(checkpoint_delay)
from coursekit.errors import classify_error as checkpoint_classify, retry_delay as checkpoint_delay
# 제공 모듈 coursekit/runner.py: 재시도 예산 dataclass(RetryPolicy, 2×2·총4회·60초 상한 검사 포함)
from coursekit.runner import RetryPolicy
# 제공 모듈 coursekit/labs.py: 모의 실패 사례 11개를 실행하고 scenario_checks.json을 쓰는 도우미
from coursekit.labs import failure_scenarios

# ============================================================
# 2. 오류 분류
# ============================================================
def classify(error):
    # 정규화된 공급자 오류(ProviderError)를 종류(kind)와 재시도 가능 여부(retryable)로 분류한다.
    # - 401                                   : auth_error, 재시도 안 함 (키 오류)
    # - 403                                   : permission_error, 재시도 안 함 (모델·조직 권한)
    # - 429 + insufficient_quota 등 과금 코드  : quota_error, 재시도 안 함 (잔액·한도 부족)
    # - 429 + rate_limit_exceeded 등           : rate_limit, 재시도 가능 (일시적 속도 제한)
    # - 그 외 (timeout·404·400·500·알 수 없는 429 등) : 제공 분류기 checkpoint_classify에 맡김
    #   예: 429 + "other" -> unknown_429 (재시도 안 함), 500 -> server_error (재시도 가능)

    # TODO D22-04-B01: 401·403은 중단하고, 429의 quota와 일시적 제한을 구분하세요.
    # 입력·완료 기준: 알 수 없는429는 무조건 재시도하지 않습니다. 실제로 제한을 유발하는 호출은 금지합니다.
    # BEGIN D22-04-B01
    # 해설: 알 수 없는429는 무조건 재시도하지 않습니다. 실제로 제한을 유발하는 호출은 금지합니다.
    if error.status == 401:
        return {"kind": "auth_error", "retryable": False}
    if error.status == 403:
        return {"kind": "permission_error", "retryable": False}
    # codes : 오류 본문의 code와 type을 한 집합으로 모아 비교 (둘 중 어디에 와도 인식)
    codes = {error.code, error.error_type}
    # & : 집합 교집합. 하나라도 겹치면 해당 분류
    if error.status == 429 and codes & {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "usage_limit_reached"}:
        return {"kind": "quota_error", "retryable": False}
    if error.status == 429 and codes & {"rate_limit_exceeded", "rate_limit_error", "requests", "tokens"}:
        return {"kind": "rate_limit", "retryable": True}
    return checkpoint_classify(error)
    # END D22-04-B01

# ============================================================
# 3. 재시도 예산 설정
# ============================================================
def make_policy():
    # 한 작업이 쓸 수 있는 재시도 예산(RetryPolicy)을 만든다.
    # - 반환값 : RetryPolicy(생성 2회, 생성별 전송 2회, 전체 전송 4회, 경과 시간 60초)

    # TODO D22-04-B02: 생성 최대2·생성별 전송최대2·전체 전송최대4를 설정하세요.
    # 입력·완료 기준: SDK max_retries=0은 제공 코드에 설정되어 있습니다. 무한 fallback을 추가하지 않습니다.
    # BEGIN D22-04-B02
    # 해설: SDK max_retries=0은 제공 코드에 설정되어 있습니다. 무한 fallback을 추가하지 않습니다.
    # RetryPolicy 인자
    # - max_generations=2          : 형식 수정(JSON·Schema 오류 뒤 다시 생성)을 포함한 생성 횟수 상한
    # - max_sends_per_generation=2 : 한 생성 안에서 통신 오류(429·5xx) 재전송 포함 전송 상한
    # - max_sends_total=4          : 두 종류 재시도가 함께 쓰는 전체 전송 상한 (2×2를 넘지 않음)
    # - elapsed_budget_s=60.0      : 작업 전체 경과 시간 예산(초)
    # 주의: RetryPolicy는 이 범위를 넘는 값(예: max_sends_total=5)을 ValueError로 거부한다
    return RetryPolicy(max_generations = 2, max_sends_per_generation = 2, max_sends_total = 4, elapsed_budget_s = 60.0)
    # END D22-04-B02

# ============================================================
# 4. 재시도 대기 시간
# ============================================================
def delay(attempt, retry_after_s, remaining_s, jitter=0.0):
    # 다음 재전송 전 대기 시간(초)을 제공 대기 함수로 계산해 그대로 돌려준다.
    # - attempt       : 이번 생성 안의 전송 번호 (1부터)
    # - retry_after_s : 서버 Retry-After 헤더의 초 값 (없으면 None)
    # - remaining_s   : 남은 작업 예산(초)
    # - jitter        : 무작위 추가 대기 (simulated는 0.0)
    # - 반환값        : max(Retry-After, 지수 대기 1·2·4·8초) + jitter. 남은 예산 이상이면 None (-> deferred 종료)
    #   예: attempt=1, retry_after_s=2.0, remaining_s=59.9 -> 2.0초
    #   예: attempt=1, retry_after_s=120.0, remaining_s=59.9 -> None

    # TODO D22-04-B03: 제공된 대기 함수에 남은 예산을 전달하고 반환값을 유지하세요.
    # 입력·완료 기준: 서버의 최소 대기를 짧게 줄이지 않습니다.
    # BEGIN D22-04-B03
    # 해설: 서버의 최소 대기를 짧게 줄이지 않습니다.
    # checkpoint_delay 인자 : 받은 네 값을 순서대로 그대로 전달 (Retry-After를 줄이거나 None을 0으로 바꾸지 않음)
    wait = checkpoint_delay(attempt, retry_after_s, remaining_s, jitter)
    return wait
    # END D22-04-B03

# 강사용 핵심 해설
# 같은 429라도 일시적 속도 제한과 할당량 부족은 조치가 다릅니다. 원인을 모르면 자동 반복하지 않습니다.
# 모델 출력 수정 2회 × 통신 전송 2회가 총4회를 넘지 않도록 한 예산을 공유합니다.
# timeout은 서버에서 처리하지 않았다는 증거가 아닙니다. 이 수업은 중복 비용을 줄이기 위해 자동 재전송하지 않습니다.
# Retry-After가 남은 시간보다 크면 기다림을 줄이는 대신 deferred로 종료합니다.

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # 모의 실패 사례 11개를 위 분류기·대기 함수·예산으로 실행하고 기대값과 대조한다.
    # - outputs/runs/<run_id>/run.json : 사례별 실행 기록 (case_id = auth, quota, ..., shared_four)
    # - outputs/scenario_checks.json   : 사례별 actual·expected·passed
    # - 하나라도 불일치 : ValueError -> "실행 전 확인: 실패 정책의 예상 결과와 다릅니다" (종료코드 2)
    args = parse_args(HERE)
    # failure_scenarios 인자
    # - classifier=classify     : 위 오류 분류 함수
    # - delay=delay             : 위 대기 함수
    # - policy=make_policy()    : 2×2·총4회 예산
    # 주의: 내부에서 load_settings("simulated")와 SimClient만 쓰므로 실제 전송 0회
    results = failure_scenarios(args, classifier = classify, delay = delay, policy = make_policy())

# ============================================================
# 6. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    guarded(main)
