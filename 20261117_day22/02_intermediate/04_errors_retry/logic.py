"""Day22 · 오류와 제한 재시도 | PPT 32~38 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 04 응용 TODO 모듈: 오류 분류·2×2 예산에 더해 대기 시간(Retry-After와 지수 대기 중 큰 값)을 직접 계산하고,
남은 예산을 넘는 대기는 None(deferred)으로 끊으며, 시도 번호가 1부터 빠짐없이 4 이하인지 확인한다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단
from coursekit.support import parse_args
# 제공 모듈 coursekit/errors.py: 기본 오류 분류기(checkpoint_classify)
from coursekit.errors import classify_error as checkpoint_classify
# 제공 모듈 coursekit/runner.py: 재시도 예산 dataclass(RetryPolicy)
from coursekit.runner import RetryPolicy
# 제공 모듈 coursekit/labs.py: 모의 실패 사례 11개 실행 도우미(failure_scenarios)
from coursekit.labs import failure_scenarios

# ============================================================
# 2. 오류 분류
# ============================================================
def classify(error):
    # 정규화된 공급자 오류(ProviderError)를 종류(kind)와 재시도 가능 여부(retryable)로 분류한다.
    # - 401                                  : auth_error, 재시도 안 함
    # - 403                                  : permission_error, 재시도 안 함
    # - 429 + insufficient_quota 등 과금 코드 : quota_error, 재시도 안 함
    # - 429 + rate_limit_exceeded 등          : rate_limit, 재시도 가능
    # - 그 외                                 : 제공 분류기 (예: 429 "other" -> unknown_429, 500 -> server_error, timeout -> timeout)

    # TODO D22-04-I01: 401·403은 중단하고, 429의 quota와 일시적 제한을 구분하세요.
    # 입력·완료 기준: 알 수 없는429는 무조건 재시도하지 않습니다. 실제로 제한을 유발하는 호출은 금지합니다.
    # BEGIN D22-04-I01
    # 해설: 알 수 없는429는 무조건 재시도하지 않습니다. 실제로 제한을 유발하는 호출은 금지합니다.
    if error.status == 401:
        return {"kind": "auth_error", "retryable": False}
    if error.status == 403:
        return {"kind": "permission_error", "retryable": False}
    # codes : 오류 본문의 code와 type을 한 집합으로 모아 비교
    codes = {error.code, error.error_type}
    if error.status == 429 and codes & {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "usage_limit_reached"}:
        return {"kind": "quota_error", "retryable": False}
    if error.status == 429 and codes & {"rate_limit_exceeded", "rate_limit_error", "requests", "tokens"}:
        return {"kind": "rate_limit", "retryable": True}
    return checkpoint_classify(error)
    # END D22-04-I01

# ============================================================
# 3. 재시도 예산 설정
# ============================================================
def make_policy():
    # 한 작업이 쓸 수 있는 재시도 예산(RetryPolicy)을 만든다.
    # - 반환값 : RetryPolicy(생성 2회, 생성별 전송 2회, 전체 전송 4회, 경과 시간 60초)

    # TODO D22-04-I02: 생성 최대2·생성별 전송최대2·전체 전송최대4를 설정하세요.
    # 입력·완료 기준: SDK max_retries=0은 제공 코드에 설정되어 있습니다. 무한 fallback을 추가하지 않습니다.
    # BEGIN D22-04-I02
    # 해설: SDK max_retries=0은 제공 코드에 설정되어 있습니다. 무한 fallback을 추가하지 않습니다.
    # RetryPolicy 인자
    # - max_generations=2          : 형식 수정 포함 생성 횟수 상한
    # - max_sends_per_generation=2 : 한 생성 안의 통신 재전송 포함 전송 상한
    # - max_sends_total=4          : 두 종류 재시도가 함께 쓰는 전체 전송 상한
    # - elapsed_budget_s=60.0      : 작업 전체 경과 시간 예산(초)
    return RetryPolicy(max_generations = 2, max_sends_per_generation = 2, max_sends_total = 4, elapsed_budget_s = 60.0)
    # END D22-04-I02

# ============================================================
# 4. 재시도 대기 시간 직접 계산
# ============================================================
def delay(attempt, retry_after_s, remaining_s, jitter=0.0):
    # 다음 재전송 전 대기 시간(초)을 계산한다. 남은 예산 안에 끝나지 않으면 None을 돌려준다.
    # - attempt       : 이번 생성 안의 전송 번호 (1부터)
    # - retry_after_s : 서버 Retry-After 초 값 (없으면 None)
    # - remaining_s   : 남은 작업 예산(초)
    # - jitter        : 무작위 추가 대기 (simulated는 0.0)
    # - 반환값        : max(Retry-After, 지수 대기) + jitter, 또는 None
    #   지수 대기 min(8, 2 ** min(attempt - 1, 3)) : attempt 1 -> 1초, 2 -> 2초, 3 -> 4초, 4 이상 -> 8초
    #   예: attempt=1, Retry-After 2초, 남은 예산 약 60초  -> 2.0초 (서버 최소 대기 2초가 지수 대기 1초보다 큼)
    #   예: attempt=1, Retry-After 120초, 남은 예산 약 60초 -> None (-> run_status "deferred")

    # TODO D22-04-I03: 헤더 최소 대기와 지수 대기 중 큰 값을 계산하세요.
    # 입력·완료 기준: 함수에 전달되는 헤더값은 이미 검증된 초 단위입니다.
    # BEGIN D22-04-I03
    # 해설: 함수에 전달되는 헤더값은 이미 검증된 초 단위입니다.
    # hinted : Retry-After가 없으면 0.0초로 보고 지수 대기만 사용
    hinted = 0.0 if retry_after_s is None else retry_after_s
    # 음수 헤더나 이미 소진된 예산 : 기다리지 않고 중단
    if hinted < 0 or remaining_s <= 0:
        return None
    # 주의: 서버 최소 대기(hinted)보다 짧게 줄이지 않도록 max 사용
    wait = max(hinted, min(8.0, 2 ** min(attempt - 1, 3))) + jitter
    # END D22-04-I03
    # TODO D22-04-I04: 대기시간이 남은 작업 예산 이상이면 None으로 중단하세요.
    # 입력·완료 기준: 대기 후 실행기는 남은 예산과 횟수를 다시 검사합니다.
    # BEGIN D22-04-I04
    # 해설: 대기 후 실행기는 남은 예산과 횟수를 다시 검사합니다.
    # wait >= remaining_s : 기다리면 예산을 다 쓰므로 None (실행기가 deferred로 기록)
    return None if wait >= remaining_s else wait
    # END D22-04-I04

# ============================================================
# 5. 시도 번호 연속성 검사
# ============================================================
def check_attempts(results):
    # 모든 사례의 attempts가 4개 이하이고 send_attempt가 1, 2, ... 로 빠짐없이 이어지는지 확인한다.
    # - 예: shared_four -> send_attempt [1, 2, 3, 4]
    # - 실패한 첫 시도를 지웠다면 [2] 처럼 1이 빠져 AssertionError
    # TODO D22-04-I05: 각 시도의 번호가 누락되지 않고 총4이하인지 확인하세요.
    # 입력·완료 기준: 실패한 첫 시도도 목록에서 지우지 않습니다.
    # BEGIN D22-04-I05
    # 해설: 실패한 첫 시도도 목록에서 지우지 않습니다.
    for result in results:
        attempts = result["attempts"]
        assert len(attempts) <= 4
        # list(range(1, n + 1)) : [1, 2, ..., n]
        assert [a["send_attempt"] for a in attempts] == list(range(1, len(attempts) + 1))
    return True
    # END D22-04-I05

# HERE : 이 logic.py가 있는 교시 폴더
HERE = Path(__file__).resolve().parent

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
def main():
    # 모의 실패 사례 11개를 실행해 기대값과 대조하고, 시도 번호 연속성을 추가로 검사한다.
    # - outputs/runs/<run_id>/run.json : 사례별 실행 기록
    # - outputs/scenario_checks.json   : 사례별 actual·expected·passed
    args = parse_args(HERE)
    # failure_scenarios 인자 : classifier=분류 함수, delay=대기 함수, policy=2×2·총4회 예산
    results = failure_scenarios(args, classifier = classify, delay = delay, policy = make_policy())
    check_attempts(results)
