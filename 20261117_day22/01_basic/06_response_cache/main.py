"""Day22 · 승인 응답 캐시 | PPT 47~52 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 06 기본필수: 사람이 승인한 답만 캐시에 넣어 재사용한다.
캐시 키에는 scope·모델·전체 요청(질문·지식 본문)·Schema·검증 버전이 모두 들어가 같은 ID의 본문 변경도 미스가 된다.
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

from copy import deepcopy
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·캐시 실습 흐름(cache_lab: run/reuse/changed)
from coursekit.support import parse_args, guarded, cache_lab
# 제공 모듈 coursekit/io.py: JSON 저장, 정규화 JSON의 SHA-256 해시(digest)
from coursekit.io import write_json, digest
# 제공 모듈 coursekit/request.py: 기본 요청 생성기(checkpoint_request)
from coursekit.request import build_request as checkpoint_request
# 제공 모듈 coursekit/cache.py: 만료·승인·scope·원출처·검증을 모두 확인하는 안전한 캐시 조회기(checkpoint_cache)
from coursekit.cache import read_cache as checkpoint_cache

# ============================================================
# 2. 캐시 키 만들기
# ============================================================
def key_of(request, scope, validator_version="day21-v1"):
    # 요청 전체와 실행 조건으로 캐시 키(SHA-256 16진 64자)를 만든다.
    # - request           : 전송할 요청 dict (model·instructions·input(질문+지식 본문)·text.format 포함)
    # - scope             : 권한 범위 이름 (기본 "classroom-demo")
    # - validator_version : 로컬 검증기 버전 ("day21-v1")
    # - 반환값            : digest(basis) -> 같은 입력이면 같은 키, 하나라도 다르면 다른 키

    # TODO D22-06-B01: 질문뿐 아니라 모델·전체요청·Schema·검증버전·scope로 캐시키를 만드세요.
    # 입력·완료 기준: API키는 포함하지 않습니다. 전체 본문이 키에 들어가야 같은 ID의 내용 변경도 감지합니다.
    # BEGIN D22-06-B01
    # 해설: API키는포함하지않습니다. 전체본문이키에들어가야같은ID의내용변경도감지합니다.
    # basis 키 : 의미
    # - scope             : 권한 범위 (다른 그룹의 승인 답을 재사용하지 않기 위함)
    # - model             : 모델 ID (fast·quality가 다르면 다른 키)
    # - request           : 요청 전체 (질문·지식 본문·지시문·출력 상한)
    # - format            : text.format의 Wire Schema (없으면 None)
    # - validator_version : 로컬 검증 규칙 버전
    # 주의: API key는 넣지 않는다
    basis = {"scope": scope, "model": request["model"], "request": request,
             "format": request.get("text", {}).get("format"), "validator_version": validator_version}
    # digest : 키 정렬·공백 없는 JSON으로 바꾼 뒤 SHA-256 -> 예: "3f1c...a9" (64자)
    return digest(basis)
    # END D22-06-B01

# ============================================================
# 3. 승인 캐시 조회
# ============================================================
def approved_cache(cache, key, now, provided_ids, *, expected_source, scope):
    # 제공 조회기로 캐시를 찾는다. 조건 하나라도 맞지 않으면 None(미스)을 돌려준다.
    # - cache           : {키: 항목} dict (사람 승인 review.json으로 만든 항목 1개)
    # - key             : 현재 요청으로 만든 캐시 키
    # - now             : 현재 epoch 초. now >= expires_at 이면 만료
    # - provided_ids    : 현재 제공한 지식 ID 집합 (예: {103})
    # - expected_source : 현재 실행 출처 (live -> "api_live", simulated -> "simulated")
    # - scope           : 현재 권한 범위
    # - hit 반환값      : source_mode="app_cache", new_api_send_count=0, usage 모두 0, origin_run_id=원래 실행 ID

    # TODO D22-06-B02: 현재문맥·실행출처·scope와 함께 제공된 안전한 캐시검사를 호출하세요.
    # 입력·완료 기준: 만료·미승인·현재출처 불일치는 미스입니다. 승인하려면 tools/review.py로 사람이 기록합니다.
    # BEGIN D22-06-B02
    # 해설: 만료·미승인·현재출처 불일치는 미스입니다. 승인하려면 tools/review.py로 사람이 기록합니다.
    # checkpoint_cache 인자 : 받은 값을 모두 그대로 전달 (scope·expected_source를 빼면 다른 범위·출처의 답을 재사용할 위험)
    return checkpoint_cache(cache, key, now, provided_ids, expected_source = expected_source, scope = scope)
    # END D22-06-B02

# ============================================================
# 4. 본문 변경 키 비교 (전송 0회)
# ============================================================
def compare_context_keys():
    # H01 요청과, 같은 ID 103의 본문만 바꾼 요청의 캐시 키가 다른지 확인한다.
    # - 반환값 : {"changed_context_miss": True, "source_mode": "simulated", "new_api_send_count": 0}
    # - 키가 같으면 AssertionError
    # TODO D22-06-B03: 같은ID의본문변경이캐시키를바꾸는지확인하세요.
    # 입력·완료 기준: 이 키 비교는 전송 0회의 모의검사입니다. 실제 재사용은 먼저 생성·사람 승인후 reuse 명령으로 확인합니다.
    # BEGIN D22-06-B03
    # 해설: 이 키 비교는 전송 0회의 모의검사입니다. 실제 재사용은 먼저 생성·사람 승인 후 reuse 명령으로 확인합니다.
    # 제공 모듈 coursekit/io.py: 사례·선택 지식·지시문 읽기(input_parts)
    from coursekit.io import input_parts
    # 제공 모듈 coursekit/config.py: 설정 읽기(load_settings). simulated는 .env·API key를 읽지 않음
    from coursekit.config import load_settings
    # case : H01 사례, selected : [지식 103], instructions : instructions_v1.txt
    case, selected, instructions = input_parts("H01")
    model = load_settings("simulated").models["fast"]
    first = checkpoint_request(model, instructions, case["question"], selected)
    # 같은 ID 103의 본문 끝에 문장 추가
    # 예: "오류 메시지와 시도한 작업을 함께 적습니다." -> "... 함께 적습니다. 실행 환경도 함께 적습니다."
    selected[0]["content"] += " 실행 환경도 함께 적습니다."
    second = checkpoint_request(model, instructions, case["question"], selected)
    # ID 목록은 [103]으로 같아도 본문이 달라 키가 달라야 한다
    assert key_of(first, "classroom-demo") != key_of(second, "classroom-demo")
    return {"changed_context_miss": True, "source_mode": "simulated", "new_api_send_count": 0}
    # END D22-06-B03

# 강사용 핵심 해설
# 같은 ID라도 본문이 바뀌면 답의 근거가 달라집니다. ID 목록만 캐시 키로 쓰지 않습니다.
# 자동 Schema 통과는 의미적 사실성 통과가 아닙니다. 사람이 질문·주장·인용 본문을 검토한 answered만 승인합니다.
# scope 키는 권한 검사 자체가 아닙니다. 실제 서비스에서는 인증된 권한 범위와 무효화 정책이 필요합니다.
# hit는 새로운 API 전송과 사용량이 0입니다. origin_run_id로 원래 생성 비용을 중복 합산하지 않습니다.

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # 키 비교 결과를 저장한 뒤 --action에 따라 생성·재사용·변경 흐름을 실행한다.
    # - outputs/cache_key_checks.json : 본문 변경 키 비교 결과
    # - --action run (기본)  : H01 생성. validated이면 review.py 명령 안내 출력
    # - --action reuse       : --run-file의 review.json(approve) 확인 후 app_cache hit (새 전송 0)
    # - --action changed     : 본문 변경 요청 -> cache_miss 기록 -> 별도 새 호출 기록
    args = parse_args(HERE)
    write_json(args.output_dir / "cache_key_checks.json", compare_context_keys())
    # cache_lab 인자
    # - key_builder=key_of          : 위 캐시 키 함수
    # - cache_reader=approved_cache : 위 조회 함수
    result = cache_lab(args, key_builder = key_of, cache_reader = approved_cache)

# ============================================================
# 6. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    guarded(main)
