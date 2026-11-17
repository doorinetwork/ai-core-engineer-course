"""Day22 · 승인 응답 캐시 | PPT 47~52 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 06 응용 TODO 모듈: 전체 요청 기반 캐시 키, 안전한 캐시 검사 연결에 더해
같은 ID의 본문 변경·모델 변경이 키를 바꾸는지, TTL 만료 시각에 재사용하지 않는지 전송 0회로 확인한다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
from copy import deepcopy
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·캐시 실습 실행기(cache_lab)
from coursekit.support import parse_args, cache_lab
# 제공 모듈 coursekit/io.py: UTF-8 JSON 저장(write_json)·정렬 JSON SHA-256 요약(digest)
from coursekit.io import write_json, digest
# 제공 모듈 coursekit/request.py: 기본 요청 생성기(checkpoint_request)
from coursekit.request import build_request as checkpoint_request
# 제공 모듈 coursekit/cache.py: 만료·승인·scope·출처·검증 버전을 확인하는 안전한 캐시 읽기(checkpoint_cache)
from coursekit.cache import read_cache as checkpoint_cache

# ============================================================
# 2. 캐시 키 만들기
# ============================================================
def key_of(request, scope, validator_version="day21-v1"):
    # 요청 전체와 사용 범위·검증 버전을 묶어 캐시 키(해시 문자열)를 만든다.
    # - request           : 공급자에 보낼 요청 dict 전체 (model·instructions·input·text 등)
    # - scope             : 캐시 사용 범위 이름 (예: "classroom-demo")
    # - validator_version : 로컬 검증기 버전 (기본 "day21-v1")
    # - 반환값            : digest(basis) 해시 문자열. 한 글자만 달라도 다른 키
    
    # TODO D22-06-I01: 질문뿐 아니라 모델·전체요청·Schema·검증버전·scope로 캐시키를 만드세요.
    # 입력·완료 기준: API키는 포함하지 않습니다. 전체 본문이 키에 들어가야 같은 ID의 내용 변경도 감지합니다.
    # BEGIN D22-06-I01
    # 해설: API키는 포함하지 않습니다. 전체 본문이 키에 들어가야 같은 ID의 내용 변경도 감지합니다.
    # basis 키 : 의미
    # - scope             : 사용 범위
    # - model             : 모델 ID
    # - request           : 요청 전체 (지식 본문 포함)
    # - format            : text.format Schema (없으면 None)
    # - validator_version : 검증 버전
    # 주의: API key·Authorization 헤더는 넣지 않는다
    basis = {"scope": scope, "model": request["model"], "request": request,
             "format": request.get("text", {}).get("format"), "validator_version": validator_version}
    return digest(basis)
    # END D22-06-I01

# ============================================================
# 3. 승인 캐시 읽기
# ============================================================
def approved_cache(cache, key, now, provided_ids, *, expected_source, scope):
    # 제공된 안전한 캐시 검사로 재사용 가능한 결과를 찾는다. 조건이 하나라도 맞지 않으면 None(미스)이다.
    # - cache           : {키: 캐시 항목} dict
    # - now             : 현재 시각(초). expires_at 이상이면 만료
    # - provided_ids    : 이번 요청에 제공된 지식 ID 집합 (출처 검사용)
    # - expected_source : 현재 실행 출처 (예: "simulated"). 원본 출처와 달라도 미스
    # - scope           : 사용 범위. 다르면 미스
    # - 반환값          : 적중이면 source_mode "app_cache"·새 전송 0회 결과 dict, 아니면 None
    
    # TODO D22-06-I02: 현재문맥·실행출처·scope와함께 제공된 안전한캐시검사를 호출하세요.
    # 입력·완료 기준: 만료·미승인·현재출처불일치는미스입니다. 승인하려면tools/review.py로사람이기록합니다.
    # BEGIN D22-06-I02
    # 해설: 만료·미승인·현재출처불일치는미스입니다. 승인하려면tools/review.py로사람이기록합니다.
    return checkpoint_cache(cache, key, now, provided_ids, expected_source = expected_source, scope = scope)
    # END D22-06-I02

# ============================================================
# 4. 문맥 변경 키 비교 (전송 0회)
# ============================================================
def compare_context_keys():
    # H01 요청에서 지식 본문만 한 문장 바꿨을 때 캐시 키가 달라지는지 확인한다.
    # - first  : 원래 지식 103으로 만든 요청
    # - second : 같은 ID 103의 content 끝에 " 실행 환경도 함께 적습니다." 를 붙인 요청
    # - 반환값 : {"changed_context_miss": True, "source_mode": "simulated", "new_api_send_count": 0}
    # TODO D22-06-I03: 같은ID의본문변경이캐시키를바꾸는지확인하세요.
    # 입력·완료 기준: 이키비교는전송0회의모의검사입니다. 실제재사용은먼저생성·사람승인후reuse명령으로확인합니다.
    # BEGIN D22-06-I03
    # 해설: 이키비교는전송0회의모의검사입니다. 실제재사용은먼저생성·사람승인후reuse명령으로확인합니다.
    from coursekit.io import input_parts
    from coursekit.config import load_settings
    # input_parts("H01") : (사례 dict, 선택 지식 목록, 지시문 문자열)
    case, selected, instructions = input_parts("H01")
    model = load_settings("simulated").models["fast"]
    first = checkpoint_request(model, instructions, case["question"], selected)
    # 같은 ID의 본문만 변경 (ID 목록은 그대로)
    selected[0]["content"] += " 실행 환경도 함께 적습니다."
    second = checkpoint_request(model, instructions, case["question"], selected)
    assert key_of(first, "classroom-demo") != key_of(second, "classroom-demo")
    return {"changed_context_miss": True, "source_mode": "simulated", "new_api_send_count": 0}
    # END D22-06-I03

# ============================================================
# 5. 모델 변경 키 비교 (전송 0회)
# ============================================================
def model_key_check():
    # 질문·문맥은 같고 모델만 fast -> quality로 바꿨을 때 캐시 키가 달라지는지 확인한다.
    # - a : fast 모델 요청 (예: "gpt-5.4-mini-2026-03-17")
    # - b : quality 모델 요청 (예: "gpt-5.4-2026-03-05")
    # - 반환값 True (같으면 AssertionError)
    # TODO D22-06-I04: 모델선택만바뀌어도다른키가되는지확인하세요.
    # 입력·완료 기준: 스트리밍여부는출력전달방식이며요청내생성설정은같게유지합니다.
    # BEGIN D22-06-I04
    # 해설: 스트리밍여부는출력전달방식이며요청내생성설정은같게유지합니다.
    from coursekit.io import input_parts
    from coursekit.config import load_settings
    case, rows, prompt = input_parts("H01")
    # models : {"fast": 모델 ID, "quality": 모델 ID}
    models = load_settings("simulated").models
    a = checkpoint_request(models["fast"], prompt, case["question"], rows)
    b = checkpoint_request(models["quality"], prompt, case["question"], rows)
    assert key_of(a, "classroom-demo") != key_of(b, "classroom-demo")
    return True
    # END D22-06-I04

# ============================================================
# 6. TTL 만료 검사
# ============================================================
def ttl_check():
    # 만료 시각과 현재 시각이 같은(10.0초) 캐시 항목을 재사용하지 않는지 확인한다.
    # - fake : 승인은 되어 있지만 expires_at=10.0인 가짜 항목
    # - now=10.0 : 만료 시각과 같음 -> now >= expires_at 이므로 미스(None)
    # - 반환값 True
    # TODO D22-06-I05: 만료시간에같거나지난캐시가재사용되지않는지검사하세요.
    # 입력·완료 기준: TTL을읽을때마다연장하지않습니다.
    # BEGIN D22-06-I05
    # 해설: TTL을읽을때마다연장하지않습니다.
    fake = {"k": {"expires_at": 10.0, "human_approved": True}}
    assert approved_cache(fake, "k", 10.0, {103}, expected_source = "simulated", scope = "classroom-demo") is None
    return True
    # END D22-06-I05

# HERE : 이 logic.py가 있는 교시 폴더
HERE = Path(__file__).resolve().parent

# ============================================================
# 7. 실행 흐름 (main)
# ============================================================
def main():
    # 전송 0회 키 검사 3종을 먼저 실행한 뒤 --action에 따라 생성·재사용·변경 미스를 실행한다.
    # - outputs/cache_key_checks.json : 문맥 변경 키 비교 결과
    # - --action run     : H01 생성 후 tools/review.py 승인 안내 출력
    # - --action reuse   : 승인 캐시 적중이면 app_cache (새 전송 0회)
    # - --action changed : 변경된 문맥으로 미스 확인 후 새로 실행
    args = parse_args(HERE)
    write_json(args.output_dir / "cache_key_checks.json", compare_context_keys())
    model_key_check()
    ttl_check()
    # cache_lab 인자
    # - key_builder=key_of           : 위 키 생성 함수
    # - cache_reader=approved_cache  : 위 캐시 읽기 함수
    result = cache_lab(args, key_builder = key_of, cache_reader = approved_cache)
