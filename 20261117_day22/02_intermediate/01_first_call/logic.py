"""Day22 · 첫 실제 응답 | PPT 6~13 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 01 응용 TODO 모듈: 첫 요청 생성·한 번 호출·키 없는 연결 요약에 더해 
"키 미설정"과 "모델 접근 거부"가 어디서 생기는 오류인지 모의 비교표(settings_cases.json)로 정리한다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·교시 공통 실행(execute_case)·결과 저장(emit)
from coursekit.support import parse_args, execute_case, emit
# 제공 모듈 coursekit/io.py: UTF-8 JSON 저장 도구
from coursekit.io import write_json

# ============================================================
# 2. 첫 요청 딕셔너리 만들기
# ============================================================
def make_first_request(model, instructions, question, selected, *, structured=False):
    # Responses API에 보낼 첫 요청(request) 딕셔너리를 만든다.
    # - model        : 설정에서 고른 모델 ID (예: "gpt-5.4-mini-2026-03-17")
    # - instructions : 작업 규칙 문장 (FIRST 사례는 "한국어 한 문장으로 답하세요.")
    # - question     : 이번 질문 (FIRST 사례는 "API의 역할을 쉽게 설명해 주세요.")
    # - selected     : 선택 지식 목록. FIRST 사례는 [] 이고 이 함수에서는 쓰지 않는다
    # - structured   : JSON Schema 사용 여부. 첫 호출은 일반 텍스트라 False
    # - 반환값       : model·instructions·input·reasoning·max_output_tokens·store·service_tier 7개 키 dict

    # TODO D22-01-I01: 첫 요청의 model·instructions·input·출력 상한·store를 완성하세요.
    # 입력·완료 기준: 입력은 전달받은 값입니다. 새 질문·임의 모델을 하드코딩하지 않으며 JSON Schema는 다음 교시입니다.
    # BEGIN D22-01-I01
    # 해설: 입력은 전달받은 값입니다. 새 질문·임의 모델을 하드코딩하지 않으며 JSON Schema는 다음 교시입니다.
    # request 키 : 의미
    # - model             : 전달받은 모델 ID를 그대로 사용 (임의 모델 하드코딩 금지)
    # - instructions      : 작업 규칙 (앱이 정한 지시문)
    # - input             : 이번 질문 문자열
    # - reasoning         : {"effort": "none"} -> 추론 Token을 쓰지 않는 짧은 응답 설정
    # - max_output_tokens : 출력 상한 1200 Token (제공 검사 check_request_for_send가 1~1200만 허용)
    # - store             : False -> 공급자 쪽에 응답을 저장하지 않음
    # - service_tier      : "default" -> 단가표가 적용되는 Standard 등급
    request = {"model": model, "instructions": instructions, "input": question,
               "reasoning": {"effort": "none"}, "max_output_tokens": 1200,
               "store": False, "service_tier": "default"}
    return request
    # END D22-01-I01

# ============================================================
# 3. Responses API 한 번 호출
# ============================================================
def create_response(client, request, stream=False):
    # 준비한 요청으로 client.responses.create를 정확히 한 번 호출하고 반환값을 돌려준다.
    # - client  : live이면 OpenAI(max_retries=0), simulated이면 제공 모의 SimClient
    # - request : make_first_request가 만든 dict
    # - stream  : False이면 완성된 Response 객체

    # TODO D22-01-I02: Responses API를 한 번 호출하고 반환값을 돌려주세요.
    # 입력·완료 기준: 키를 직접 문자열로 쓰지 않습니다. 한 함수 호출은 create 한 번만 수행합니다.
    # BEGIN D22-01-I02
    # 해설: 키를 직접 문자열로 쓰지 않습니다. 한 함수 호출은 create 한 번만 수행합니다.
    # 실제 SDK 호출 위치입니다. max_retries=0인 클라이언트를 제공받습니다.
    # client.responses.create 인자
    # - **request     : 요청 dict를 키워드 인자로 펼침
    # - stream=stream : 스트리밍 여부
    response = client.responses.create(**request, stream = stream)
    return response
    # END D22-01-I02

# ============================================================
# 4. 연결 요약
# ============================================================
def make_view(result):
    # 실행 결과에서 비밀 정보 없이 연결 확인에 필요한 네 값만 뽑는다.
    # - 정상 실행 : source_mode·model·run_status·마지막 시도의 response_id
    # - 미실행    : attempts가 [] 이므로 response_id는 None
    # 예: simulated -> {"source_mode": "simulated", "model": "gpt-5.4-mini-2026-03-17",
    #                   "run_status": "completed_text", "response_id": "simulated-response"}
    # TODO D22-01-I03: 키 없이 source·model·종료 상태·응답 ID를 요약하세요.
    # 입력·완료 기준: 미실행이면 response_id는 None입니다. 0이나 가짜 응답 ID를 채우지 않습니다.
    # BEGIN D22-01-I03
    # 해설: 미실행이면 response_id는 None입니다. 0이나 가짜 응답 ID를 채우지 않습니다.
    # last : 마지막 전송 시도 기록 (시도가 없으면 빈 dict)
    last = result.get("attempts", [])[-1] if result.get("attempts") else {}
    # 반환 dict 키 : 의미
    # - source_mode : api_live·simulated·app_cache·not_run
    # - model       : 요청한 모델 ID
    # - run_status  : 실행 종료 상태 (예: completed_text, not_run)
    # - response_id : 공급자 응답 ID. 미실행이면 None
    return {"source_mode": result["source_mode"], "model": result.get("model"),
            "run_status": result["run_status"], "response_id": last.get("response_id")}
    # END D22-01-I03

# ============================================================
# 5. 설정 오류 두 종류 모의 비교표
# ============================================================
def setting_cases():
    # "키 미설정"과 "모델 접근 거부"를 발생 위치와 전송 수로 구분한 표를 돌려준다.
    # - missing_key  : 로컬 설정 단계(load_settings)에서 SetupError -> 전송 전에 멈춤, 실제 전송 0회
    # - model_denied : 공급자가 요청을 받은 뒤 403/404로 거부 -> 전송 1회가 이미 일어남 (여기서는 모의 1회로 표기)
    # TODO D22-01-I04: 키 미설정과 모델 접근 오류의 차이를 모의 비교표에 적으세요.
    # 입력·완료 기준: 반복 실제 호출로 오류를 만들지 않습니다. 모델 접근 거부는 403/404 등 공급자 원인을 봅니다.
    # BEGIN D22-01-I04
    # 해설: 반복 실제 호출로 오류를 만들지 않습니다. 모델 접근 거부는 403/404 등 공급자 원인을 봅니다.
    # 행 키 : 의미
    # - case                 : 오류 사례 이름
    # - where                : 오류가 드러나는 위치 (local_settings = 내 PC 설정, provider_permission = 공급자 권한)
    # - new_api_send_count   : 실제 전송 수 (키 미설정은 0)
    # - simulated_send_count : 모의 전송 수 (모델 거부는 1번 보내야 알 수 있음)
    # 주의: 이 표는 직접 작성한 비교이며 실제 403/404를 유발해 얻은 값이 아니다
    return [{"case": "missing_key", "where": "local_settings", "new_api_send_count": 0},
            {"case": "model_denied", "where": "provider_permission", "simulated_send_count": 1}]
    # END D22-01-I04

# HERE : 이 logic.py가 있는 교시 폴더 (parse_args가 outputs 위치·TODO 검사 범위로 사용)
HERE = Path(__file__).resolve().parent

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
def main():
    # 첫 요청 실행 -> 결과 저장 -> 연결 요약 저장 -> 설정 오류 비교표 저장 순서로 실행한다.
    # - outputs/runs/<run_id>/run.json : 전체 실행 기록
    # - outputs/connection_view.json   : source·model·상태·응답 ID
    # - outputs/settings_cases.json    : 설정 오류 두 종류 비교표
    args = parse_args(HERE)
    # execute_case 인자
    # - "FIRST"                            : 첫 연결 사례 (지식 없음, 일반 텍스트)
    # - request_builder=make_first_request : 위 요청 생성 함수 연결
    # - hooks={"create": create_response}  : 전송에 위 함수 사용
    result = execute_case(args, "FIRST", request_builder = make_first_request, hooks = {"create": create_response})
    emit(args, result)
    view = make_view(result)
    write_json(args.output_dir / "connection_view.json", view)
    write_json(args.output_dir / "settings_cases.json", setting_cases())
