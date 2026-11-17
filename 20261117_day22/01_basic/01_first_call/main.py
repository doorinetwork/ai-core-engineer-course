"""Day22 · 첫 실제 응답 | PPT 6~13 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

01 기본필수: 첫 Responses API 요청을 만들고 한 번 보낸 뒤 source·모델·종료 상태·응답 ID만 요약해 connection_view.json에 저장한다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee(또는 mentor) 폴더. coursekit/·data/·config/가 이 위치에 있다
# sys.path.insert(0, ...) : 어느 작업 폴더에서 실행해도 coursekit 패키지를 import하도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))

# 제공 모듈 coursekit/support.py: 명령행 인자 해석·미완성 TODO 차단·교시 공통 실행(execute_case)·결과 저장(emit)
from coursekit.support import parse_args, guarded, execute_case, emit
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

    # TODO D22-01-B01: 첫 요청의 model·instructions·input·출력 상한·store를 완성하세요.
    # 입력·완료 기준: 입력은 전달받은 값입니다. 새 질문·임의 모델을 하드코딩하지 않으며 JSON Schema는 다음 교시입니다.
    # BEGIN D22-01-B01
    # 해설: 입력은 전달받은 값입니다. 새 질문·임의 모델을 하드코딩하지 않으며 JSON Schema는 다음 교시입니다.
    # request 키 : 의미
    # - model             : 전달받은 모델 ID를 그대로 사용 (임의 모델 하드코딩 금지)
    # - instructions      : 작업 규칙 (앱이 정한 지시문)
    # - input             : 이번 질문 문자열.
    # - reasoning         : {"effort": "none"} -> 추론 Token을 쓰지 않는 짧은 응답 설정
    # - max_output_tokens : 출력 상한 1200 Token (제공 코드 check_request_for_send가 1~1200만 허용)
    # - store             : False -> 공급자 쪽에 응답을 저장하지 않음 (False가 아니면 전송 전 ValueError)
    # - service_tier      : "default" -> 단가표(config/prices.json)가 적용되는 Standard 등급
    request = {"model": model, "instructions": instructions, "input": question,
               "reasoning": {"effort": "none"}, "max_output_tokens": 1200,
               "store": False, "service_tier": "default"}
    return request
    # END D22-01-B01

# ============================================================
# 3. Responses API 한 번 호출
# ============================================================
def create_response(client, request, stream=False):
    # 준비한 요청으로 client.responses.create를 정확히 한 번 호출하고 반환값을 돌려준다.
    # - client  : live이면 OpenAI(max_retries=0), simulated이면 제공 모의 SimClient
    # - request : make_first_request가 만든 dict
    # - stream  : False이면 완성된 Response 객체, True이면 이벤트 스트림

    # TODO D22-01-B02: Responses API를 한 번 호출하고 반환값을 돌려주세요.
    # 입력·완료 기준: 키를 직접 문자열로 쓰지 않습니다. 한 함수 호출은 create 한 번만 수행합니다.
    # BEGIN D22-01-B02
    # 해설: 키를 직접 문자열로 쓰지 않습니다. 한 함수 호출은 create 한 번만 수행합니다.
    # 실제 SDK 호출 위치입니다. max_retries=0인 클라이언트를 제공받습니다.
    # client.responses.create 인자
    # - **request     : model=..., instructions=..., input=... 등 요청 dict를 키워드 인자로 펼침
    # - stream=stream : 스트리밍 여부
    response = client.responses.create(**request, stream = stream)
    return response
    # END D22-01-B02

# ============================================================
# 4. 연결
# ============================================================
def make_view(result):
    # 실행 결과(result)에서 비밀 정보 없이 연결 확인에 필요한 네 값만 뽑는다.
    # - 정상 실행 : source_mode·model·run_status·마지막 시도의 response_id
    # - 미실행    : attempts가 [] 이므로 last = {} -> response_id는 None
    # - 반환값    : {"source_mode", "model", "run_status", "response_id"} 4개 키 dict
    # 예: simulated -> {"source_mode": "simulated", "model": "gpt-5.4-mini-2026-03-17",
    #                   "run_status": "completed_text", "response_id": "simulated-response"}
    # TODO D22-01-B03: 키 없이 source·model·종료 상태·응답 ID를 요약하세요.
    # 입력·완료 기준: 미실행이면 response_id는 None입니다. 0이나 가짜 응답 ID를 채우지 않습니다.
    # BEGIN D22-01-B03
    # last : 마지막 전송 시도 기록 (시도가 없으면 빈 dict)
    last = result.get("attempts", [])[-1] if result.get("attempts") else {}
    # 해설: 미실행이면 response_id는 None입니다. 0이나 가짜 응답 ID를 채우지 않습니다.
    # 반환 dict 키 : 의미
    # - source_mode : api_live(실제)·simulated(모의)·app_cache(캐시)·not_run(미실행)
    # - model       : 요청한 모델 ID
    # - run_status  : 실행 종료 상태 (예: completed_text, not_run, auth_error)
    # - response_id : 공급자 응답 ID. 미실행이면 None (가짜 ID를 채우지 않음)
    return {"source_mode": result["source_mode"], "model": result.get("model"),
            "run_status": result["run_status"], "response_id": last.get("response_id")}
    # END D22-01-B03

# 강사용 핵심 해설
# model은 문서의 후보 식별자이며 계정 접근권까지 보장하지 않습니다.
# create 한 번은 전송 한 번입니다. 재시도는 공통 실행기 한 곳에서만 관리합니다.
# 첫 성공은 연결 확인이고, 사실에 맞는 답인지와 출력 계약은 별도 검사입니다.

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # 첫 요청 생성 -> 한 번 전송 -> 결과 저장 -> 연결 요약 저장 순서로 실행한다.
    # - outputs/runs/<run_id>/run.json  : 전체 실행 기록 (attempts·usage·시간)
    # - outputs/connection_view.json    : source·model·상태·응답 ID 4개 값
    # - --source live (--allow-paid 없음) : 전송하지 않고 not_run 기록
    # --------------------------------------------------------
    # 5-1. 인자 해석 및 미완성 TODO 검사
    # --------------------------------------------------------
    # parse_args(HERE) : --source·--allow-paid 등 해석. 이 폴더에 raise NotImplementedError가 남아 있으면 전송 전에 중단
    args = parse_args(HERE)
    # --------------------------------------------------------
    # 5-2. FIRST 사례 실행
    # --------------------------------------------------------
    # execute_case 인자
    # - args                                : 명령행 인자 (source, allow_paid, mode 등)
    # - "FIRST"                             : 첫 연결 사례 (질문 "API의 역할을 쉽게 설명해 주세요.", 지식 없음)
    # - request_builder=make_first_request  : 위에서 작성한 요청 생성 함수를 실제 경로에 연결
    # - hooks={"create": create_response}   : 공통 실행기가 전송할 때 위 함수를 사용
    result = execute_case(args, "FIRST", request_builder = make_first_request, hooks = {"create": create_response})
    # emit : outputs/runs/<run_id>/ 에 run.json·request.json·raw 원문 저장, 콘솔에 status·source 출력
    emit(args, result)
    # --------------------------------------------------------
    # 5-3. 연결 요약 저장
    # --------------------------------------------------------
    view = make_view(result)
    write_json(args.output_dir / "connection_view.json", view)

# ============================================================
# 6. 스크립트 실행
# ============================================================
# guarded(main) : 미완성 TODO·설정 오류(SetupError·ValueError 등)를 짧은 안내와 종료코드 2로 바꾼다
if __name__ == "__main__":
    guarded(main)
