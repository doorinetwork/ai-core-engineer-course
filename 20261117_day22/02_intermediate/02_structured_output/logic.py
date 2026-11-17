"""Day22 · 메시지와 출력 계약 | PPT 14~23 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
import json
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·교시 공통 실행(execute_case)·결과 저장(emit)
from coursekit.support import parse_args, execute_case, emit
# 제공 모듈 coursekit/request.py: 공급자용 Wire Schema(OUTPUT_FORMAT)와 기본 요청 생성기(checkpoint_request)
from coursekit.request import OUTPUT_FORMAT, build_request as checkpoint_request
# 제공 모듈 coursekit/response.py: dict·SDK 객체 공통 값 읽기(get), 응답 본문 텍스트 추출(output_text)
from coursekit.response import get, output_text

# ============================================================
# 2. 요청 만들기: user input 직렬화 + Wire Schema 연결
# ============================================================
def make_request(model, instructions, question, selected, *, structured=True):
    # 질문·선택 지식을 user 메시지로 직렬화하고 text.format에 네 필드 JSON Schema를 붙인 요청을 만든다.
    # - question : 이번 질문 (D02: "서버는 어느 폴더에서 실행하나요?", D03: "파일은 몇 년 보관하나요?")
    # - selected : 선택 지식 dict 목록 (D02: 지식 102, D03: 지식 101)
    # - 반환값   : input = [{"role": "user", "content": JSON 문자열}], text = {"format": OUTPUT_FORMAT} 인 요청
    # TODO D22-02-I01: 질문과 선택 지식만 user input으로 직렬화하세요.
    # 입력·완료 기준: selected_ids가 아닌 선택된 지식의 id·title·content·category를 보냅니다. expected_behavior는 넣지 않습니다.
    # BEGIN D22-02-I01
    # 해설: selected_ids가 아닌 선택된 지식의 id·title·content·category를 보냅니다. expected_behavior는 넣지 않습니다.
    # payload 키 : 의미
    # - question  : 질문 원문
    # - knowledge : 선택 지식 본문 목록 (ID 목록이 아니라 id·title·content·category 전체)
    payload = {"question": question, "knowledge": selected}
    # ensure_ascii=False : 한글을 \uXXXX로 바꾸지 않고 그대로 직렬화
    serialized = json.dumps(payload, ensure_ascii = False)
    # END D22-02-I01
    # TODO D22-02-I02: 공급자용 Schema를 요청의 text.format으로 연결하세요.
    # 입력·완료 기준: response_format이 아니라 Responses의 text.format입니다. Wire와 로컬 KnowledgeAnswer는 다릅니다.
    # BEGIN D22-02-I02
    # 해설: response_format이 아니라 Responses의 text.format입니다. Wire와 로컬 KnowledgeAnswer는 다릅니다.
    # checkpoint_request(..., structured=False) : 모델 허용·빈 질문·지식 ID·입력 크기를 먼저 검사한 기본 요청
    request = checkpoint_request(model, instructions, question, selected, structured = False)
    # input : 역할이 user인 메시지 한 개 (지시문 instructions와 분리)
    request["input"] = [{"role": "user", "content": serialized}]
    # text.format = OUTPUT_FORMAT : {"type": "json_schema", "name": "knowledge_answer", "strict": True, "schema": ...}
    # 주의: Wire Schema는 네 필드 필수·추가 필드 금지만 요구한다. 길이·출처·상태 조합은 로컬 KnowledgeAnswer가 따로 검사
    request["text"] = {"format": OUTPUT_FORMAT}
    return request
    # END D22-02-I02

# ============================================================
# 3. 응답 게이트: 거절·미완료·빈 본문 먼저 구분
# ============================================================
def response_gate(response):
    # 공급자 응답을 로컬 JSON 검증 전에 먼저 걸러 (원문, 오류) 두 값으로 돌려준다.
    # - refusal 콘텐츠 있음       : (None, "refused")
    # - status != "completed"    : (None, API 상태 문자열) 예: "incomplete"
    # - completed인데 본문이 공백 : (None, "empty_output")
    # - 정상                      : (원문, None) -> 다음 단계 inspect_answer가 JSON·Schema·출처 검사

    # TODO D22-02-I03: 거절 콘텐츠와 미완료 상태를 먼저 구분하세요.
    # 입력·완료 기준: API status와 answer JSON의 status는 다릅니다. 거절을 형식 오류로 바꾸지 않습니다.
    # BEGIN D22-02-I03
    # 해설: API status와 answer JSON의 status는 다릅니다. 거절을 형식 오류로 바꾸지 않습니다.
    # --------------------------------------------------------
    # 3-1. 거절(refusal) 콘텐츠 검사
    # --------------------------------------------------------
    # output -> type "message" 항목 -> content 조각 중 type "refusal"이 있으면 거절
    for item in get(response, "output", []) or []:
        if get(item, "type") == "message":
            for part in get(item, "content", []) or []:
                if get(part, "type") == "refusal":
                    return None, "refused"
    # --------------------------------------------------------
    # 3-2. API 종료 상태 검사
    # --------------------------------------------------------
    # 주의: 공급자 status(completed·incomplete)와 JSON 안의 status(answered·insufficient_context)는 다른 값
    if get(response, "status") != "completed":
        return None, get(response, "status") or "response_error"
    # END D22-02-I03
    # TODO D22-02-I04: 완료 응답에서도 빈 본문을 검사한 뒤 원문을 반환하세요.
    # 입력·완료 기준: 원문을 수정해 유효한 JSON처럼 만들지 않습니다. 로컬 검증기는 다음 단계에서 실행됩니다.
    # BEGIN D22-02-I04
    # 해설: 원문을 수정해 유효한 JSON처럼 만들지 않습니다. 로컬 검증기는 다음 단계에서 실행됩니다.
    # output_text(response) : output_text 속성 또는 message 안 output_text 조각을 이어 붙인 원문
    raw = output_text(response)
    if not raw.strip():
        return None, "empty_output"
    return raw, None
    # END D22-02-I04

# ============================================================
# 4. 실행 사례 순서
# ============================================================
def choose_cases():
    # 실행할 사례 ID를 순서대로 돌려준다.
    # - 반환값 : ["D02", "D03"]
    #   D02 "서버는 어느 폴더에서 실행하나요?" -> 지식 102로 답변 가능 (answered, source_ids [102])
    #   D03 "파일은 몇 년 보관하나요?"         -> 지식 101에 보관 기간이 없어 정보 부족 (insufficient_context)
    # TODO D22-02-I05: 정상 질문 D02와 정보 부족 질문 D03의 순서를 지정하세요.
    # 입력·완료 기준: 실제 호출 2개입니다. 부족 응답은 유효한 업무 결과일 수 있습니다.
    # BEGIN D22-02-I05
    # 해설: 실제 호출 2개입니다. 부족 응답은 유효한 업무 결과일 수 있습니다.
    return ["D02", "D03"]
    # END D22-02-I05

# HERE : 이 logic.py가 있는 교시 폴더 (outputs 위치·TODO 검사 범위)
HERE = Path(__file__).resolve().parent

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # D02·D03을 구조화 출력으로 차례로 실행하고 결과를 저장한다.
    # - outputs/runs/<run_id>/run.json : 사례별 run_status·answer_object·attempts·usage
    # - 인증·할당량·권한 오류·미실행이면 다음 사례를 보내지 않고 멈춤
    args = parse_args(HERE)
    cases = choose_cases()
    for case_id in ([args.case] if args.case else cases):
        # execute_case 인자
        # - request_builder=make_request  : 위 요청 생성 함수
        # - hooks={"gate": response_gate} : 응답 검사 첫 단계로 위 게이트 함수 사용
        result = execute_case(args, case_id, request_builder = make_request, hooks = {"gate": response_gate})
        emit(args, result)
        if result["run_status"] in {"auth_error", "quota_error", "permission_error", "not_run"}:
            break
