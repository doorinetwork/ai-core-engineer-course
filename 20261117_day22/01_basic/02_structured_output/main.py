"""Day22 · 메시지와 출력 계약 | PPT 14~23 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 02 기본필수: 질문과 선택 지식만 user input으로 보내고,
공급자 Wire Schema(text.format)로 네 필드 JSON을 요청한 뒤 거절·미완료·빈 본문을 먼저 걸러낸다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee 폴더. coursekit/·data/·config/가 이 위치에 있다
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))

import json
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·교시 공통 실행(execute_case)·결과 저장(emit)
from coursekit.support import parse_args, guarded, execute_case, emit
# 제공 모듈 coursekit/request.py: 공급자용 Wire Schema(OUTPUT_FORMAT)와 기본 요청 생성기(checkpoint_request)
from coursekit.request import OUTPUT_FORMAT, build_request as checkpoint_request
# 제공 모듈 coursekit/response.py: dict·SDK 객체 공통 값 읽기(get), 응답 본문 텍스트 추출(output_text)
from coursekit.response import get, output_text

# ============================================================
# 2. 요청 만들기: user input 직렬화 + Wire Schema 연결
# ============================================================
def make_request(model, instructions, question, selected, *, structured=True):
    # 질문·선택 지식을 user 메시지로 직렬화하고 text.format에 네 필드 JSON Schema를 붙인 요청을 만든다.
    # - question : 이번 질문 (D02: "서버는 어느 폴더에서 실행하나요?")
    # - selected : 선택 지식 dict 목록 (D02: 지식 102 한 건)
    # - 반환값   : input = [{"role": "user", "content": JSON 문자열}], text = {"format": OUTPUT_FORMAT} 인 요청
    # TODO D22-02-B01: 질문과 선택 지식만 user input으로 직렬화하세요.
    # 입력·완료 기준: selected_ids가 아닌 선택된 지식의 id·title·content·category를 보냅니다. expected_behavior는 넣지 않습니다.
    # BEGIN D22-02-B01
    # 해설: selected_ids가 아닌 선택된 지식의 id·title·content·category를 보냅니다. expected_behavior는 넣지 않습니다.
    # payload 키 : 의미
    # - question  : 질문 원문
    # - knowledge : 선택 지식 본문 목록 (ID 목록이 아니라 id·title·content·category 전체)
    payload = {"question": question, "knowledge": selected}
    # json.dumps 인자
    # - payload            : 위 dict
    # - ensure_ascii=False : 한글을 \uXXXX로 바꾸지 않고 그대로 직렬화
    # 예: {"question": "서버는 어느 폴더에서 실행하나요?", "knowledge": [{"id": 102, "title": "서버 실행", ...}]}
    serialized = json.dumps(payload, ensure_ascii = False)
    # END D22-02-B01
    # TODO D22-02-B02: 공급자용 Schema를 요청의 text.format으로 연결하세요.
    # 입력·완료 기준: response_format이 아니라 Responses의 text.format입니다. Wire와 로컬 KnowledgeAnswer는 다릅니다.
    # BEGIN D22-02-B02
    # 해설: response_format이 아니라 Responses의 text.format입니다. Wire와 로컬 KnowledgeAnswer는 다릅니다.
    # checkpoint_request(..., structured=False) : 제공 생성기로 모델 허용·빈 질문·지식 ID·입력 크기를 먼저 검사한 기본 요청
    request = checkpoint_request(model, instructions, question, selected, structured = False)
    # input : 역할이 user인 메시지 한 개. 지시문(instructions)과 참고 데이터(input)를 분리한다
    request["input"] = [{"role": "user", "content": serialized}]
    # text.format = OUTPUT_FORMAT : {"type": "json_schema", "name": "knowledge_answer", "strict": True, "schema": WireAnswer 스키마}
    # 주의: Wire Schema는 네 필드 필수·추가 필드 금지만 공급자에 요구한다. 길이·출처·상태 조합은 로컬 KnowledgeAnswer가 따로 검사
    request["text"] = {"format": OUTPUT_FORMAT}
    return request
    # END D22-02-B02

# ============================================================
# 3. 응답 게이트: 거절·미완료·빈 본문 먼저 구분
# ============================================================
def response_gate(response):
    # 공급자 응답을 로컬 JSON 검증 전에 먼저 걸러 (원문, 오류) 두 값으로 돌려준다.
    # - refusal 콘텐츠 있음       : (None, "refused")  -> 형식 오류로 바꿔 재시도하지 않음
    # - status != "completed"    : (None, "incomplete" 등 API 상태)
    # - completed인데 본문이 공백 : (None, "empty_output")
    # - 정상                      : (원문 문자열, None) -> 다음 단계 inspect_answer가 JSON·Schema·출처 검사

    # TODO D22-02-B03: 거절 콘텐츠와 미완료 상태를 먼저 구분하세요.
    # 입력·완료 기준: API status와 answer JSON의 status는 다릅니다. 거절을 형식 오류로 바꾸지 않습니다.
    # BEGIN D22-02-B03
    # 해설: API status와 answer JSON의 status는 다릅니다. 거절을 형식 오류로 바꾸지 않습니다.
    # --------------------------------------------------------
    # 3-1. 거절(refusal) 콘텐츠 검사
    # --------------------------------------------------------
    # output 목록 -> type이 "message"인 항목 -> content 조각 중 type이 "refusal"인 것이 있으면 거절
    # get(..., []) or [] : 값이 None이어도 빈 목록으로 반복
    for item in get(response, "output", []) or []:
        if get(item, "type") == "message":
            for part in get(item, "content", []) or []:
                if get(part, "type") == "refusal":
                    return None, "refused"
    # --------------------------------------------------------
    # 3-2. API 종료 상태 검사
    # --------------------------------------------------------
    # 주의: 여기 status는 공급자 응답 상태(completed·incomplete·failed)다. JSON 안의 answered와 다른 값
    # 예: max_output_tokens에 걸리면 status="incomplete" -> (None, "incomplete")
    if get(response, "status") != "completed":
        return None, get(response, "status") or "response_error"
    # END D22-02-B03
    # TODO D22-02-B04: 완료 응답에서도 빈 본문을 검사한 뒤 원문을 반환하세요.
    # 입력·완료 기준: 원문을 수정해 유효한 JSON처럼 만들지 않습니다. 로컬 검증기는 다음 단계에서 실행됩니다.
    # BEGIN D22-02-B04
    # 해설: 원문을 수정해 유효한 JSON처럼 만들지 않습니다. 로컬 검증기는 다음 단계에서 실행됩니다.
    # output_text(response) : output_text 속성 또는 message 안 output_text 조각들을 이어 붙인 원문
    raw = output_text(response)
    # raw.strip() : 앞뒤 공백 제거 후 비어 있으면 빈 본문
    if not raw.strip():
        return None, "empty_output"
    # 주의: 원문을 고치거나 JSON처럼 다듬지 않고 그대로 넘긴다
    return raw, None
    # END D22-02-B04

# 강사용 핵심 해설
# instructions는 작업 규칙, input은 이번 질문과 선택된 근거입니다. 예상 정답은 전송하지 않습니다.
# 공급자 Wire Schema와 로컬 KnowledgeAnswer를 나눕니다. 로컬 길이·양수 ID·상태 조합은 계속 검사합니다.
# 공급자 status=completed와 JSON 안의 status=answered는 다른 값입니다.
# 거절은 refusal 콘텐츠로 올 수 있으며 이를 JSON 오류로 바꿔 재시도해서는 안 됩니다.

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
def main():
    # D02 한 사례를 구조화 출력으로 실행하고 결과를 저장한다.
    # - outputs/runs/<run_id>/run.json : run_status(validated 등)·answer_object·attempts·usage
    # - --case D03 처럼 지정하면 그 사례만 실행
    # - 인증·할당량·권한 오류·미실행이면 다음 사례를 보내지 않고 멈춤
    args = parse_args(HERE)
    # cases : 기본필수는 정상 질문 D02 한 건 (응용·심화는 D02·D03 두 건)
    cases = ["D02"]
    for case_id in ([args.case] if args.case else cases):
        # execute_case 인자
        # - case_id                         : "D02" -> data/cases.json의 질문·selected_ids [102]
        # - request_builder=make_request    : 위 요청 생성 함수 연결
        # - hooks={"gate": response_gate}   : 응답 검사 첫 단계로 위 게이트 함수 사용
        result = execute_case(args, case_id, request_builder = make_request, hooks = {"gate": response_gate})
        emit(args, result)
        # 주의: 인증·할당량·권한 오류는 반복해도 해결되지 않으므로 남은 사례 전송을 멈춘다
        if result["run_status"] in {"auth_error", "quota_error", "permission_error", "not_run"}:
            break

# ============================================================
# 5. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    guarded(main)
