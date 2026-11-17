"""Day22 · Streaming과 완료 판정 | PPT 24~31 | 01_basic
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.

실습 03 기본필수: H01을 Streaming으로 받아 텍스트 조각(delta)을 모으고,
첫 비어 있지 않은 텍스트 시각(first_text_ms)과 최종 Response를 따로 보관한 뒤 완료 상태로만 성공을 판정한다.
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

# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·교시 공통 실행·결과 저장
from coursekit.support import parse_args, guarded, execute_case, emit
# 제공 모듈 coursekit/response.py: dict·SDK 객체 공통 값 읽기(get)
from coursekit.response import get

# ============================================================
# 2. 스트리밍 이벤트 한 개 처리
# ============================================================
def receive_event(state, event, clock, started):
    # 스트리밍 이벤트 한 개를 받아 StreamState(state)에 누적한다. 공통 실행기가 이벤트마다 호출한다.
    # - state   : StreamState (parts=텍스트 조각 목록, first_text_ms, final=최종 Response, error, events_seen)
    # - event   : 공급자 이벤트 (type 필드로 종류 구분)
    # - clock   : 단조 시계 함수 (live: perf_counter, simulated: VirtualClock.now)
    # - started : 전송 시작 시각 (clock 기준)
    # - "response.output_text.delta"                       : 텍스트 조각 누적, 첫 비어 있지 않은 조각 시각 기록
    # - "response.completed"/"incomplete"/"failed"         : 최종 Response 보관
    # - "error"                                             : state.error = "stream_error"
    # - 그 외 (created, output_text.done 등)                : 개수만 세고 성공 판정에 쓰지 않음

    # events_seen : 받은 이벤트 수 (중단 시 부분 텍스트 보존 여부 판단에 사용)
    state.events_seen += 1
    kind = get(event, "type")
    if kind == "response.output_text.delta":
        # TODO D22-03-B01: 비어 있지 않은 첫 텍스트의 시간과 delta를 누적하세요.
        # 입력·완료 기준: clock은 단조 시계입니다. 빈 delta를 첫 표시로 세지 않습니다.
        # BEGIN D22-03-B01
        # 해설: clock은 단조 시계입니다. 빈 delta를 첫 표시로 세지 않습니다.
        delta = get(event, "delta", "")
        # delta가 문자열이 아니면 프로토콜 오류로 표시하고 누적하지 않는다
        if not isinstance(delta, str):
            state.error = "stream_protocol_error"
            return
        # 첫 비어 있지 않은 delta만 첫 표시 시간으로 기록 (ms 단위)
        # 예: 모의 이벤트 [""(빈 조각), "{\"status\"...", "...}"] -> 두 번째 이벤트 시각이 first_text_ms
        # 주의: 빈 delta는 화면에 아무것도 보이지 않으므로 시간 기준이 아니다
        if delta and state.first_text_ms is None:
            state.first_text_ms = (clock() - started) * 1000
        # 조각은 빈 문자열도 순서대로 보관 ("".join(parts)가 부분 원문)
        state.parts.append(delta)
        # END D22-03-B01
    elif kind in {"response.completed", "response.incomplete", "response.failed"}:
        # TODO D22-03-B02: 최종 이벤트의 Response를 보관하세요.
        # 입력·완료 기준: 부분 문자열을 합쳤다고 completed를 직접 만들어서는 안 됩니다.
        # BEGIN D22-03-B02
        # 해설: 부분 문자열을 합쳤다고 completed를 직접 만들어서는 안 됩니다.
        # 최종 이벤트가 두 번 오면 프로토콜 이상으로 기록
        if state.final is not None:
            state.error = "duplicate_terminal_event"
        # event.response : 공급자가 보낸 최종 Response 객체 (status·output·usage 포함)
        # 주의: 최종 판정은 이 객체의 status로 한다. parts를 이어 붙여 completed를 만들지 않는다
        state.final = get(event, "response")
        # END D22-03-B02
    elif kind == "error":
        state.error = "stream_error"

# ============================================================
# 3. 스트리밍 사용 여부 선택
# ============================================================
def streaming_choice():
    # 이 교시의 기본 실행이 스트리밍 요청인지 돌려준다.
    # - 반환값 True : execute_case(..., stream=True) -> client.responses.create(..., stream=True)
    # TODO D22-03-B03: 실제 스트리밍을 요청하도록 선택값을 반환하세요.
    # 입력·완료 기준: 메시지 UI를 새로 만드는 실습이 아닙니다. 최종 상태·부분 파일·first_text_ms를 확인합니다.
    # BEGIN D22-03-B03
    # 해설: 메시지 UI를 새로 만드는 실습이 아닙니다. 최종 상태·부분 파일·first_text_ms를 확인합니다.
    return True
    # END D22-03-B03

# 강사용 핵심 해설
# delta는 텍스트 조각이며 Token 1개와 대응하지 않습니다. 조각 수로 비용을 계산하지 않습니다.
# first_text_ms는 첫 비어 있지 않은 delta를 관측한 시간입니다. 소켓 연결 완료 시간과 다릅니다.
# output_text.done은 문자열 한 부분의 종료일 수 있습니다. 최종 Response의 상태도 확인해야 합니다.
# 중간 출력은 콘솔에 미검증으로 표시하며 DB·캐시에 저장 승인하지 않습니다.

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
def main():
    # H01을 스트리밍으로 한 번 실행하고 결과를 저장한다.
    # - 콘솔 : "[Streaming · simulated · 미검증 중간 출력]" 뒤에 조각을 이어 출력, 끝나면 수신 종료 안내
    # - outputs/runs/<run_id>/run.json             : first_text_ms·request_ms·run_status·usage
    # - outputs/runs/<run_id>/partial_attempt_01.txt : 받은 조각을 이어 붙인 부분 원문
    args = parse_args(HERE)
    # modes : 기본필수는 스트리밍 1회 [True]
    modes = [streaming_choice()]
    # observations : 응용·심화의 시간 비교표용 목록 (기본필수에서는 채우지 않음)
    observations = []
    for use_stream in modes:
        # execute_case 인자
        # - args.case or "H01"             : 기본 질문 H01 "오류 질문에 무엇을 적나요?" (지식 103)
        # - stream=use_stream              : True -> 스트리밍 요청
        # - hooks={"event": receive_event} : 이벤트 처리에 위 함수를 사용
        result = execute_case(args, args.case or "H01", stream = use_stream, hooks = {"event": receive_event})
        emit(args, result)

# ============================================================
# 5. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    guarded(main)
