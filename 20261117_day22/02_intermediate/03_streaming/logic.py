"""Day22 · Streaming과 완료 판정 | PPT 24~31 | 02_intermediate
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.
"""
# ============================================================
# 1. 모듈 import
# ============================================================
from pathlib import Path
# 제공 모듈 coursekit/support.py: 인자 해석·TODO 차단·교시 공통 실행·결과 저장
from coursekit.support import parse_args, execute_case, emit
# 제공 모듈 coursekit/io.py: UTF-8 JSON 저장 도구
from coursekit.io import write_json
# 제공 모듈 coursekit/response.py: dict·SDK 객체 공통 값 읽기(get)
from coursekit.response import get

# ============================================================
# 2. 스트리밍 이벤트 한 개 처리
# ============================================================
def receive_event(state, event, clock, started):
    # 스트리밍 이벤트 한 개를 받아 StreamState(state)에 누적한다. 공통 실행기가 이벤트마다 호출한다.
    # - "response.output_text.delta"               : 조각 누적, 첫 비어 있지 않은 조각 시각을 first_text_ms(ms)로 기록
    # - "response.completed"/"incomplete"/"failed" : 최종 Response를 state.final에 보관
    # - "error"                                     : state.error = "stream_error"
    # - 그 외 (created, output_text.done 등)        : 개수만 세고 성공 판정에 쓰지 않음

    state.events_seen += 1
    kind = get(event, "type")
    if kind == "response.output_text.delta":
        # TODO D22-03-I01: 비어 있지 않은 첫 텍스트의 시간과 delta를 누적하세요.
        # 입력·완료 기준: clock은 단조 시계입니다. 빈 delta를 첫 표시로 세지 않습니다.
        # BEGIN D22-03-I01
        # 해설: clock은 단조 시계입니다. 빈 delta를 첫 표시로 세지 않습니다.
        delta = get(event, "delta", "")
        # 문자열이 아닌 delta : 프로토콜 오류로 표시하고 누적하지 않음
        if not isinstance(delta, str):
            state.error = "stream_protocol_error"
            return
        # 첫 비어 있지 않은 delta만 첫 표시 시간으로 기록. 빈 delta("")는 시간 기준이 아니다
        if delta and state.first_text_ms is None:
            state.first_text_ms = (clock() - started) * 1000
        state.parts.append(delta)
        # END D22-03-I01
    elif kind in {"response.completed", "response.incomplete", "response.failed"}:
        # TODO D22-03-I02: 최종 이벤트의 Response를 보관하세요.
        # 입력·완료 기준: 부분 문자열을 합쳤다고 completed를 직접 만들어서는 안 됩니다.
        # BEGIN D22-03-I02
        # 해설: 부분 문자열을 합쳤다고 completed를 직접 만들어서는 안 됩니다.
        # 최종 이벤트가 두 번 오면 프로토콜 이상으로 기록
        if state.final is not None:
            state.error = "duplicate_terminal_event"
        # 주의: 최종 판정은 공급자가 보낸 Response의 status로 한다
        state.final = get(event, "response")
        # END D22-03-I02
    elif kind == "error":
        state.error = "stream_error"

# ============================================================
# 3. 스트리밍 선택값과 비교 순서
# ============================================================
def streaming_choice():
    # 기본 실행(--action run)이 스트리밍 요청인지 돌려준다.
    # - 반환값 True : 스트리밍 1회
    # TODO D22-03-I03: 실제 스트리밍을 요청하도록 선택값을 반환하세요.
    # 입력·완료 기준: 메시지 UI를 새로 만드는 실습이 아닙니다. 최종 상태·부분 파일·first_text_ms를 확인합니다.
    # BEGIN D22-03-I03
    # 해설: 메시지 UI를 새로 만드는 실습이 아닙니다. 최종 상태·부분 파일·first_text_ms를 확인합니다.
    return True
    # END D22-03-I03

def compare_runs():
    # --action compare에서 실행할 stream 값 순서를 돌려준다.
    # - 반환값 [False, True] : 같은 H01 입력을 비스트리밍 1회 -> 스트리밍 1회 (총 2회)
    # TODO D22-03-I04: 동일 입력의 비스트리밍과 스트리밍 순서를 지정하세요.
    # 입력·완료 기준: --action compare에서만 2회 호출합니다. 기본 run은 스트리밍1회입니다.
    # BEGIN D22-03-I04
    # 해설: --action compare에서만 2회 호출합니다. 기본 run은 스트리밍1회입니다.
    return [False, True]
    # END D22-03-I04

# ============================================================
# 4. 시간 지표 분리 요약
# ============================================================
def timing_view(result):
    # 실행 결과에서 시간 지표 세 가지를 출처와 함께 뽑는다.
    # - first_text_ms : 첫 비어 있지 않은 텍스트 관측 시간. 비스트리밍이면 None (0이 아님)
    # - request_ms    : 전송 시도들의 요청 시간 합
    # - e2e_ms        : 준비부터 검증까지 전체 경과 시간
    # 예: simulated 비스트리밍 -> {"source_mode": "simulated", "first_text_ms": None, "request_ms": 0.001, ...}
    # TODO D22-03-I05: 첫 표시 시간과 전체 요청 시간을 분리해 요약하세요.
    # 입력·완료 기준: 미표시인 None과 실제 관측0은 다릅니다. 시뮬레이션 시간을 실제 LLM 속도로 해석하지 않습니다.
    # BEGIN D22-03-I05
    # 해설: 미표시인 None과 실제 관측0은 다릅니다. 시뮬레이션 시간을 실제 LLM 속도로 해석하지 않습니다.
    # 주의: source_mode를 함께 남겨 simulated 가상 시계 값을 실측 속도로 읽지 않게 한다
    return {"source_mode": result["source_mode"], "first_text_ms": result.get("first_text_ms"),
            "request_ms": result.get("request_ms"), "e2e_ms": result.get("e2e_ms")}
    # END D22-03-I05

# HERE : 이 logic.py가 있는 교시 폴더
HERE = Path(__file__).resolve().parent

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
def main():
    # 기본은 스트리밍 1회, --action compare이면 비스트리밍·스트리밍 2회를 실행하고 시간 비교표를 저장한다.
    # - outputs/runs/<run_id>/run.json    : 실행 기록 (스트리밍은 partial_attempt_01.txt 포함)
    # - outputs/timing_comparison.json   : timing_view 행 목록 (run 1행, compare 2행)
    args = parse_args(HERE)
    # modes : compare -> [False, True], 그 외 -> [True]
    modes = compare_runs() if args.action == "compare" else [streaming_choice()]
    observations = []
    for use_stream in modes:
        # execute_case 인자
        # - args.case or "H01"             : 기본 질문 H01 (지식 103)
        # - stream=use_stream              : 이번 실행의 스트리밍 여부
        # - hooks={"event": receive_event} : 이벤트 처리에 위 함수를 사용
        result = execute_case(args, args.case or "H01", stream = use_stream, hooks = {"event": receive_event})
        emit(args, result)
        observations.append(timing_view(result))
    write_json(args.output_dir / "timing_comparison.json", observations)
