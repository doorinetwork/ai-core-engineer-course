"""Day22 · 메시지와 출력 계약 | PPT 14~23 | 03_advanced
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.
"""
from pathlib import Path
from coursekit.support import parse_args, execute_case, emit

from .contracts import make_request, response_gate, choose_cases, check_wire_schema
HERE=Path(__file__).resolve().parents[1]

def main():
    args=parse_args(HERE)
    cases=choose_cases()
    for case_id in ([args.case] if args.case else cases):
        result=execute_case(args,case_id,request_builder=make_request,hooks={"gate":response_gate})
        emit(args,result)
        if result["run_status"] in {"auth_error","quota_error","permission_error","not_run"}:
            break
    check_wire_schema()
