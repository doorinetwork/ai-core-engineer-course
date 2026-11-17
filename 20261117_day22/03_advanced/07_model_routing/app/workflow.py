"""Day22 · 명시적 모델 선택 | PPT 53~59 | 03_advanced
공통 제공 코드는 교시 독립 시작점입니다. 본 파일의 TODO 함수를 실행 경로에 연결합니다.
"""
from pathlib import Path
from coursekit.support import parse_args
from coursekit.labs import routing_lab

from .routing import route, record_view, comparison_modes, reject_unknown_mode, check_unconfigured_quality, check_evidence_view
HERE=Path(__file__).resolve().parents[1]

def main():
    args=parse_args(HERE)
    routing_lab(args,router=route,record_view=record_view,compare_modes=comparison_modes)
    reject_unknown_mode()
    check_unconfigured_quality()
    check_evidence_view()
