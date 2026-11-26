"""01 · AI Agent vs 고정 RAG/Workflow · 응용 · 실제 11/20 report_agent_lab 기반 실습."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import tasks


def main():
    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()
    if args.check:
        # 실습 진입점에서의 확인은 코드 구조만 진단합니다. 정답 코드는 학생이 작성합니다.
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    serve=args.serve
    from coursekit.db import preflight,fetch_report_status
    print('[기존 DB]',preflight())
    p=tasks.plan_for_request('RAG 근거 보고서를 찾아 보고서 63번 상태도 알려줘')
    print(p,'복합 소스:',tasks.needs_two_sources(p))
    print('[실제 상태]',fetch_report_status(63))

if __name__=='__main__':main()
