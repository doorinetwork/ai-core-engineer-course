"""
03 · @tool·Pydantic·JSON Schema · 심화 · 실제 11/20 report_agent_lab 기반 실습.

Day29 실습 03 심화 실행 진입점. tasks.py + policy_03.py 의 TODO 함수를 차례로 호출해 결과를 출력
"""
from pathlib import Path
import sys, argparse

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
# ROOT : 이 파일 기준 두 단계 위 폴더 
ROOT = Path(__file__).resolve().parents[2]
# 역할 폴더를 import 경로 맨 앞에 추가해 coursekit 패키지를 찾게 함
sys.path.insert(0, str(ROOT))

# 같은 폴더 tasks.py : 이 실습의 TODO 함수 모음 
import tasks

# ============================================================
# 2. 실행 흐름 main()
# ============================================================
# main : 명령행 인자를 읽고 --check 면 함수 목록만, 아니면 실제 DB·GPU·LLM 으로 실습 흐름을 실행
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    ap = argparse.ArgumentParser(description = 'Day29 · 기존 DB 재사용 / CPU fallback 없음')
    # action='store_true' : 옵션을 주면 True, 안 주면 False
    ap.add_argument('--check', action = 'store_true', help = '실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve', action = 'store_true', help = '실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args = ap.parse_args()
    # --------------------------------------------------------
    # --check : 함수 이름만 출력하고 종료
    # --------------------------------------------------------
    if args.check:
        # 실습 진입점에서의 확인은 코드 구조만 진단합니다. 정답 코드는 학생이 작성합니다.
        # dir(tasks) 중 '_' 로 시작하지 않고 호출 가능한 이름 -> 함수 이름 목록 (예: ['classify_request', 'plan_for_request'])
        print('TODO 함수:', [name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks, name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    # serve : --serve 지정 여부 (실습 08 에서만 분기에 사용)
    serve = args.serve
    # --------------------------------------------------------
    # Tool 등록 -> 필수 인수 강제 -> 실제 조회
    # --------------------------------------------------------
    registered = tasks.register_status_tool()
    # enforce_explicit_schema() : report_id 가 required 면 {'name': 'lookup_status', 'required': ['report_id']}
    print('[Schema]', tasks.enforce_explicit_schema(registered))
    # invoke({'report_id': 63}) : 이 시점에 실제 DB SELECT 실행, JSON 문자열 반환
    print('[실제 DB 조회]', registered.invoke({'report_id': 63}))

# ============================================================
# 3. 스크립트 실행 진입점
# ============================================================
if __name__ == '__main__':
    main()
