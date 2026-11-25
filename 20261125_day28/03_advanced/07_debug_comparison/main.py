"""
Day28 실습 07 · 03_advanced · 실행 진입점.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Debug 비교(한 번에 하나의 변경)
"""
from pathlib import Path
import argparse
import sys

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
# ROLE_ROOT : 이 파일 기준 두 단계 위 폴더 
ROLE_ROOT = Path(__file__).resolve().parents[2]
# 역할 폴더가 import 경로에 없을 때만 맨 앞에 추가 (coursekit 패키지를 찾기 위함)
if str(ROLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROLE_ROOT))

# 제공 모듈 coursekit/eval_workshops.py 
from coursekit.eval_workshops import execute
# 제공 모듈 coursekit/eval_report.py : dict 를 UTF-8 JSON 파일로 저장하는 write_json()
from coursekit.eval_report import write_json

# 같은 실습 폴더의 app/experiments/comparison.py : TODO 함수 score_delta() Before/After 변화량, find_problem_stage() 문제 단계 진단
from app.experiments.comparison import score_delta, find_problem_stage
# 같은 실습 폴더의 app/diagnostics/stages.py : TODO 함수 compare_four_variants() 4개 변형 비교표
from app.diagnostics.stages import compare_four_variants
# 같은 실습 폴더의 app/policies/regression.py : TODO 함수 regression_decision() 회귀·배포 보류 판정
from app.policies.regression import regression_decision


# ============================================================
# 2. 실행 흐름 main()
# ============================================================
# main 
# - --offline 또는 인자 없음 : live=False, 고정 교육용 데이터
# - --live                    : live=True (실제 RAG 실행)
# - 실패 시 : outputs/failure.json 에 오류 종류만 기록하고 예외를 다시 발생
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = "Day28 RAG Evaluation 실습")
    # add_mutually_exclusive_group() : --offline 과 --live 를 동시에 줄 수 없게 묶음
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action = 'store_true', help = '고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live', action = 'store_true', help = '실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args = parser.parse_args()
    # --------------------------------------------------------
    # TODO 함수 연결 (hooks)
    # --------------------------------------------------------
    # hooks : execute() 가 이름으로 찾아 호출하는 함수 dict (키가 있으면 해당 검사 단계 추가)
    # - "score_delta" : score_delta() Before/After 변화량
    # - "find_problem_stage" : find_problem_stage() 문제 단계 진단
    # - "compare_four_variants" : compare_four_variants() 4개 변형 비교표
    # - "regression_decision" : regression_decision() 회귀·배포 보류 판정
    hooks = {
        "score_delta": score_delta,
        "find_problem_stage": find_problem_stage,
        "compare_four_variants": compare_four_variants,
        "regression_decision": regression_decision
    }
    # 학생 과제 함수가 실제 서비스가 아니라 이 지도에 정의된 실습 단계에서 호출됩니다.
    # --------------------------------------------------------
    # 실행 및 실패 기록
    # --------------------------------------------------------
    try:
        # execute() 인자
        # 7               : 실습 번호
        # hooks           : 위 TODO 함수 dict
        # live=args.live  : True 면 실제 RAG 경로
        # level='03_advanced' : 결과 JSON 의 level 값
        result = execute(7, hooks, live = args.live, level = '03_advanced')
    except Exception as exc:
        # 실패 기록에는 Key·Password·원본 Prompt를 포함하지 않습니다.
        # failure 키
        # - status     : 'execution_failed'
        # - stage      : 예외에 stage 속성이 있으면 그 값, 없으면 'evaluation'
        # - error_type : 예외 클래스 이름만 (메시지 원문은 남기지 않음)
        # - mode       : 'live' 또는 'offline_fixture'
        failure = {'status': 'execution_failed', 'exercise': 7,
                   'stage': getattr(exc, 'stage', 'evaluation'),
                   'error_type': type(exc).__name__, 'mode': 'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent / 'outputs' / 'failure.json', failure)
        raise
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # output : 이 실습 폴더의 outputs/result.json
    output = Path(__file__).resolve().parent / 'outputs' / 'result.json'
    write_json(output, result)
    # 학생이 코드로 만든 설명 보고서는 별도 Markdown 파일로 저장합니다.
    # 주의: markdown_report 는 실습 08 응용·심화에서만 생김 (write_summary_report hook)
    if isinstance(result.get('markdown_report'), str):
        (output.parent / 'report.md').write_text(result['markdown_report'], encoding = 'utf-8')
    print('Day28 실습 7 · 03_advanced ·', result['mode'])
    print('출력:', output)
    return result

# ============================================================
# 3. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
