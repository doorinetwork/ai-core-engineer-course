"""
Day28 실습 01 · 01_basic · 실행 진입점.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

검색 실패 위치 진단
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


# ============================================================
# 2. 실패 단계 분류 : classify_case() 
# ============================================================
# classify_case : 실패 사례 1건을 System -> Candidate -> Context -> Generation 순서로 보고 첫 실패 단계를 반환
# - system_error True               : 'system_failure' (지표 계산 이전의 장애)
# - candidate_has_gold False        : 'candidate_miss' (Gold 가 검색 후보에 없음)
# - context_has_gold False          : 'ranking_or_context_miss' (후보에는 있으나 Context 에서 빠짐)
# - answer_grounded False           : 'generation_failure' (근거는 있으나 답변이 근거와 다름)
# - 모두 통과                       : 'passed'
# 예: F02 {system_error: False, candidate_has_gold: True, context_has_gold: False} -> 'ranking_or_context_miss'
def classify_case(case: dict) -> str:
    # 정답 D28-01-B01: case의 system_error·candidate_has_gold·context_has_gold·answer_grounded를 차례로 확인하고 stage 문자열을 반환하세요. 시스템 장애와 검색 실패를 구분하세요.
    # 절차: 먼저 System → Candidate → Context → Generation 순서로 판단합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # --------------------------------------------------------
    # 1) 시스템 장애를 가장 먼저 분리
    # --------------------------------------------------------
    # 주의: API 인증 오류·GPU OOM 같은 장애를 검색 실패(candidate_miss)로 세면 원인을 잘못 짚음
    if case.get('system_error'):
        return 'system_failure'
    # --------------------------------------------------------
    # 2) 파이프라인 순서대로 Gold 의 위치 확인
    # --------------------------------------------------------
    # 후보에 Gold 가 없으면 Retriever 단계 문제 (Reranker 는 후보 밖 문서를 되살리지 못함)
    if not case.get('candidate_has_gold'):
        return 'candidate_miss'
    # 후보에는 있는데 LLM Context 에 못 들어가면 순위·Context Top-N 문제
    if not case.get('context_has_gold'):
        return 'ranking_or_context_miss'
    # Context 에 근거가 있는데 답변 주장이 지지되지 않으면 생성 단계 문제
    if not case.get('answer_grounded'):
        return 'generation_failure'
    return 'passed'


# ============================================================
# 3. 판정 근거 문장 : describe_evidence() 
# ============================================================
# describe_evidence : classify_case() 가 정한 stage 를 설명하는 한 줄 근거 문장을 반환
# - case  : 원래 실패 사례 (이 구현에서는 stage 만으로 문장을 정함)
# - stage : classify_case() 반환값 5종 중 하나
# - 없는 stage : dict 조회에서 KeyError 발생 (빈 문자열로 넘기지 않음)
def describe_evidence(case: dict, stage: str) -> str:
    # 정답 D28-01-B02: 분류된 stage를 설명할 수 있는 구체적인 한 줄의 근거를 반환하세요. 빈 문자열은 안 됩니다.
    # 절차: 후보 누락과 Context 누락을 구별하는 문장으로 작성합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # stage : 근거 문장
    # - system_failure          : 실행 실패로 지표 계산 불가
    # - candidate_miss          : Gold ID 가 최초 검색 후보에 없음
    # - ranking_or_context_miss : Gold 가 후보에는 있으나 Context 에 없음
    # - generation_failure      : 근거는 Context 에 있으나 답변 주장이 지지되지 않음
    # - passed                  : 후보·Context·답변 조건 모두 충족
    evidence = {
        'system_failure': '서비스 실행이 실패해 지표를 계산할 수 없음',
        'candidate_miss': 'Gold 문서 ID가 최초 검색 후보에 존재하지 않음',
        'ranking_or_context_miss': 'Gold는 후보에 있으나 LLM Context에 포함되지 않음',
        'generation_failure': '근거가 Context에 있는데 답변 주장이 지지되지 않음',
        'passed': '검색 후보·Context·답변 검토 조건 충족',
    }
    return evidence[stage]

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 
# - --offline 또는 인자 없음 : live=False, 고정 교육용 데이터
# - --live 
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
    # - "classify_case" : classify_case() 실패 단계 분류
    # - "describe_evidence" : describe_evidence() 판정 근거 문장
    hooks = {
        "classify_case": classify_case,
        "describe_evidence": describe_evidence
    }
    # 학생 과제 함수가 실제 서비스가 아니라 이 지도에 정의된 실습 단계에서 호출됩니다.
    # --------------------------------------------------------
    # 실행 및 실패 기록
    # --------------------------------------------------------
    try:
        # execute() 인자
        # 1               : 실습 번호
        # hooks           : 위 TODO 함수 dict
        # live=args.live  : True 면 실제 RAG 경로 (06~08만)
        # level='01_basic' : 결과 JSON 의 level 값
        result = execute(1, hooks, live = args.live, level = '01_basic')
    except Exception as exc:
        # 실패 기록에는 Key·Password·원본 Prompt를 포함하지 않습니다.
        # failure 키
        # - status     : 'execution_failed'
        # - stage      : 예외에 stage 속성이 있으면 그 값, 없으면 'evaluation'
        # - error_type : 예외 클래스 이름만 (메시지 원문은 남기지 않음)
        # - mode       : 'live' 또는 'offline_fixture'
        failure = {'status': 'execution_failed', 'exercise': 1,
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
    print('Day28 실습 1 · 01_basic ·', result['mode'])
    print('출력:', output)
    return result

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
