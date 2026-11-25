"""
Day28 실습 06 · 01_basic · 실행 진입점.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

LangChain Target Runner
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
# 2. Target 입력 분리 : safe_target_input() 
# ============================================================
# safe_target_input : 평가 예제에서 Target(RAG) 입력 4개 필드만 새 dict 로 복사
# - 포함 : question_id, question, category, history
# - 제외 : answerable, reference_answer 등 정답 정보
# 예: Q01 예제 -> {'question_id': 'Q01', 'question': '회의실 예약을 취소하는 방법은?', 'category': '시설', 'history': []}
def safe_target_input(example: dict) -> dict:
    # 정답 D28-06-B01: question_id·question·category·history만 새 dict로 복사하고 Gold·Reference를 제외하세요.
    # 절차: Target Runner는 정답을 모르는 상태에서 실행해야 공정합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # Gold Label 및 Reference Answer는 모델 입력으로 전달하지 않습니다.
    # 허용 키 목록을 직접 지정해 새 dict 생성 (example 에서 빼는 방식은 새 정답 필드가 생기면 누출됨)
    return {key: example[key] for key in ('question_id', 'question', 'category', 'history')}


# ============================================================
# 3. Runnable.invoke 호출 : invoke_target() 
# ============================================================
# invoke_target : LangChain Runnable(Target)의 invoke() 를 호출하고 출력 계약을 확인
# - target  : coursekit 가 RunnableLambda 로 감싼 RAG 실행 함수
# - request : safe_target_input() 결과
# - Gold 관련 키가 request 에 있으면 ValueError (assert_no_gold_leakage)
# - 결과가 dict 가 아니거나 question_id 가 다르면 ValueError
def invoke_target(target, request: dict) -> dict:
    # 정답 D28-06-B02: 실제 Runnable.invoke(request)를 호출하고 결과가 dict이며 질문 ID가 같은지 확인하세요.
    # 절차: 단순 함수 직접 호출 대신 RunnableLambda를 감싼 객체의 invoke()입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_dataset.py : 금지 키(gold_chunk_ids·reference_answer 등) 포함 여부 검사
    from coursekit.eval_dataset import assert_no_gold_leakage
    assert_no_gold_leakage(request)
    # target.invoke(request) : RunnableLambda 안의 함수 실행 (offline 은 고정 Fixture, live 는 실제 RAG)
    result = target.invoke(request)
    if not isinstance(result, dict) or result.get('question_id') != request['question_id']:
        raise ValueError('Target 출력의 질문 ID 불일치')
    return result

# ============================================================
# 4. 실행 흐름 main()
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
    # - "safe_target_input" : safe_target_input() Target 입력 분리
    # - "invoke_target" : invoke_target() Runnable.invoke 호출
    hooks = {
        "safe_target_input": safe_target_input,
        "invoke_target": invoke_target
    }
    # 학생 과제 함수가 실제 서비스가 아니라 이 지도에 정의된 실습 단계에서 호출됩니다.
    # --------------------------------------------------------
    # 실행 및 실패 기록
    # --------------------------------------------------------
    try:
        # execute() 인자
        # 6               : 실습 번호
        # hooks           : 위 TODO 함수 dict
        # live=args.live  : True 면 실제 RAG 경로 (06~08만)
        # level='01_basic' : 결과 JSON 의 level 값
        result = execute(6, hooks, live = args.live, level = '01_basic')
    except Exception as exc:
        # 실패 기록에는 Key·Password·원본 Prompt를 포함하지 않습니다.
        # failure 키
        # - status     : 'execution_failed'
        # - stage      : 예외에 stage 속성이 있으면 그 값, 없으면 'evaluation'
        # - error_type : 예외 클래스 이름만 (메시지 원문은 남기지 않음)
        # - mode       : 'live' 또는 'offline_fixture'
        failure = {'status': 'execution_failed', 'exercise': 6,
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
    print('Day28 실습 6 · 01_basic ·', result['mode'])
    print('출력:', output)
    return result

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
