"""
Day28 실습 02 · 01_basic · 실행 진입점.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Gold Dataset 구성
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
# 2. Gold ID 선택 : select_gold() 
# ============================================================
# select_gold : Gold Label 파일에서 질문의 관련 청크 ID 목록을 읽어 중복을 없앤 list 로 반환
# - question_id : 'Q01' 같은 질문 ID
# - gold        : data/gold_labels.json 의 gold dict (질문 ID -> 청크 ID 목록)
# - 반환        : 처음 나온 순서를 유지한 중복 없는 ID list
# 예: gold['Q07'] = ['FAC-104-C1', 'FAC-102-C1'] -> ['FAC-104-C1', 'FAC-102-C1']
def select_gold(question_id: str, gold: dict) -> list[str]:
    # 정답 D28-02-B01: gold[question_id]에서 관련 청크 ID 목록을 읽어 중복 없는 list로 반환하세요. 모델 검색 결과를 입력으로 쓰지 마세요.
    # 절차: Gold는 Retriever의 정답이 아니라 평가용 레이블입니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 주의: Retriever 검색 결과가 아닌 사람이 만든 평가 레이블만 입력으로 사용
    ids = list(gold[question_id])
    # dict.fromkeys(): 순서를 유지하면서 중복 키 제거 (set 은 순서를 보장하지 않음)
    return list(dict.fromkeys(ids))


# ============================================================
# 3. Gold ID 검증 : validate_gold_ids() 
# ============================================================
# validate_gold_ids : Gold ID 목록이 중복 없이 모두 Corpus 에 실제 존재하는지 검사
# - ids    : select_gold() 결과
# - corpus : 청크 ID -> 청크 dict (data/corpus.json)
# - 반환   : 두 조건을 모두 만족하면 True, 하나라도 어기면 False
# 예: [] (No-answer) -> 중복 없음·존재 검사 대상 없음 -> True
def validate_gold_ids(ids: list[str], corpus: dict) -> bool:
    # 정답 D28-02-B02: ID 중복이 없고 모든 ID가 Corpus에 실제 존재하는지 검사하세요.
    # 절차: set의 길이와 corpus.keys()를 활용하세요.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # len(ids) == len(set(ids))       : 중복 ID 없음
    # all(cid in corpus for cid in ids): 모든 ID 가 Corpus 키에 존재
    return len(ids) == len(set(ids)) and all(cid in corpus for cid in ids)

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 실행 모드를 정하고 TODO 함수를 execute(2) 에 넘겨 결과를 outputs/ 에 저장
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
    # - "select_gold" : select_gold() Gold ID 선택
    # - "validate_gold_ids" : validate_gold_ids() Gold ID 검증
    hooks = {
        "select_gold": select_gold,
        "validate_gold_ids": validate_gold_ids
    }
    # 학생 과제 함수가 실제 서비스가 아니라 이 지도에 정의된 실습 단계에서 호출됩니다.
    # --------------------------------------------------------
    # 실행 및 실패 기록
    # --------------------------------------------------------
    try:
        # execute() 인자
        # 2               : 실습 번호
        # hooks           : 위 TODO 함수 dict
        # live=args.live  : True 면 실제 RAG 경로 (06~08만)
        # level='01_basic' : 결과 JSON 의 level 값
        result = execute(2, hooks, live = args.live, level = '01_basic')
    except Exception as exc:
        # 실패 기록에는 Key·Password·원본 Prompt를 포함하지 않습니다.
        # failure 키
        # - status     : 'execution_failed'
        # - stage      : 예외에 stage 속성이 있으면 그 값, 없으면 'evaluation'
        # - error_type : 예외 클래스 이름만 (메시지 원문은 남기지 않음)
        # - mode       : 'live' 또는 'offline_fixture'
        failure = {'status': 'execution_failed', 'exercise': 2,
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
    print('Day28 실습 2 · 01_basic ·', result['mode'])
    print('출력:', output)
    return result

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
