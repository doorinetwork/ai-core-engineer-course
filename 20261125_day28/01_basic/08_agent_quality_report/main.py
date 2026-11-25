"""
Day28 실습 08 · 01_basic · 실행 진입점.
수강생은 TODO의 요구사항을 해결하고 예상 출력과 비교합니다.
강사용 코드에는 데이터 계약, 단계별 처리 의도와 예외를 주석으로 설명합니다.

Agent 품질 보고서·1차 Gate
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
# 2. 질문별 평가 레코드 : evaluate_one() 
# ============================================================
# evaluate_one : 질문 1개의 Target 출력과 Gold 를 질문 ID 로 결합해 평가 레코드를 반환
# - Trace 필수 필드(ranked_ids·candidate_ids·included_ids·answer·status·sources)가 빠지면 ValueError
def evaluate_one(example: dict, run: dict, gold: list[str], grades: dict) -> dict:
    # 정답 D28-08-B01: 질문 ID로 Target 출력과 Gold를 결합해 평가 레코드를 반환하세요.
    # 절차: Trace 필드가 빠졌다면 빈 리스트로 바꾸지 말고 오류로 보고합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 질문 단위 평가 레코드 생성
    from coursekit.eval_report import evaluate_record
    # k=3 : Top-3 기준 지표
    return evaluate_record(example, run, gold, grades, k = 3)

# ============================================================
# 3. 전체 집계 : summarize_all() 
# ============================================================
# summarize_all : 질문별 평가 레코드를 집계 (유효 Gold 질문만 평균, No-answer·오류·출처 실패는 개수로)
# - rows 비어 있음 또는 모드·버전·변형이 섞임 : ValueError
# - 반환 예: {'mean_recall_at_k': 1.0, 'mrr_at_k': 1.0, 'no_answer_excluded': 1, 'source_integrity_failures': 0, ...}
def summarize_all(rows: list[dict]) -> dict:
    # 정답 D28-08-B02: 개별 질문 결과의 평균(유효 Gold만)과 오류·출처 상태를 집계하세요.
    # 절차: No-answer 질문은 Retrieval 평균에서 제외하고 별도 개수로 기록합니다.
    # 이 구현은 모델·DB를 호출하지 않는 규칙 / 또는 필요한 함수에 위임하는 부분입니다.
    # 제공 모듈 coursekit/eval_report.py : 평균·개수 집계
    from coursekit.eval_report import report_summary
    return report_summary(rows)

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
    # - "evaluate_one" : evaluate_one() 질문별 평가 레코드
    # - "summarize_all" : summarize_all() 전체 집계
    hooks = {
        "evaluate_one": evaluate_one,
        "summarize_all": summarize_all
    }
    # 학생 과제 함수가 실제 서비스가 아니라 이 지도에 정의된 실습 단계에서 호출됩니다.
    # --------------------------------------------------------
    # 실행 및 실패 기록
    # --------------------------------------------------------
    try:
        # execute() 인자
        # 8               : 실습 번호
        # hooks           : 위 TODO 함수 dict
        # live=args.live  : True 면 실제 RAG 경로 (06~08만)
        # level='01_basic' : 결과 JSON 의 level 값
        result = execute(8, hooks, live = args.live, level = '01_basic')
    except Exception as exc:
        # 실패 기록에는 Key·Password·원본 Prompt를 포함하지 않습니다.
        # failure 키
        # - status     : 'execution_failed'
        # - stage      : 예외에 stage 속성이 있으면 그 값, 없으면 'evaluation'
        # - error_type : 예외 클래스 이름만 (메시지 원문은 남기지 않음)
        # - mode       : 'live' 또는 'offline_fixture'
        failure = {'status': 'execution_failed', 'exercise': 8,
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
    print('Day28 실습 8 · 01_basic ·', result['mode'])
    print('출력:', output)
    return result

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출
if __name__ == '__main__':
    main()
