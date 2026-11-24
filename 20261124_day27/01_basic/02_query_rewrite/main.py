"""
Day27 실습 02 · Query Rewrite
"""
from pathlib import Path
import sys

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
# HERE : 이 main.py 가 있는 실습 폴더 (결과 outputs/ 저장 위치)
HERE = Path(__file__).resolve().parent
# ROLE_ROOT : 상위 폴더를 차례로 보며 coursekit/ 폴더가 있는 첫 폴더 (= mentee 역할 폴더)
ROLE_ROOT = next(p for p in HERE.parents if (p / 'coursekit').is_dir())
# 역할 폴더를 import 경로 맨 앞에 넣어 coursekit 을 찾게 함
sys.path.insert(0, str(ROLE_ROOT))

import argparse

# 제공 모듈 coursekit/workshops.py : 실습 번호별 공통 실행 골격 dict
from coursekit.workshops import STAGE_RUNNERS
# 제공 모듈 coursekit/io.py : 결과 dict 를 <폴더>/outputs/<파일명> JSON 으로 저장
from coursekit.io import save_report

# ============================================================
# 2. Query Rewrite 실행 : apply_rewrite() 
# ============================================================
# apply_rewrite : 대화 History 를 반영해 마지막 질문을 독립적인 검색 질문으로 변환
# - offline=True  : coursekit 고정 규칙(rewrite_fixture) 사용, LLM 호출 없음
# - offline=False : ChatPromptTemplate -> Chat Model(Structured Output) 실제 호출 (API 비용 발생)
# - 반환 : RewriteResult (status 'ready' 또는 'clarification_required', search_query, reason)
# 예: '그럼 취소는 어떻게 해?' + 회의실 History -> search_query '회의실 예약을 취소하는 방법은?'
def apply_rewrite(question, history, offline):
    # BEGIN D27-02-B01
    # TODO D27-02-B01 완료: offline이면 고정 예시, 기본 실행이면 실제 ChatPromptTemplate → Chat Model을 호출합니다.
    # 1. 입력: question, history, offline (필요한 값만 받습니다.)
    # 2. 절차: offline이면 고정 예시, 기본 실행이면 실제 ChatPromptTemplate → Chat Model을 호출합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/rewrite.py 의 rewrite_query() 사용 (빈 질문은 ValueError)
    from coursekit.rewrite import rewrite_query
    # rewrite_query() 인자
    # question        : 사용자가 입력한 원 질문
    # history         : 이전 대화 Turn 목록 (role·content)
    # offline=offline : True 면 고정 규칙, False 면 실제 LLM
    return rewrite_query(question, history, offline = offline)
    # END D27-02-B01

# ============================================================
# 3. 원 질문·검색 질문 분리 기록 : record_rewrite() 
# ============================================================
# record_rewrite : 원 질문과 검색용 질문을 별도 키로 나눠 기록
# - original : 사용자가 입력한 원 질문 (덮어쓰지 않음)
# - result   : apply_rewrite() 가 돌려준 RewriteResult
# - 반환 dict
#   - original_question : 원 질문 그대로
#   - search_query      : 검색에만 쓰는 독립 질문 (확인 필요이면 '')
#   - status            : 'ready' 또는 'clarification_required'
#   - reason            : 변환 또는 확인 요청 이유
def record_rewrite(original, result):
    # BEGIN D27-02-B02
    # TODO D27-02-B02 완료: 원래 질문을 검색 질문으로 덮어쓰지 않습니다.
    # (mentor 정답 주석 원문)
    # ANSWER D27-02-B02: 원래 질문을 검색 질문으로 덮어쓰지 않습니다.
    # 1. 입력: original, result (필요한 값만 받습니다.)
    # 2. 절차: 원래 질문을 검색 질문으로 덮어쓰지 않습니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 주의: 답변 생성·로그에는 원 질문이 남아야 하므로 search_query 로 대체하지 않음
    return {'original_question': original,
            'search_query': result.search_query,
            'status': result.status,
            'reason': result.reason}
    # END D27-02-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main :
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_02_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_02_live.json (미실행)
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 02 · 02_query_rewrite')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[2] = run_02() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_02() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - apply_rewrite() : Query Rewrite 실행
    # - record_rewrite() : 원 질문·검색 질문 분리 기록
    report = STAGE_RUNNERS[2](args.offline, [apply_rewrite, record_rewrite])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_02_fixture.json', 아니면 'day27_02_live.json'
    path = save_report(HERE, 'day27_02_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 02 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
