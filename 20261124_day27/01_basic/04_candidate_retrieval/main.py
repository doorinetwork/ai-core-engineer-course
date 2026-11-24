"""
Day27 실습 04 · Candidate Retrieval
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
# 2. 후보 검색 요청 생성 : make_candidate_request() 
# ============================================================
# make_candidate_request : 질문 dict 로 Retriever 입력 payload 를 만듦
# - 반환 dict
#   - query       : 검색 질문 (질문 원문)
#   - category    : 검색 범위 Category (예: '시설')
#   - candidate_k : Retriever 가 가져올 후보 수 5
#   - context_k   : 최종 Context 에 넣을 근거 수 2
def make_candidate_request(case):
    # BEGIN D27-04-B01
    # TODO D27-04-B01 완료: candidate_k와 context_k는 다른 제한값입니다.
    # 1. 입력: case (필요한 값만 받습니다.)
    # 2. 절차: candidate_k와 context_k는 다른 제한값입니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    return {'query': case['question'],
            'category': case['category'],
            'candidate_k': 5,
            'context_k': 2}
    # END D27-04-B01

# ============================================================
# 3. Retriever 호출 : fetch_documents() 
# ============================================================
# fetch_documents : Retriever(Runnable)를 invoke 해 후보 Document 목록을 받음
# - retriever : RunnableLambda (offline 고정 검색 / live E5+pgvector 검색)
# - payload   : make_candidate_request() 결과
# - 반환      : Document 목록 (후보 수가 candidate_k 이하)
# - 후보 수가 candidate_k 초과 : ValueError
def fetch_documents(retriever, payload):
    # BEGIN D27-04-B02
    # TODO D27-04-B02 완료: Day26에서 배운 Runnable.invoke로 후보 Document를 받아옵니다.
    # 1. 입력: retriever, payload (필요한 값만 받습니다.)
    # 2. 절차: Day26에서 배운 Runnable.invoke로 후보 Document를 받아옵니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # Day26 에서 배운 Runnable.invoke() : 입력 dict 를 넣으면 실제 검색이 실행됨
    docs = retriever.invoke(payload)
    # 요청한 후보 수보다 많이 돌아오면 검색 계약 위반으로 멈춤
    if len(docs) > payload['candidate_k']:
        raise ValueError('candidate_k 초과')
    return docs
    # END D27-04-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_04_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_04_live.json (미실행)
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 04 · 04_candidate_retrieval')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[4] = run_04() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_04() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - make_candidate_request() : 후보 검색 요청 생성
    # - fetch_documents() : Retriever 호출
    report = STAGE_RUNNERS[4](args.offline, [make_candidate_request, fetch_documents])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_04_fixture.json', 아니면 'day27_04_live.json'
    path = save_report(HERE, 'day27_04_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 04 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
