"""
Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다.
"""
from pathlib import Path
import sys

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
# HERE : 이 main.py 가 있는 실습 폴더 (결과 outputs/ 저장 위치)
HERE = Path(__file__).resolve().parent
# ROLE_ROOT : 상위 폴더를 차례로 보며 coursekit/ 폴더가 있는 첫 폴더 
ROLE_ROOT = next(p for p in HERE.parents if (p / 'coursekit').is_dir())
# 역할 폴더를 import 경로 맨 앞에 넣어 coursekit 을 찾게 함
sys.path.insert(0, str(ROLE_ROOT))

import argparse

# 제공 모듈 coursekit/workshops.py : 실습 번호별 공통 실행 골격 dict 
from coursekit.workshops import STAGE_RUNNERS
# 제공 모듈 coursekit/io.py : 결과 dict 를 <폴더>/outputs/<파일명> JSON 으로 저장
from coursekit.io import save_report

# ============================================================
# 2. CrossEncoder 입력 쌍 생성 : create_pairs() 
# ============================================================
# create_pairs : CrossEncoder 입력용 (질문, 문서본문) 쌍 목록을 만듦
# - question : 검색 질문
# - docs     : 후보 Document 목록
# - 반환     : [(질문, page_content), ...] (후보 순서 유지)
# 예: [('회의실 예약을 취소하는 방법은?', '회의실은 사내 예약 시스템에서 …'), ...]
def create_pairs(question, docs):
    # BEGIN D27-05-B01
    # TODO D27-05-B01 완료: CrossEncoder 입력은 [질문, 문서본문]의 쌍 목록입니다.
    # 1. 입력: question, docs (필요한 값만 받습니다.)
    # 2. 절차: CrossEncoder 입력은 [질문, 문서본문]의 쌍 목록입니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 질문 하나와 각 후보 본문을 짝지음 (Metadata 는 넣지 않음)
    return [(question, d.page_content) for d in docs]
    # END D27-05-B01

# ============================================================
# 3. 후보 재정렬 : rank_documents() 
# ============================================================
# rank_documents : 같은 후보들을 CrossEncoder 점수로 다시 정렬하고 상위 3개를 반환
# - question : 검색 질문
# - docs     : Retriever 가 찾은 후보 Document 목록 (새 후보를 추가하지 않음)
# - scorer   : 쌍 목록 -> 점수 목록 함수 (offline 고정 점수 / live CUDA CrossEncoder)
# - 반환     : 복사본 Document 목록 (metadata 에 pre_rank·rerank_score·post_rank 추가)
# 예: before [FAC-101, FAC-107, FAC-102, …] -> after [FAC-102(3.2), FAC-103(2.6), FAC-104(2.1)]
def rank_documents(question, docs, scorer):
    # BEGIN D27-05-B02
    # TODO D27-05-B02 완료: Reranker는 후보를 새로 찾지 않고 동일 후보를 순위 변경합니다.
    # 1. 입력: question, docs, scorer (필요한 값만 받습니다.)
    # 2. 절차: Reranker는 후보를 새로 찾지 않고 동일 후보를 순위 변경합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/reranking.py 의 rerank_documents() 사용
    from coursekit.reranking import rerank_documents
    # rerank_documents() 인자
    # question : 질문 (빈 문자열이면 ValueError)
    # docs     : 후보 Document 목록
    # scorer   : 점수 함수 (점수 개수 불일치·NaN·Inf 이면 ValueError)
    # top_n=3  : 재정렬 후 상위 3개만 반환
    return rerank_documents(question, docs, scorer, top_n = 3)
    # END D27-05-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main :
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_05_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_05_live.json 
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 05 · 05_cross_encoder_reranking')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[5] = run_05() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_05() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - create_pairs() : CrossEncoder 입력 쌍 생성
    # - rank_documents() : 후보 재정렬
    report = STAGE_RUNNERS[5](args.offline, [create_pairs, rank_documents])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_05_fixture.json', 아니면 'day27_05_live.json'
    path = save_report(HERE, 'day27_05_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 05 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
