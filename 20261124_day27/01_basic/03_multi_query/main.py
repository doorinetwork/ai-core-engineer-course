"""
Day27 실습 03 · Multi-query 후보 병합
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

# 제공 모듈 coursekit/workshops.py : 실습 번호별 공통 실행 골격 dict (1~8 -> run_01~run_08)
from coursekit.workshops import STAGE_RUNNERS
# 제공 모듈 coursekit/io.py : 결과 dict 를 <폴더>/outputs/<파일명> JSON 으로 저장
from coursekit.io import save_report


# ============================================================
# 2. 검색 질문 목록 계획 : plan_queries() 
# ============================================================
# plan_queries : 원 질문을 맨 앞에 두고 추가 질문을 붙여 중복 없는 검색 질문 목록을 만듦 (기본필수: 2개)
# - base   : 원 질문 (항상 첫 번째로 보존)
# - extras : 추가 검색 표현 후보 (offline 고정 3개 / live 는 LLM 이 최대 2개 생성)
# - 반환   : 공백 제거·중복 제거된 질문 목록, 최대 2개
# 예: base + extras 3개 -> extras[:1] 만 사용 -> ['회의실 예약을 취소하는 방법은?', '회의실 예약 철회는 어떻게 하나요?']
def plan_queries(base, extras):
    # BEGIN D27-03-B01
    # TODO D27-03-B01 완료: 원 질문을 포함한 서로 다른 질문 2개로 시작합니다.
    # 1. 입력: base, extras (필요한 값만 받습니다.)
    # 2. 절차: 원 질문을 포함한 서로 다른 질문 2개로 시작합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/multi_query.py 의 unique_queries() 사용
    from coursekit.multi_query import unique_queries
    # unique_queries() 인자
    # base           : 원 질문 (목록 맨 앞)
    # extras[:1]     : 추가 질문은 첫 1개만 (기본필수는 서로 다른 질문 2개로 시작)
    # max_queries=2  : 원 질문 포함 최대 2개
    return unique_queries(base, extras[:1], max_queries = 2)
    # END D27-03-B01

# ============================================================
# 3. 후보 병합·중복 제거 : merge_candidates() 
# ============================================================
# merge_candidates : 질문별 검색 결과를 chunk_id 기준으로 합치고 중복을 제거
# - pairs : [(질문, [Document, ...]), ...]
# - 반환  : 처음 나온 순서를 유지한 Document 목록
#   - 각 Document.metadata['found_by_query'] 에 그 청크를 찾은 질문 목록 기록
# - 같은 chunk_id 인데 본문이 다름 : ValueError (데이터 불일치를 숨기지 않음)
# 예: FAC-102-C1 이 모든 질문에서 발견되면 found_by_query 에 그 질문들이 순서대로 모두 기록됨
def merge_candidates(pairs):
    # BEGIN D27-03-B02
    # TODO D27-03-B02 완료: chunk_id를 중복 제거 키로 사용하고 found_by_query를 Metadata에 기록합니다.
    # 1. 입력: pairs (필요한 값만 받습니다.)
    # 2. 절차: chunk_id를 중복 제거 키로 사용하고 found_by_query를 Metadata에 기록합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/multi_query.py 의 merge_found() 사용 (원본 Document 는 깊은 복사 후 수정)
    from coursekit.multi_query import merge_found
    return merge_found(pairs)
    # END D27-03-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main :
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_03_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_03_live.json (미실행)
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 03 · 03_multi_query')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[3] = run_03() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_03() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - plan_queries() : 검색 질문 목록 계획
    # - merge_candidates() : 후보 병합·중복 제거
    report = STAGE_RUNNERS[3](args.offline, [plan_queries, merge_candidates])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_03_fixture.json', 아니면 'day27_03_live.json'
    path = save_report(HERE, 'day27_03_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 03 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
