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
# 2. 재정렬 전·후 Top-k 비교 : compare_rankings() 
# ============================================================
# compare_rankings : 재정렬 전·후 ID 목록에서 Gold 청크가 Top-2 에 들어 있는지 비교
# - before : 재정렬 전 chunk_id 목록
# - after  : 재정렬 후 chunk_id 목록
# - gold   : 평가 전용 정답 chunk_id 목록 (검색·Reranker 입력으로 쓰지 않음)
# - 반환 dict : before · after · gold · k · before_contains_gold · after_contains_gold
#   - gold 가 비어 있으면 *_contains_gold 는 None
def compare_rankings(before, after, gold):
    # BEGIN D27-07-B01
    # TODO D27-07-B01 완료: 고정한 관련 문서 ID를 Top-2에 포함했는지 확인합니다.
    # 1. 입력: before, after, gold (필요한 값만 받습니다.)
    # 2. 절차: 고정한 관련 문서 ID를 Top-2에 포함했는지 확인합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/evaluation.py 의 compare_lists() 사용
    from coursekit.evaluation import compare_lists
    # k=2 : 상위 2개(Context 에 들어가는 개수) 안에 gold 가 있는지 확인
    return compare_lists(before, after, gold, k = 2)
    # END D27-07-B01

# ============================================================
# 3. 비교 결과 해석 : interpret_change() 
# ============================================================
# interpret_change : 비교 결과를 단정하지 않는 해석 문장으로 바꿈
# - gold 없음(before_contains_gold None) : 'gold 없음: 인간 검토 필요'
# - 전·후 포함 여부 같음                 : 'Top-k 관련 문서 포함 여부는 변하지 않음'
# - 전·후 포함 여부 다름                 : '관련 문서 Top-k 포함 여부가 달라짐 (추가 검증 필요)'
# 주의: 순위가 바뀌어도 "개선됐다" 고 결론 내리지 않음 (단일 사례·고정 점수)
def interpret_change(report):
    # BEGIN D27-07-B02
    # TODO D27-07-B02 완료: 순위가 바뀌어도 반드시 개선됐다는 결론을 내리지 않습니다.
    # 1. 입력: report (필요한 값만 받습니다.)
    # 2. 절차: 순위가 바뀌어도 반드시 개선됐다는 결론을 내리지 않습니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 전·후 Top-k 에 gold 가 들어 있었는지 꺼냄 (True / False / None)
    before = report['before_contains_gold']
    after = report['after_contains_gold']
    # 평가 기준(gold)이 없으면 자동 판정하지 않음
    if before is None:
        return 'gold 없음: 인간 검토 필요'
    if before == after:
        return 'Top-k 관련 문서 포함 여부는 변하지 않음'
    return '관련 문서 Top-k 포함 여부가 달라짐 (추가 검증 필요)'
    # END D27-07-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main :
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_07_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_07_live.json
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 07 · 07_before_after')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[7] = run_07() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_07() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - compare_rankings() : 재정렬 전·후 Top-k 비교
    # - interpret_change() : 비교 결과 해석
    report = STAGE_RUNNERS[7](args.offline, [compare_rankings, interpret_change])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_07_fixture.json', 아니면 'day27_07_live.json'
    path = save_report(HERE, 'day27_07_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 07 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
