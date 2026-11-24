"""
Day27 실습 01 · 검색 실패 위치 진단 — 기본필수 실행 진입점

- 실행 흐름: TODO 함수 2개(diagnose_stage(), recommend_action())를 coursekit/workshops.py 의 run_01() 에 주입해 실행
  - run_01() : 고정 실패 사례 7개(F01~F07)를 하나씩 진단해 rows 에 기록
- --offline : 고정 Fixture·고정 점수 경로 (DB·GPU·LLM 호출 없음) -> outputs/day27_01_fixture.json
- 기본 실행 : 실습 01 은 기본 실행도 같은 고정 사례만 사용 (외부 모델·DB 호출 없음, 파일 이름만 day27_01_live.json) -> outputs/day27_01_live.json
- 실행 위치: 역할 폴더(mentee)에서 python 01_basic/01_failure_diagnosis/main.py --offline
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
# 2. 실패 단계 판정 : diagnose_stage() 
# ============================================================
# diagnose_stage : 고정 실패 사례 1건에서 "가장 먼저" 발생한 문제 단계를 판정
# - 권한 없음(authorized False)            : 'authorization_filter'
# - 옛 버전 문서(current False)            : 'version_filter'
# - 색인 미공개(ready False)               : 'index_not_ready'
# - 질문이 모호함(ambiguous True)           : 'rewrite'
# - 정답 청크가 후보에 없음                  : 'retrieval'
# - 후보에는 있으나 Context 에 못 들어감      : 'reranking'
# - 근거와 답이 맞지 않음(valid_source False): 'validation'
# - 위 조건에 하나도 걸리지 않음              : 'healthy'
# 예: F03 {gold_in_candidates: True, gold_in_context: False} -> 'reranking'
def diagnose_stage(case):
    # BEGIN D27-01-B01
    # TODO D27-01-B01 완료: 어떤 문제가 먼저 발생했는지 위에서부터 조건을 검사합니다.
    # --------------------------------------------------------
    # 1) 검색 이전 단계(권한·버전·색인 준비)를 먼저 검사
    # --------------------------------------------------------
    if not case['authorized']:
        return 'authorization_filter'
    if not case['current']:
        return 'version_filter'
    if not case['ready']:
        return 'index_not_ready'
    # --------------------------------------------------------
    # 2) 파이프라인 순서(Rewrite -> Retrieval -> Reranking)대로 검사
    # --------------------------------------------------------
    # 질문 자체가 모호하면 검색 결과를 보기 전에 Rewrite 문제로 판정
    if case['ambiguous']:
        return 'rewrite'
    # 정답 청크가 후보 목록에 없으면 Retriever 문제 (Reranker 는 없는 후보를 찾지 못함)
    if not case['gold_in_candidates']:
        return 'retrieval'
    # 후보에는 있는데 Context 에 못 들어가면 순위(Reranking) 문제
    if not case['gold_in_context']:
        return 'reranking'
    # --------------------------------------------------------
    # 3) 생성 후 검증 단계
    # --------------------------------------------------------
    # 주의: valid_source 는 None(검증 전) 일 수 있으므로 "is False" 로만 실패 판정
    if case['valid_source'] is False:
        return 'validation'
    return 'healthy'
    # END D27-01-B01

# ============================================================
# 3. 단계별 개선 조치 : recommend_action()
# ============================================================
# recommend_action : 진단 단계 이름을 받아 그 단계에 맞는 개선 조치 문장을 반환
# - 'reranking' : 후보 순위를 다시 평가 (새 후보를 찾는 조치가 아님)
# - 'retrieval' : 검색 대상·필터·Top-k 점검 (Reranker 로는 누락 후보를 복구할 수 없음)
# - 없는 단계 이름 : dict 조회에서 KeyError 발생 (조용히 기본값을 주지 않음)
def recommend_action(stage):
    # BEGIN D27-01-B02
    # TODO D27-01-B02 완료: 문제 단계에 맞는 개선 방법을 반환합니다. Reranker는 누락된 후보를 새로 찾지 않습니다.
    # 단계 이름 : 조치 문장 (diagnose_stage() 가 돌려주는 8개 값과 1:1 대응)
    # - authorization_filter : 서버에서 권한 검사
    # - version_filter       : 현재 유효 문서만 검색
    # - index_not_ready      : Indexing 공개 상태 확인
    # - rewrite              : History 에서 명확한 대상만 복원
    # - retrieval            : 검색 대상·필터·Top-k 점검
    # - reranking            : 후보 관련성 순위 재평가
    # - validation           : 답변과 실제 근거 대조
    # - healthy              : 정상 흐름 유지
    return {'authorization_filter': '서버에서 접근 권한을 검사합니다.',
            'version_filter': '현재 유효한 문서만 검색합니다.',
            'index_not_ready': 'Indexing 공개 상태를 확인합니다.',
            'rewrite': 'History에서 명확한 대상만 복원합니다.',
            'retrieval': '검색 대상·필터·Top-k를 점검합니다.',
            'reranking': '후보 문서의 관련성 순위를 다시 평가합니다.',
            'validation': '답변과 실제 근거를 대조합니다.',
            'healthy': '정상 흐름을 유지합니다.'}[stage]
    # END D27-01-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 
# # - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_01_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_01_live.json 
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 01 · 01_failure_diagnosis')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[1] = run_01() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_01() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - diagnose_stage() : 실패 단계 판정
    # - recommend_action() : 단계별 개선 조치
    report = STAGE_RUNNERS[1](args.offline, [diagnose_stage, recommend_action])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_01_fixture.json', 아니면 'day27_01_live.json'
    path = save_report(HERE, 'day27_01_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 01 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')


# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
