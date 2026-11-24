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
# 2. Pipeline 요청 생성 : prepare_pipeline_request() 
# ============================================================
# prepare_pipeline_request : 질문 dict 를 RAG Pipeline 입력(RagRequest 형식) dict 로 만듦
# - 반환 dict
#   - query          : 원 질문 (검색용 질문은 Chain 안 Rewrite 단계가 만듦)
#   - history        : 이전 대화 Turn 목록
#   - category       : 검색 Category
#   - candidate_k    : 검색 후보 수 5
#   - context_k      : Context 근거 수 2
#   - use_rewrite    : True  (History 반영 Rewrite 사용)
#   - use_multiquery : False (추가 질문 검색 사용 안 함)
#   - use_rerank     : True  (CrossEncoder 재정렬 사용)
def prepare_pipeline_request(case):
    # BEGIN D27-06-B01
    # TODO D27-06-B01 완료: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 1. 입력: case (필요한 값만 받습니다.)
    # 2. 절차: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 주의: role(권한)은 요청 dict 에 넣지 않음 (서버 코드가 trusted_role 로 지정)
    return {'query': case['question'],
            'history': case['history'],
            'category': case['category'],
            'candidate_k': 5,
            'context_k': 2,
            'use_rewrite': True,
            'use_multiquery': False,
            'use_rerank': True}
    # END D27-06-B01

# ============================================================
# 3. LCEL Pipeline 조립 : wire_langchain_pipeline()
# ============================================================
# wire_langchain_pipeline : 주입받은 부품으로 LCEL RAG Chain(Runnable)을 조립
# - dependencies : (rewrite, search, rerank, generate) 함수 4개 튜플 (create_dependencies() 결과)
# - offline      : True 면 고정 답변 경로, False 면 실제 LLM 생성 경로
# - 반환         : start | rewrite | multiquery | retrieval | rerank | context | generation Runnable
# 주의: 조립만 하고 실행하지 않음. 실제 Stage 실행은 chain.invoke(req) 호출 시점
def wire_langchain_pipeline(dependencies, offline):
    # BEGIN D27-06-B02
    # TODO D27-06-B02 완료: LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다.
    # 1. 입력: dependencies, offline (필요한 값만 받습니다.)
    # 2. 절차: LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/pipeline.py 의 create_rag_pipeline() 사용
    from coursekit.pipeline import create_rag_pipeline
    # create_rag_pipeline() 인자
    # *dependencies   : rewrite_fn, search_fn, rank_fn, generation_fn 순서로 펼쳐 전달
    # offline=offline : 실행 모드 (trace.source_mode 에 기록됨)
    return create_rag_pipeline(*dependencies, offline = offline)
    # END D27-06-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_06_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_06_live.json 
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 06 · 06_langchain_pipeline')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[6] = run_06() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_06() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - prepare_pipeline_request() : Pipeline 요청 생성
    # - wire_langchain_pipeline() : LCEL Pipeline 조립
    report = STAGE_RUNNERS[6](args.offline, [prepare_pipeline_request, wire_langchain_pipeline])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_06_fixture.json', 아니면 'day27_06_live.json'
    path = save_report(HERE, 'day27_06_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 06 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
