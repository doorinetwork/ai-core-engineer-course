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
# 2. API 용 RAG Chain 생성 : make_service_chain() 
# ============================================================
# make_service_chain : Rewrite -> Retrieve -> Rerank -> Context -> Generation·Validation Chain 생성
# - offline=True  : 고정 검색·고정 점수·고정 답변 (DB·GPU·LLM 호출 없음)
# - offline=False : require_cuda() 확인 후 실제 E5·pgvector·CrossEncoder·LLM 준비
# - 반환          : LCEL Runnable Chain (invoke 시 실행)
def make_service_chain(offline):
    # BEGIN D27-08-B01
    # TODO D27-08-B01 완료: Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다.
    # 1. 입력: offline (필요한 값만 받습니다.)
    # 2. 절차: Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/runtime.py 의 create_demo_chain() : 의존성 생성 + create_rag_pipeline() 조립
    from coursekit.runtime import create_demo_chain
    return create_demo_chain(offline = offline)
    # END D27-08-B01

# ============================================================
# 3. FastAPI app 연결 : wire_fastapi() 
# ============================================================
# wire_fastapi : Chain 을 POST /api/v1/rag/answer 경로에 연결한 FastAPI app 을 만듦
# - chain : make_service_chain() 결과 (또는 고정 오류를 내는 Runnable)
# - 반환  : FastAPI app
#   - 요청 형식 오류 : 422 (Pydantic RagRequest 검증, Chain 호출 전)
#   - StageError     : 502 (detail 에 stage·code 만 노출)
# 주의: 요청 JSON 의 role 을 받지 않고 서버가 trusted_role='learner' 로 지정
def wire_fastapi(chain):
    # BEGIN D27-08-B02
    # TODO D27-08-B02 완료: API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다.
    # 1. 입력: chain (필요한 값만 받습니다.)
    # 2. 절차: API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/api.py 의 create_app() 사용
    from coursekit.api import create_app
    # create_app() 인자
    # chain                  : 요청마다 chain.invoke(body) 실행
    # trusted_role='learner' : 서버 측 교육용 고정 역할 (실무는 인증 계층이 결정)
    return create_app(chain, trusted_role = 'learner')
    # END D27-08-B02

# ============================================================
# 4. 실행 흐름 main()
# ============================================================
# main : 
# - --offline 있음 : 고정 Fixture 경로 -> outputs/day27_08_fixture.json
# - --offline 없음 : 실제 경로 -> outputs/day27_08_live.json 
def main():
    # --------------------------------------------------------
    # 명령행 인자 해석
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description = 'Day27 실습 08 · 08_enhanced_rag_api')
    # --offline : 지정하면 args.offline = True (고정 데이터·점수 경로)
    parser.add_argument('--offline', action = 'store_true', help = '교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args = parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    # --------------------------------------------------------
    # TODO 함수 주입 실행
    # --------------------------------------------------------
    # STAGE_RUNNERS[8] = run_08() 인자
    # args.offline : 실행 모드 (True = 고정 Fixture)
    # [...]        : 이 수준의 TODO 함수 2개, run_08() 가 앞에서부터 순서대로 사용 (개수가 많을수록 검사 항목 추가)
    # - make_service_chain() : API 용 RAG Chain 생성
    # - wire_fastapi() : FastAPI app 연결
    report = STAGE_RUNNERS[8](args.offline, [make_service_chain, wire_fastapi])
    # --------------------------------------------------------
    # 결과 저장·안내 출력
    # --------------------------------------------------------
    # save_report(HERE, 파일명, report) : HERE/outputs/파일명 에 JSON 저장 후 경로 반환
    # 파일명 : offline 이면 'day27_08_fixture.json', 아니면 'day27_08_live.json'
    path = save_report(HERE, 'day27_08_' + ('fixture' if args.offline else 'live') + '.json', report)
    print('실습 08 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# ============================================================
# 5. 스크립트 직접 실행
# ============================================================
# python main.py 로 실행할 때만 main() 호출 (import 할 때는 실행하지 않음)
if __name__ == '__main__':
    main()
