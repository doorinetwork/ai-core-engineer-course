"""
단계별 역할 분리: steps.py / 독립 실행은 main.py.
"""

# ============================================================
# 1. API 용 RAG Chain 생성 : make_service_chain() 
# ============================================================
# make_service_chain : Rewrite -> Retrieve -> Rerank -> Context -> Generation·Validation Chain 생성
# - offline=True  : 고정 검색·고정 점수·고정 답변 (DB·GPU·LLM 호출 없음)
# - offline=False : require_cuda() 확인 후 실제 E5·pgvector·CrossEncoder·LLM 준비
# - 반환          : LCEL Runnable Chain (invoke 시 실행)
def make_service_chain(offline):
    # BEGIN D27-08-I01
    # TODO D27-08-I01 완료: Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다.
    # 1. 입력: offline (필요한 값만 받습니다.)
    # 2. 절차: Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/runtime.py 의 create_demo_chain() : 의존성 생성 + create_rag_pipeline() 조립
    from coursekit.runtime import create_demo_chain
    return create_demo_chain(offline = offline)
    # END D27-08-I01

# ============================================================
# 2. FastAPI app 연결 : wire_fastapi() 
# ============================================================
# wire_fastapi : Chain 을 POST /api/v1/rag/answer 경로에 연결한 FastAPI app 을 만듦
# - chain : make_service_chain() 결과 (또는 고정 오류를 내는 Runnable)
# - 반환  : FastAPI app
#   - 요청 형식 오류 : 422 (Pydantic RagRequest 검증, Chain 호출 전)
#   - StageError     : 502 (detail 에 stage·code 만 노출)
# 주의: 요청 JSON 의 role 을 받지 않고 서버가 trusted_role='learner' 로 지정
def wire_fastapi(chain):
    # BEGIN D27-08-I02
    # TODO D27-08-I02 완료: API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다.
    # 1. 입력: chain (필요한 값만 받습니다.)
    # 2. 절차: API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다.
    # 3. 반환: 호출한 Stage가 사용할 값/상태를 반환합니다.
    # 4. 주의: 오류를 빈 결과로 숨기지 않고 출처·필터 계약을 유지합니다.
    # 제공 모듈 coursekit/api.py 의 create_app() 사용
    from coursekit.api import create_app
    # create_app() 인자
    # chain                  : 요청마다 chain.invoke(body) 실행
    # trusted_role='learner' : 서버 측 교육용 고정 역할 
    return create_app(chain, trusted_role = 'learner')
    # END D27-08-I02
