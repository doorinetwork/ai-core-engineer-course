"""
단계별 역할 분리: steps.py / 독립 실행은 main.py.
"""

# ============================================================
# 1. Pipeline 요청 생성 : prepare_pipeline_request() 
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
    # BEGIN D27-06-I01
    # TODO D27-06-I01 완료: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
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
    # END D27-06-I01

# ============================================================
# 2. LCEL Pipeline 조립 : wire_langchain_pipeline() 
# ============================================================
# wire_langchain_pipeline : 주입받은 부품으로 LCEL RAG Chain(Runnable)을 조립
# - dependencies : (rewrite, search, rerank, generate) 함수 4개 튜플 (create_dependencies() 결과)
# - offline      : True 면 고정 답변 경로, False 면 실제 LLM 생성 경로
# - 반환         : start | rewrite | multiquery | retrieval | rerank | context | generation Runnable
# 주의: 조립만 하고 실행하지 않음. 실제 Stage 실행은 chain.invoke(req) 호출 시점
def wire_langchain_pipeline(dependencies, offline):
    # BEGIN D27-06-I02
    # TODO D27-06-I02 완료: LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다.
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
    # END D27-06-I02
