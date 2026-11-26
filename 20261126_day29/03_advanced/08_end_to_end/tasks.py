"""
08 · FastAPI → Agent → Tool → 응답 · 심화.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.

Agent 실행 결과(run_agent 반환 dict)를 API 응답 계약으로 바꾸고, 검증되지 않은 출처 인용은 성공으로 내보내지 않는다
"""

# ============================================================
# 기본 질문 : default_question() 
# ============================================================
# default_question : 실제 보고서 검색 + 63번 인덱싱 상태를 묻는 요청 문자열.
# 반환 예: 'RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 찾아 출처를 제시하고, 보고서 63번의 인덱싱 상태도 알려줘.'
def default_question() -> str:
    # 해설 D29-08-A-01: FastAPI는 사용자의 입력을 검증해 Agent에 전달하고 최종 결과를 반환합니다.
    # 검수 기준: 인용 ID가 실제 Tool 근거 집합 밖이면 성공 응답으로 위장하지 않습니다.
    # 제공 모듈 coursekit/live.py : 실습 공통 예시 질문을 돌려주는 demo_question()
    from coursekit.live import demo_question
    return demo_question()

# ============================================================
# API 응답 계약 : api_response() 
# ============================================================
# api_response : 성공/실패와 최종 답변·출처 검증을 API 계약으로 변환.
# - run['status'] != 'completed'            : RuntimeError('Agent did not complete')
# - run['citation_ids_valid'] 가 True 아님 : RuntimeError('UNVERIFIED_CITATION: ...')
#   (답변이 인용한 RPT-번호 중 ToolMessage 출처에 없는 번호가 있음)
# - 둘 다 통과                              : 아래 5개 키 dict
#   - status                : 'completed'
#   - answer                : 최종 AIMessage 텍스트
#   - tool_trace            : 실제 실행된 ToolMessage 기록 [{tool, tool_call_id, status}]
#   - citation_ids_valid    : True
#   - verified_evidence_ids : Tool 이 실제 제공한 출처 ID 목록 (예: ['RPT-63'])
# 주의: RuntimeError 는 starter.py 의 FastAPI 처리기에서 HTTP 502 로 바뀐다
def api_response(run: dict) -> dict:
    # 해설 D29-08-A-02: FastAPI는 사용자의 입력을 검증해 Agent에 전달하고 최종 결과를 반환합니다.
    # 검수 기준: 인용 ID가 실제 Tool 근거 집합 밖이면 성공 응답으로 위장하지 않습니다.
    # --------------------------------------------------------
    # 실패를 성공으로 내보내지 않는 검사
    # --------------------------------------------------------
    if run.get('status') != 'completed':
        raise RuntimeError('Agent did not complete')
    # 주의: 'is not True' 로 비교해 값 누락(None)도 실패로 처리
    if run.get('citation_ids_valid') is not True:
        raise RuntimeError('UNVERIFIED_CITATION: Agent가 제공되지 않은 출처 ID를 인용했습니다.')
    # --------------------------------------------------------
    # API 응답 dict (cited_source_ids·tool_call_summary 는 응답에서 제외)
    # --------------------------------------------------------
    return {
        'status': 'completed',
        'answer': run['answer'],
        'tool_trace': run['tool_trace'],
        'citation_ids_valid': run['citation_ids_valid'],
        'verified_evidence_ids': run['verified_evidence_ids'],
    }

# ============================================================
# FastAPI 앱 만들기 : make_api() 
# ============================================================
# make_api : FastAPI에 POST /api/agent/ask 를 등록합니다. 로컬 교육용·인증 미구현.
# 요청 본문 : {"question": "..."} (1~1000자, Pydantic 이 검사해 어기면 FastAPI 가 422 응답)
# 처리 흐름 : run_agent(question) -> api_response(결과) -> JSON 응답
# 예외 -> HTTP 상태 코드 (detail 은 고정 문구라 예외 원문이 밖으로 나가지 않음)
# - ValueError      : 422 '잘못된 입력'
# - PermissionError : 403 '접근 정책 위반' (허용 Category 밖)
# - RuntimeError    : 502 'Agent 결과/출처 계약 검증 실패' (api_response 의 UNVERIFIED_CITATION 포함)
# - 그 밖의 예외    : 503 'Agent/DB/GPU/LLM 실행 실패'
# 주의: except 순서가 중요하다. Exception 을 먼저 쓰면 아래 구체적 예외가 모두 503 이 된다
def make_api():
    # 해설 D29-08-A-03: FastAPI는 사용자의 입력을 검증해 Agent에 전달하고 최종 결과를 반환합니다.
    # 검수 기준: 인용 ID가 실제 Tool 근거 집합 밖이면 성공 응답으로 위장하지 않습니다.
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    # 제공 모듈 coursekit/agent.py : 실제 Agent 를 실행하고 출처 검증 결과를 붙여 주는 run_agent()
    from coursekit.agent import run_agent

    # --------------------------------------------------------
    # 요청 본문 Schema
    # --------------------------------------------------------
    # Ask : question 문자열 하나만 받는 요청 모델 (min_length=1 : 빈 문자열 거절, max_length=1000 : 긴 입력 거절)
    class Ask(BaseModel):
        question: str = Field(min_length = 1, max_length = 1000)

    app = FastAPI(title = 'Day29 Existing Report DB Agent', version = '1.0')

    # --------------------------------------------------------
    # POST /api/agent/ask 처리기
    # --------------------------------------------------------
    # api_response : 이 파일 위쪽에 정의된 응답 계약 함수 (같은 모듈이라 바로 호출 가능)
    @app.post('/api/agent/ask')
    def ask(body: Ask):
        try:
            return api_response(run_agent(body.question))
        # raise ... from exc : 원래 예외를 __cause__ 로 남겨 서버 로그에서 원인을 추적
        except ValueError as exc:
            raise HTTPException(status_code = 422, detail = '잘못된 입력') from exc
        except PermissionError as exc:
            raise HTTPException(status_code = 403, detail = '접근 정책 위반') from exc
        except RuntimeError as exc:
            raise HTTPException(status_code = 502, detail = 'Agent 결과/출처 계약 검증 실패') from exc
        except Exception as exc:
            raise HTTPException(status_code = 503, detail = 'Agent/DB/GPU/LLM 실행 실패') from exc

    return app

# ============================================================
# 4. 추가 검증 모듈 연결
# ============================================================
# 추가 구현: 같은 프로젝트의 별도 책임 모듈
from policy_08 import audit_response
