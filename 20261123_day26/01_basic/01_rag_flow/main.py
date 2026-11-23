"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다.

실습 01 RAG Flow · 기본필수
- 역할: RAG 다섯 단계(검색 -> 근거 선택 -> 메시지 작성 -> 생성 -> 확인)의 순서를 정리하고,
        출석 문서(201) 한 건과 "15분 늦으면?" 질문의 판단을 사람이 읽은 기록으로 남긴다
- 기능: stage_order()(D26-01-B01) 단계 순서, evidence_note()(D26-01-B02) 근거·판단 기록
- 결과: outputs/rag_flow.json (stages 5행, case.judgment = '지각')
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# ROLE_ROOT : HERE 기준 두 단계 위 = 역할 폴더(mentee). coursekit/·data/가 있는 곳
ROLE_ROOT = HERE.parents[1]
# 역할 폴더를 import 경로 맨 앞에 넣어 어느 위치에서 실행해도 coursekit을 찾는다
sys.path.insert(0, str(ROLE_ROOT))
# 제공 모듈 coursekit/io.py: data/ 경로·JSON 읽기·실습별 outputs/ 저장·완료 메시지
from coursekit.io import save_json, show_success
# 제공 모듈 coursekit/data.py: 강의 예시 Hit 3건(lecture_hits.json)·질문 5개 읽기
from coursekit.data import example_hit, load_hits
# 제공 모듈 coursekit/reporting.py: 실행 모드·AIMessage·bundle을 JSON 기록용 dict로 바꾸기
from coursekit.reporting import execution_record

# ============================================================
# 2. 단계 카드
# ============================================================
# 다섯 단계 이름 -> 입력·출력 설명.
# - Retriever               : 질문·검색 조건 -> 관련 청크 후보
# - Context Builder         : 후보 목록 -> 선택한 근거·포함 ID
# - Prompt                  : 규칙·질문·Context -> 모델이 읽을 메시지
# - Generator               : 모델 입력 -> 생성된 답변
# - Validator · Source 연결 : 생성 결과·허용 ID -> 검증 결과·원문 위치
# 입력·출력 설명은 제공됩니다. TODO에서는 올바른 처리 순서를 연결합니다.
STAGE_CARDS = {
    "Generator": {"input": "모델 입력", "output": "생성된 답변"},
    "Prompt": {"input": "규칙·질문·Context", "output": "모델이 읽을 메시지"},
    "Retriever": {"input": "질문·검색 조건", "output": "관련 청크 후보"},
    "Validator · Source 연결": {"input": "생성 결과·허용 ID", "output": "검증 결과·원문 위치"},
    "Context Builder": {"input": "후보 목록", "output": "선택한 근거·포함 ID"},
}

# ============================================================
# 3. RAG 단계 순서 정하기
# ============================================================
# STAGE_CARDS의 다섯 이름을 처리 순서대로 담은 목록을 반환한다.
# - 반환 : ['Retriever', 'Context Builder', 'Prompt', 'Generator', 'Validator · Source 연결']
def stage_order() -> list[str]:
    # STAGE_CARDS의 이름 다섯 개를 처리 순서대로 반환합니다.
    # BEGIN D26-01-B01
    # TODO D26-01-B01 완료: 다섯 단계의 이름을 처리 순서대로 반환합니다.
    # 검색 -> 근거 선택 -> 메시지 작성 -> 생성 -> 확인 순서
    # 검색 후보를 먼저 얻고 이번 답변에 넣을 근거를 고릅니다.
    # Prompt는 그 근거와 질문·규칙을 메시지로 만들고, Generator가 답을 생성합니다.
    # 마지막 확인은 반환값 검사와 원문 위치 연결입니다. 실제 함수 실행을 흉내 낸 것이 아닌 역할 정리표입니다.
    return ["Retriever", "Context Builder", "Prompt", "Generator", "Validator · Source 연결"]
    # END D26-01-B01

# ============================================================
# 4. 출석 근거와 판단 기록
# ============================================================
# 입력 Hit의 출처·본문을 그대로 옮기고, "15분 늦으면?" 질문에 대해 사람이 읽은 판단을 붙인다.
# - hit : data/lecture_hits.json 첫 Hit (citation_id 201, DOC-ATTEND v2, attendance_v2.md#sec-1)
# - 반환 : 출처 4값 + 원문 본문 + 질문·판단·확인할 예외를 담은 dict
def evidence_note(hit: dict) -> dict:
    # 출석 문서와 사람이 읽은 판단을 함께 기록합니다. 모델 호출 결과가 아닙니다.
    # BEGIN D26-01-B02
    # TODO D26-01-B02 완료: 입력 Hit의 출처·본문과 15분 질문의 판단·예외를 한 dict로 묶습니다.
    # 반환 dict 키 : 의미
    # - citation_id        : 출처 번호 (예: 201). 입력 값을 그대로 보존
    # - doc_id             : 문서 ID (예: 'DOC-ATTEND')
    # - source_version     : 문서 버전 (예: 'v2')
    # - source_locator     : 원문 위치 (예: 'attendance_v2.md#sec-1')
    # - evidence           : 원문 본문 전체. 숫자 15만 떼어 적지 않고 "까지"·"초과" 경계를 함께 남김
    # - question           : 판단 대상 질문
    # - judgment           : 원문 "15분까지 입실하면 지각" -> '지각'
    # - exception_to_check : 원문 "단, 사전 승인을 받은 경우에는 운영실에 확인" -> 확인할 예외
    # 숫자 15만 떼어 적으면 “까지”와 “초과”의 경계가 사라집니다. 본문 전체를 근거로 남깁니다.
    # 판단은 강의 문서를 읽어 정리한 교육용 기록이며 실제 LLM 생성 결과가 아닙니다.
    # 출처 ID·문서·버전·위치는 입력값을 그대로 보존합니다.
    return {
        "citation_id": hit["citation_id"],
        "doc_id": hit["doc_id"],
        "source_version": hit["source_version"],
        "source_locator": hit["source_locator"],
        "evidence": hit["body"],
        "question": "15분 늦으면 어떻게 기록되나요?",
        "judgment": "지각",
        "exception_to_check": "사전 승인을 받은 경우 운영실 확인",
    }
    # END D26-01-B02

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
# 단계 순서와 출석 판단 기록을 모아 outputs/rag_flow.json 으로 저장한다.
def main():
    # 단계 이름 목록을 먼저 얻어 둔다
    order = stage_order()
    # report 키 : 의미
    # - execution   : execution_record('concept_review') -> db·embedding·llm 모두 False
    # - stages      : 순서대로 {stage, input, output} 행 (STAGE_CARDS 값을 **로 펼침)
    # - case        : evidence_note(example_hit()) 결과
    # - observation : 이 기록의 성격
    report = {
        "execution": execution_record("concept_review"),
        "stages": [{"stage": name, **STAGE_CARDS[name]} for name in order],
        "case": evidence_note(example_hit()),
        "observation": "사람이 강의 원문을 읽고 작성한 역할·근거 확인표",
    }
    # save_json(폴더, 파일명, dict) : HERE/outputs/rag_flow.json 저장 후 경로 반환
    path = save_json(HERE, "rag_flow.json", report)
    show_success("01 Basic", path)

# ============================================================
# 6. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
