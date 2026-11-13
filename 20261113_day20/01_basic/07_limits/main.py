"""수업용 답변 3건을 사실·형식·근거 불일치로 분류하고 review.json/review.md로 저장한다.

- format           : 필수 필드(sources) 없음 -> F02
- evidence         : 출처 목록이 비어 있음 -> F03
- factual_mismatch : 출처는 있으나 답변이 원본과 다름 -> F01
- ok               : 위 세 가지에 해당하지 않음
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 교시 디렉토리 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 디렉토리 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 디렉토리에서 실행해도 common 패키지를 찾습니다.
# 수준 공통 모듈 common/support.py: JSON·Markdown 저장
from common.support import write_json, write_md

# ============================================================
# 2. 수업용 답변 사례 정의
# ============================================================
# 세 예시는 수업용으로 사람이 미리 설계한 사례입니다. 실제 자동 사실성 검사기가 아닙니다.
# CASES 원소 키 : 의미
# - id       : 사례 ID (F01~F03)
# - question : 사용자 질문
# - source   : 승인된 원본 문장 (지식 A/B/C)
# - answer   : 예시 LLM 답변
# - sources  : 답변이 인용한 출처 ID 목록 (키 자체가 없거나 빈 목록일 수 있음)
# 사례별 의도
# - F01 : 출처는 있으나 답변이 원본과 다름 (사실 불일치)
# - F02 : 내용은 맞지만 sources 필드 자체가 없음 (형식 부족)
# - F03 : 내용은 맞지만 sources가 빈 목록 (근거 부족)
CASES = [
    {"id":"F01","question":"파일을 어디에 보관하나요?",
     "source":"실습 파일은 개인 폴더에 보관합니다.",
     "answer":"실습 파일은 공용 서버 폴더에 보관합니다.","sources":["K01"]},
    {"id":"F02","question":"서버는 어디에서 실행하나요?",
     "source":"프로젝트 폴더에서 서버를 실행합니다.",
     "answer":"프로젝트 폴더에서 서버를 실행합니다."},
    {"id":"F03","question":"질문할 때 무엇을 함께 적나요?",
     "source":"오류 메시지와 시도한 작업을 함께 적습니다.",
     "answer":"오류 메시지와 시도한 작업을 함께 적습니다.","sources":[]},
]

# ============================================================
# 3. 실습 본문: 답변 실패 유형 분류와 대응 정리
# ============================================================
def main():
    # --------------------------------------------------------
    # 3-1. 실패 유형 분류 규칙 적용
    # --------------------------------------------------------
    # BEGIN D20-07-B01
    # TODO D20-07-B01 완료: 3개 답변에서 정해 둔 형태·근거·사실 불일치를 분류하세요.
    # 해설: 형식 누락→근거 누락→출처와 내용 불일치 순으로 교육용 규칙을 적용합니다.
    # 판정 순서 (먼저 걸리는 규칙이 우선)
    # - "sources" 키 없음           : format (형식 누락)
    # - sources가 빈 목록           : evidence (근거 누락)
    # - answer != source            : factual_mismatch (원본과 내용 불일치)
    # - 그 외                       : ok
    # 결과: [F01 factual_mismatch, F02 format, F03 evidence]
    issues=[]
    for case in CASES:
        if "sources" not in case:
            kind="format"
        elif not case["sources"]:
            kind="evidence"
        elif case["answer"] != case["source"]:
            kind="factual_mismatch"
        else:
            kind="ok"
        issues.append({"id":case["id"],"kind":kind})
    # 해설: 문자열 동일성은 실제 복잡한 주장 검증을 대신하지 않습니다.
    # END D20-07-B01

    # --------------------------------------------------------
    # 3-2. 유형별 정의 작성
    # --------------------------------------------------------
    # BEGIN D20-07-B02
    # TODO D20-07-B02 완료: 사실성·형식·근거 검증을 서로 다른 검사라고 설명하세요.
    # explanations 키 : 의미
    # - format           : 필수 필드 존재 여부를 보는 구조 검증
    # - evidence         : 출처 유무와 근거 적절성 구분
    # - factual_mismatch : 원본·정답과 모순되는지 사람·규칙으로 확인
    explanations={
        "format":"JSON/응답에 필수 필드가 존재하는지 보는 구조 검증",
        "evidence":"인용·출처 정보가 있는지와 적절한 근거인지 구분",
        "factual_mismatch":"주장과 승인된 정답/원본이 모순되는지 사람·검증 규칙으로 확인",
    }
    # END D20-07-B02
    mitigation={}
    additional_checks=[]
    agent_case={}
    trust_rules=[]

    # --------------------------------------------------------
    # 3-3. review.json·review.md 저장
    # --------------------------------------------------------
    # BEGIN D20-07-B03
    # TODO D20-07-B03 완료: 검증 결과와 향후 후속 수업 연결을 review.json 및 review.md로 저장하세요.
    # report 키 : 의미
    # - classification : 사례별 분류 결과
    # - definitions    : 유형 정의
    # - mitigation     : 대응 방법
    # - checks         : 추가 점검 항목
    # - agent_case     : Agent 권한 사례
    # - trust_rules    : 신뢰 규칙
    # - disclaimer     : 수업용 규칙이며 실제 사실성 자동 판정이 아님
    report={
        "classification":issues,
        "definitions":explanations,
        "mitigation":mitigation,
        "checks":additional_checks,
        "agent_case":agent_case,
        "trust_rules":trust_rules,
        "disclaimer":"예상 문제 유형을 분류한 수업용 규칙이며 실제 LLM 사실성 자동 판정이 아님"}
    write_json(HERE/"outputs"/"review.json",report)
    # review.md 본문: 제목 -> 사례별 "- F01: factual_mismatch" 목록 -> 플랫폼 역할 연결
    lines=["# LLM 답변 실패·검증 결과", "", "이 결과는 사전 정의된 수업용 예시를 사용했습니다.", ""]
    lines += [f"- {row['id']}: {row['kind']}" for row in issues]
    lines += ["", "## 플랫폼 연결", "FastAPI: 입력·출력 계약 / PostgreSQL: 원본 / RAG: 검색 근거 / Agent: 도구 권한(후속)"]
    write_md(HERE/"outputs"/"review.md","\n".join(lines)+"\n")
    # 콘솔 확인: [07] [('F01', 'factual_mismatch'), ('F02', 'format'), ('F03', 'evidence')]
    print("[07]",[(r["id"],r["kind"]) for r in issues])
    # END D20-07-B03

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__=="__main__":
    main()
