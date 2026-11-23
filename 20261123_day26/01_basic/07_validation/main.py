"""Day26 실습 7 · 기본필수. 실행 위치와 무관하게 현재 역할 폴더를 사용합니다.

실습 07 Validation · 기본필수
- 역할: 고정 bundle 세 개(정상·근거 부족·허용 밖 ID)를 receive_bundle()로 검사해
        accepted / rejected 와 오류 코드를 기록한다
- 기능: allowed_source_ids()(D26-07-B01) 실제 Context 포함 ID 집합, inspect_case()(D26-07-B02) 사례 검사 행
- 결과: outputs/validation_report.json (normal accepted, insufficient accepted, unknown_source rejected)
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
from coursekit.data import example_hit
# 제공 모듈 coursekit/checkpoints.py: 앞 실습 완성 예시 (Document 변환·Retriever·Context·Schema·검증·서비스·FastAPI·Prompt)
from coursekit.checkpoints import hits_to_documents, build_context, NO_ANSWER, RagServiceError, receive_bundle
# 제공 모듈 coursekit/fixtures.py: 고정 교육 사례 (오프라인 Chat Model·Generation·Retriever·bundle)
from coursekit.fixtures import fixture_bundle
# 제공 모듈 coursekit/reporting.py: 실행 모드·AIMessage·bundle을 JSON 기록용 dict로 바꾸기
from coursekit.reporting import execution_record

# ============================================================
# 2. 공통 고정 사례 세 개
# ============================================================
# fixture_bundle(data)로 실제 응답과 같은 모양의 {raw, parsed, parsing_error} 묶음을 만든다.
# - normal         : answered, source_ids [201]  -> 통과 예상
# - insufficient   : insufficient_context, answer = NO_ANSWER, source_ids [] -> 통과 예상
# - unknown_source : normal과 같고 source_ids만 [999] -> Context에 없는 ID라 거절 예상
def common_cases():
    # 의도적으로 만든 세 입력입니다. 실제 모델의 응답 로그가 아닙니다.
    answered = {
        "status": "answered",
        "answer": "일반 기준으로 15분까지는 지각입니다. 사전 승인을 받았다면 운영실에 확인합니다.",
        "source_ids": [201],
        "missing_information": "",
    }
    insufficient = {
        "status": "insufficient_context",
        "answer": NO_ANSWER,
        "source_ids": [],
        "missing_information": "정확한 과제 마감 날짜가 자료에 없습니다.",
    }
    # {**answered, 'source_ids': [999]} : answered 사본에서 source_ids만 교체
    unknown = {**answered, "source_ids": [999]}
    return [
        {"name": "normal", "bundle": fixture_bundle(answered)},
        {"name": "insufficient", "bundle": fixture_bundle(insufficient)},
        {"name": "unknown_source", "bundle": fixture_bundle(unknown)},
    ]

# ============================================================
# 3. 허용 출처 ID 집합
# ============================================================
# 실제 Context에 포함된 Document의 citation_id만 모은 집합을 반환한다.
# - included_docs : build_context()의 두 번째 반환값 (예: 201 한 건)
# - 반환          : {201}
# 주의: 검색 후보 전체의 ID를 허용하면 Context 밖 문서를 인용해도 통과한다
def allowed_source_ids(included_docs):
    # 입력: 실제 Context에 포함된 Document 목록 / 출력: set[int].
    # BEGIN D26-07-B01
    # TODO D26-07-B01 완료: included_docs의 metadata에서 citation_id를 모아 set으로 반환합니다.
    # 해설: 검색 후보 전체의 ID를 허용하면 Context 밖 문서를 인용할 수 있습니다.
    # build_context의 두 번째 반환값인 included_docs만 순회합니다.
    return {doc.metadata["citation_id"] for doc in included_docs}
    # END D26-07-B01

# ============================================================
# 4. 사례 한 건 검사 행
# ============================================================
# receive_bundle()로 bundle을 검사하고 결과를 JSON 기록용 행으로 만든다.
# - 검사 순서 (coursekit/checkpoints.receive_bundle):
#   1) raw AIMessage 상태 (refusal·status != 'completed'·빈 응답)
#   2) parsing_error 존재 여부
#   3) parsed가 WireAnswer인지
#   4) validate_answer 업무 규칙 (빈 답·중복 ID·허용 밖 ID·status별 조건)
# - 통과 : outcome 'accepted', result = 검증된 답변 dict, human_review_required True
# - 거절 : outcome 'rejected', error = 오류 코드 (예: 'answer_validation_error')
def inspect_case(case, provided_ids):
    # 검증 결과를 JSON으로 기록합니다. 원문 의미의 참/거짓은 자동 판정하지 않습니다.
    # BEGIN D26-07-B02
    # TODO D26-07-B02 완료: receive_bundle로 검사한 결과를 accepted/rejected 형식으로 기록합니다.
    # RagServiceError(서비스가 정의한 검증 거절)만 잡는다. 다른 프로그래밍 오류는 그대로 드러나게 둔다
    # 해설: raw 완료 상태 → WireAnswer 파싱 → 로컬 Schema/업무 규칙 순서입니다.
    # 검증 거절은 결과표에 기록하되, 예상하지 못한 프로그래밍 오류는 감추지 않습니다.
    try:
        answer = receive_bundle(case["bundle"], provided_ids)
    except RagServiceError as exc:
        # 통과 행: 기계 검사 통과일 뿐이므로 사람의 원문 대조(human_review_required)가 필요
        return {
            "case": case["name"],
            "outcome": "rejected",
            "error": str(exc),
            "result": None,
            "human_review_required": False,
        }
    return {
        "case": case["name"],
        "outcome": "accepted",
        "error": None,
        "result": answer.model_dump(),
        "human_review_required": True,
    }
    # END D26-07-B02

# ============================================================
# 5. 검증표 만들기 (run)
# ============================================================
# 출석 Hit 1건으로 Context를 만들고 허용 ID {201} 기준으로 모든 사례를 검사한다.
def run():
    # example_hit()(201) -> Document 1개 -> build_context -> Context·포함 문서
    documents = hits_to_documents([example_hit()])
    context, included_docs = build_context(documents)
    provided_ids = allowed_source_ids(included_docs)
    # 기본필수: 공통 3사례
    cases = common_cases()
    # 반환 dict 키 : 의미
    # - source_mode      : 'validation_fixtures' (고정 입력 검증표)
    # - execution        : execution_record('offline') -> db·embedding·llm 모두 False
    # - context          : 검사 기준 Context
    # - included_ids     : 허용 ID 정렬 목록 ([201])
    # - cases            : inspect_case() 행 목록
    # - grounding_review : 원문 대조는 reports/review.md에서 사람이 한다는 안내
    return {
        "source_mode": "validation_fixtures",
        "execution": execution_record("offline"),
        "context": context,
        "included_ids": sorted(provided_ids),
        "cases": [inspect_case(case, provided_ids) for case in cases],
        "grounding_review": "reports/review.md에서 숫자·조건·예외와 원문 위치를 직접 대조합니다.",
    }

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# 검증표를 outputs/validation_report.json 으로 저장하고 사례별 결과를 출력한다.
def main():
    report = run()
    path = save_json(HERE, "validation_report.json", report)
    show_success("07 · 고정 입력 검증표 (모델·DB 미실행)", path)
    # 출력 예: 'normal accepted None', 'unknown_source rejected answer_validation_error'
    for row in report["cases"]:
        print(row["case"], row["outcome"], row["error"])
    print("accepted는 기계 검사를 통과했다는 뜻입니다. 원문 대조를 마쳐야 학습이 완료됩니다.")

# ============================================================
# 7. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
