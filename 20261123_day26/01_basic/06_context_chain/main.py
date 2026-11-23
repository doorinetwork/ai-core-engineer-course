"""실습 06 · Context·출처를 보관하고 Runnable Chain을 실행합니다.

실습 06 Context & Runnable Chain · 기본필수
- 역할: Document 목록으로 Context·포함 문서·포함 ID를 만들고, prompt | structured_model 체인을 실행해
        raw·parsed·parsing_error 묶음(bundle)을 관찰한다
- 기능: prepare_context()(D26-06-B01) build_context 결과 정리, run_generation()(D26-06-B02) 체인 연결·실행
- 결과: outputs/context_chain.json (included_ids [201, 202], observed_source_ids [201])
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import argparse
import sys
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent

# 어느 폴더에서 실행해도 이 역할 폴더의 coursekit을 사용합니다.
# HERE.parents[1] : <수준>/<실습>_주석 기준 두 단계 위 = mentee 역할 폴더
# -> import 경로 맨 앞에 넣어 이 역할 폴더의 coursekit 패키지를 불러온다
sys.path.insert(0, str(HERE.parents[1]))
# 제공 모듈 coursekit/data.py: 강의 예시 Hit 3건(lecture_hits.json)·질문 5개 읽기
from coursekit.data import load_hits
# 제공 모듈 coursekit/checkpoints.py: 앞 실습 완성 예시 (Document 변환·Retriever·Context·Schema·검증·서비스·FastAPI·Prompt)
from coursekit.checkpoints import hits_to_documents, build_context, make_prompt, WireAnswer
# 제공 모듈 coursekit/llm.py: 실제 ChatOpenAI 준비와 Prompt | Structured Model 생성 (호출 시 API 비용)
from coursekit.llm import create_chat_model
# 제공 모듈 coursekit/fixtures.py: 고정 교육 사례 (오프라인 Chat Model·Generation·Retriever·bundle)
from coursekit.fixtures import fixture_generation
# 제공 모듈 coursekit/reporting.py: 실행 모드·AIMessage·bundle을 JSON 기록용 dict로 바꾸기
from coursekit.reporting import execution_record, bundle_record
# 제공 모듈 coursekit/io.py: data/ 경로·JSON 읽기·실습별 outputs/ 저장·완료 메시지
from coursekit.io import save_json, show_success

# ============================================================
# 2. Context·포함 문서·Prompt 입력 준비 
# ============================================================
# build_context()로 Context를 만들고, 실제 포함된 문서에서만 허용 출처 ID를 얻는다.
# - documents  : 학사 Document 2개 (201 출석, 202 과제)
# - max_items  : 포함 문서 수 상한 (기본 2) / max_chars : Context 전체 글자 수 상한 (기본 600)
# - 반환 dict 키 : 의미
#   - context       : '[201] …\n\n[202] …' 문자열
#   - included_docs : 실제 포함된 Document 목록 (출처 연결용 객체 보관)
#   - provided_ids  : [201, 202] (검색 후보 전체가 아니라 포함 문서의 ID)
#   - inputs        : Prompt 입력 {'question', 'context'} (문자열만)
def prepare_context(documents, question: str, max_items = 2, max_chars = 600) -> dict:
    # Document 목록 → Context·포함 문서·포함 ID·Prompt 입력. 원문은 바꾸지 않습니다.
    # BEGIN D26-06-B01
    # TODO D26-06-B01 완료: build_context로 Context와 포함 문서를 얻고, 포함 ID 및 Prompt 입력을 함께 반환합니다.
    # build_context 인자 (coursekit/checkpoints.py 제공 함수)
    # - documents           : 후보 Document 목록
    # - max_items=max_items : 포함 문서 수 상한 (1~3만 허용)
    # - max_chars=max_chars : 블록 사이 빈 줄까지 포함한 전체 글자 수 상한
    # 반환: (Context 문자열, 포함 Document 목록). 중복 chunk_id·상한 초과 블록은 통째로 제외
    # 검색 후보 전체와 최종 Context에 포함한 문서는 다를 수 있습니다.
    # 제공 함수는 중복 chunk_id를 제외하고 문자 상한에 맞는 전체 블록만 넣습니다.
    context, included_docs = build_context(documents, max_items = max_items, max_chars = max_chars)
    # 허용 출처의 후보는 실제 포함한 문서에서만 얻습니다. 원래 후보 전체를 쓰지 않습니다.
    provided_ids = [doc.metadata["citation_id"] for doc in included_docs]
    # Prompt에는 문자열만 전달하고, 출처를 연결할 Document 객체는 별도로 보관합니다.
    return {
        "context": context,
        "included_docs": included_docs,
        "provided_ids": provided_ids,
        "inputs": {"question": question, "context": context},
    }
    # END D26-06-B01

# ============================================================
# 3. Prompt | Structured Model 체인 실행 
# ============================================================
# 두 Runnable을 | 로 연결해 inputs 한 번으로 실행하고 bundle을 반환한다.
# - prompt           : make_prompt() ChatPromptTemplate
# - structured_model : with_structured_output(WireAnswer, include_raw=True) 또는 fixture_generation()
# - 반환 bundle      : {'raw': AIMessage, 'parsed': WireAnswer 또는 None, 'parsing_error': None 또는 예외}
def run_generation(prompt, structured_model, inputs: dict):
    # Prompt → Structured Model의 bundle. bundle 검증·승인은 다음 실습입니다.
    # BEGIN D26-06-B02
    # TODO D26-06-B02 완료: Prompt와 Structured Model을 |로 연결하고 inputs로 실행하여 bundle을 반환합니다.
    # prompt | structured_model : 앞 Runnable 출력(ChatPromptValue)을 뒤 Runnable 입력으로 넘기는 RunnableSequence
    # |는 첫 Runnable의 반환값을 다음 Runnable에 연결합니다.
    # 첫 단계는 ChatPromptValue를 만들고 둘째 단계는 raw/parsed/parsing_error를 반환합니다.
    generation = prompt | structured_model
    # 두 호출을 순서대로 실행합니다. 라이브 경로에서 LLM API 요청은 여기서 일어납니다.
    return generation.invoke(inputs)
    # END D26-06-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 학사 Document로 Context를 준비하고 체인을 실행해 bundle을 outputs/context_chain.json 으로 기록한다.
def main():
    # ----------------------------------------------------
    # 4-1. 명령행 인자와 입력 준비
    # ----------------------------------------------------
    # --offline : fixture_generation() 고정 bundle 사용. 지정하지 않으면 실제 Structured Output 호출
    parser = argparse.ArgumentParser(description = "Context → Prompt → Structured Model 관찰")
    parser.add_argument("--offline", action = "store_true", help = "고정 bundle 예시만 사용; 실제 모델 미실행")
    args = parser.parse_args()
    # category == '학사' 인 Hit만 고름 -> 201(출석), 202(과제). 203(시설)은 제외
    # 앞 실습의 미완성 코드에 의존하지 않는 제공된 Document 변환 Checkpoint입니다.
    academic_hits = [hit for hit in load_hits() if hit["category"] == "학사"]
    # 제공 Checkpoint(실습 04 완성 예시)로 Document 변환
    documents = hits_to_documents(academic_hits)
    question = "15분 늦으면 어떻게 기록되나요?"
    prepared = prepare_context(documents, question)
    experiments = {}
    prompt = make_prompt()

    # ----------------------------------------------------
    # 4-2. 모델 준비 (offline 고정 bundle 또는 실제 Structured Output)
    # ----------------------------------------------------
    # 실제 경로의 with_structured_output 인자
    # - WireAnswer          : 응답 JSON Schema (status·answer·source_ids·missing_information)
    # - method='json_schema': Provider의 JSON Schema Structured Output 사용
    # - strict=True         : Schema 밖의 키·자료형을 허용하지 않음
    # - include_raw=True    : 결과를 {'raw', 'parsed', 'parsing_error'} 묶음으로 반환
    if args.offline:
        # 이 Runnable은 제공된 고정 bundle을 반환합니다. 실제 모델 실행은 아닙니다.
        structured_model = fixture_generation()
    else:
        # 라이브 모델 설정과 Schema를 준비하는 단계입니다. 아직 API를 호출하지 않습니다.
        chat_model = create_chat_model()
        structured_model = chat_model.with_structured_output(
            WireAnswer,
            method = "json_schema",
            strict = True,
            include_raw = True,
        )

    # ----------------------------------------------------
    # 4-3. 체인 실행과 관찰 기록
    # ----------------------------------------------------
    bundle = run_generation(prompt, structured_model, prepared["inputs"])
    if bundle is None:
        raise AssertionError("이번 정상 예시의 Context는 비어 있지 않아야 합니다.")
    # parsed가 None(파싱 실패)이면 인용 ID도 빈 목록으로 기록
    parsed = bundle.get("parsed")
    cited_ids = list(parsed.source_ids) if parsed is not None else []

    # ----------------------------------------------------
    # 4-4. 결과 저장
    # ----------------------------------------------------
    # result 키 : 의미
    # - real_model_calls          : offline 0, 실제 1
    # - context · included_ids    : 모델에 보낸 Context와 포함 ID ([201, 202])
    # - included_sources          : 포함 문서의 metadata 사본
    # - bundle                    : bundle_record() -> raw(AIMessage 요약)·parsed·parsing_error
    # - observed_source_ids       : 모델이 인용한 ID (예: [201])
    # - source_id_membership_only : 인용 ID ⊆ 포함 ID 여부만 본 값 (원문 의미 검증 아님)
    # - experiments               : 비교 실험 결과
    result = {
        "exercise": "06_context_chain",
        "execution": execution_record(
            "offline_fixture" if args.offline else "live",
            llm = not args.offline,
        ),
        "real_model_calls": 0 if args.offline else 1,
        "context": prepared["context"],
        "included_ids": prepared["provided_ids"],
        "included_sources": [dict(doc.metadata) for doc in prepared["included_docs"]],
        "prompt_input": prepared["inputs"],
        "bundle": bundle_record(bundle),
        "observed_source_ids": cited_ids,
        "source_id_membership_only": None if parsed is None else set(cited_ids).issubset(prepared["provided_ids"]),
        "experiments": experiments,
        "human_evidence_review": "pending",
        "review_note": "bundle·포함 ID를 관찰한 기록입니다. 자료형·상태·출처의 승인 검사는 실습07에서 진행합니다.",
    }
    path = save_json(HERE, "context_chain.json", result)
    show_success("06 Context·Runnable 관찰 기록", path)
    print("실제 LLM 실행:", not args.offline, "| 검증 승인 전 bundle 관찰")

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
