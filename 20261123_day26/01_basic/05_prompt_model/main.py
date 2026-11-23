"""실습 05 · Prompt 구성과 Chat Model 호출을 구분합니다.

실습 05 Prompt & Chat Model · 기본필수
- 역할: prompt.invoke()(메시지 구성, API 호출 없음)와 chat_model.invoke()(실제 모델 호출 경계)를 구분해
        출석 질문 1건의 입력 메시지와 모델 응답을 기록한다
- 기능: make_inputs()(D26-05-B01) question·context dict, invoke_pair()(D26-05-B02) ChatPromptValue·AIMessage 쌍
- 결과: outputs/prompt_model.json (prompt_value_type = 'ChatPromptValue', response.type = 'AIMessage')
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
from coursekit.data import example_hit, load_hits
# 제공 모듈 coursekit/checkpoints.py: 앞 실습 완성 예시 (Document 변환·Retriever·Context·Schema·검증·서비스·FastAPI·Prompt)
from coursekit.checkpoints import make_prompt
# 제공 모듈 coursekit/llm.py: 실제 ChatOpenAI 준비와 Prompt | Structured Model 생성 (호출 시 API 비용)
from coursekit.llm import create_chat_model
# 제공 모듈 coursekit/fixtures.py: 고정 교육 사례 (오프라인 Chat Model·Generation·Retriever·bundle)
from coursekit.fixtures import fixture_chat_model, fixture_generation
# 제공 모듈 coursekit/reporting.py: 실행 모드·AIMessage·bundle을 JSON 기록용 dict로 바꾸기
from coursekit.reporting import execution_record, message_record
# 제공 모듈 coursekit/io.py: data/ 경로·JSON 읽기·실습별 outputs/ 저장·완료 메시지
from coursekit.io import save_json, show_success


# ============================================================
# 2. Prompt 입력 dict 만들기
# ============================================================
# 질문과 근거 문자열을 Prompt 자리표시자 이름의 두 키로 묶는다.
# - question : '15분 늦으면 어떻게 기록되나요?'
# - context  : '[201] attendance_v2.md#sec-1\n수업 시작 후 15분까지 …'
# - 반환     : {'question': …, 'context': …}
# 주의: 키 철자는 make_prompt()의 {question}, {context}와 같아야 한다
def make_inputs(question: str, context: str) -> dict:
    # 입력: 질문·근거 문자열 / 출력: question, context 두 Key를 가진 dict.
    # BEGIN D26-05-B01
    # TODO D26-05-B01 완료: 질문과 Context를 각각 question·context Key로 묶어 반환합니다.
    # question은 사용자가 묻는 값이고 context는 검색 후 선택한 원문입니다.
    # 두 값을 구분해 두면 질문을 바꿀 때 원문을 덮어쓰지 않습니다.
    # Key의 철자는 make_prompt()에서 사용하는 {question}, {context}와 맞춥니다.
    return {"question": question, "context": context}
    # END D26-05-B01

# ============================================================
# 3. Prompt 실행과 모델 호출 분리 
# ============================================================
# dict -> ChatPromptValue -> AIMessage 두 단계를 차례로 실행해 둘 다 반환한다.
# - prompt     : make_prompt()의 ChatPromptTemplate (System 규칙 + Human 질문·자료)
# - chat_model : 실제 ChatOpenAI 또는 --offline 고정 Runnable(fixture_chat_model)
# - 반환       : (prompt_value, message)
# - inputs에 context 키 없음 : prompt.invoke에서 KeyError, 모델은 호출되지 않음
def invoke_pair(prompt, chat_model, inputs: dict):
    # dict → ChatPromptValue → AIMessage. 반환: (prompt_value, message).
    # BEGIN D26-05-B02
    # TODO D26-05-B02 완료: Prompt를 먼저 실행하고 그 반환값을 Chat Model에 전달한 뒤 두 반환값을 함께 반환합니다.
    # prompt.invoke(inputs) : 템플릿 자리표시자를 채운 ChatPromptValue (System·Human 메시지 2개)
    # 이 호출은 변수 값을 채워 메시지를 만들며 LLM API를 호출하지 않습니다.
    # 입력에 context가 없으면 여기서 KeyError가 발생하여 다음 줄로 가지 않습니다.
    prompt_value = prompt.invoke(inputs)
    # chat_model.invoke(prompt_value) : AIMessage 반환. 실제 경로에서는 여기서 API 요청·비용 발생
    # 위에서 완성한 메시지를 모델에 전달하는 이 지점이 실제 API 호출 경계입니다.
    # --offline에서는 같은 자리에 고정 응답용 Runnable이 들어갑니다.
    message = chat_model.invoke(prompt_value)
    # 요청 메시지와 모델 응답을 따로 보관해야 입력·출력을 비교할 수 있습니다.
    return (prompt_value, message)
    # END D26-05-B02

# ============================================================
# 4. 출처 포함 Context 블록
# ============================================================
# 실습 02의 블록 형식 '[ID] 위치\n본문' 을 Hit 한 건에 적용한다.
# - 예: '[202] assignment.txt\n과제 파일은 개인 폴더에 보관합니다. …'
def context_of(hit: dict) -> str:
    # 앞 장에서 배운 출처 포함 블록 형식입니다. 이번 실습의 제공 코드입니다.
    return f'[{hit["citation_id"]}] {hit["source_locator"]}\n{hit["body"]}'

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
# 질문별로 Prompt 메시지와 모델 응답을 기록해 outputs/prompt_model.json 으로 저장한다.
def main():
    # ----------------------------------------------------
    # 5-1. 명령행 인자와 사례 준비
    # ----------------------------------------------------
    # --offline : 고정 응답 Runnable 사용 (실제 모델 미실행). 지정하지 않으면 실제 LLM 호출
    parser = argparse.ArgumentParser(description = "Prompt → Chat Model 입력·출력 관찰")
    parser.add_argument("--offline", action = "store_true", help = "고정 응답 예시만 사용; 실제 모델 미실행")
    args = parser.parse_args()
    attendance = example_hit()
    prompt = make_prompt()
    # 사례 = (case_id, 질문, 근거 Hit). 기본 사례는 출석 15분 질문
    cases = [("attendance_15", "15분 늦으면 어떻게 기록되나요?", attendance)]
    # 모델 준비 전에 make_inputs TODO를 먼저 실행 -> 미완성이면 API 호출 전에 멈춤
    # 모델 생성·호출 전에 로컬 TODO와 입력을 먼저 확인합니다.
    prepared = [
        (case_id, make_inputs(question, context_of(hit)), hit)
        for case_id, question, hit in cases
    ]
    observations = {}

    # ----------------------------------------------------
    # 5-2. 모델 준비와 호출
    # ----------------------------------------------------
    # --offline : fixture_chat_model() (질문·자료로 고정 규칙 답변을 돌려주는 RunnableLambda)
    # 기본      : create_chat_model() (.env의 OPENAI_API_KEY·LLM_MODEL 필요). 실패해도 고정 예시로 바꾸지 않음
    # 기본 경로는 실제 모델입니다. 환경 문제가 있어도 자동으로 예시로 전환하지 않습니다.
    chat_model = fixture_chat_model() if args.offline else create_chat_model()
    records = []
    # 기록 행 키 : 의미
    # - case_id · question · context  : 사례 이름과 Prompt 입력
    # - prompt_value_type             : 'ChatPromptValue'
    # - prompt_messages               : [{type 'system', …}, {type 'human', '질문:\n…\n\n자료:\n…'}]
    # - response                      : message_record(message) -> type·text·response_metadata·usage_metadata
    # - evidence                      : 대조할 원문 (citation_id·doc_id·source_locator·body)
    # - human_evidence_review         : 'pending' (사람이 원문과 대조해야 함)
    for case_id, inputs, hit in prepared:
        prompt_value, message = invoke_pair(prompt, chat_model, inputs)
        records.append({
            "case_id": case_id,
            "question": inputs["question"],
            "context": inputs["context"],
            "prompt_value_type": type(prompt_value).__name__,
            "prompt_messages": [
                {"type": m.type, "content": m.content}
                for m in prompt_value.to_messages()
            ],
            "response": message_record(message),
            "evidence": {
                "citation_id": hit["citation_id"],
                "doc_id": hit["doc_id"],
                "source_locator": hit["source_locator"],
                "body": hit["body"],
            },
            "human_evidence_review": "pending",
        })

    # ----------------------------------------------------
    # 5-3. 결과 저장
    # ----------------------------------------------------
    # result 키 : 의미
    # - execution          : 'offline_fixture'(llm False) 또는 'live'(llm True)
    # - real_model_calls   : 실제 모델 호출 수 (offline이면 0)
    # - fixed_example_calls: 고정 예시 호출 수 (offline이면 사례 수)
    # - observations       : 실패 관찰 기록 (기본필수는 빈 dict)
    # - cases              : 위 기록 행 목록
    result = {
        "exercise": "05_prompt_model",
        "execution": execution_record(
            "offline_fixture" if args.offline else "live",
            llm = not args.offline,
        ),
        "real_model_calls": 0 if args.offline else len(records),
        "fixed_example_calls": len(records) if args.offline else 0,
        "observations": observations,
        "cases": records,
        "review_note": "문장·출처의 정답 여부는 자동 판정하지 않았습니다. reports/review.md에 원문 대조를 기록하세요.",
    }
    path = save_json(HERE, "prompt_model.json", result)
    show_success("05 Prompt·Chat Model 관찰 기록", path)
    print("실제 LLM 실행:", not args.offline, "| 원문 대조: 수강생 확인 필요")

# ============================================================
# 6. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
