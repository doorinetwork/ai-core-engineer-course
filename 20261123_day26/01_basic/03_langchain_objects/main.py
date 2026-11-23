"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다.

실습 03 LangChain Objects · 기본필수
- 역할: Document·Retriever·Prompt·Chat Model 네 구성요소의 입력·출력 자료형과
        "데이터 읽기 vs invoke 실행" 구분을 대응표로 정리한다
- 기능: component_outputs()(D26-03-B01) 출력 자료형, access_kinds()(D26-03-B02) 접근 방식
- 결과: outputs/object_contracts.json (components 4행: input·output·access)
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
# 2. 구성요소별 입력
# ============================================================
# 구성요소
# - Document   : page_content(본문)·metadata(출처)를 담아 생성
# - Retriever  : {'query', 'category', 'candidate_k'} dict
# - Prompt     : {'question', 'context'} dict
# - Chat Model : ChatPromptValue 또는 메시지 목록
COMPONENT_INPUTS = {
    "Document": "page_content와 metadata를 담아 생성",
    "Retriever": "query·category·candidate_k가 있는 dict",
    "Prompt": "question·context가 있는 dict",
    "Chat Model": "ChatPromptValue 또는 메시지 목록",
}

# ============================================================
# 3. 구성요소별 출력 자료형
# ============================================================
# 네 구성요소가 돌려주는(또는 나타내는) 값을 대응시킨다.
def component_outputs() -> dict[str, str]:
    # 강의에서 다룬 네 구성요소의 출력 자료형을 대응시킵니다.
    # BEGIN D26-03-B01
    # TODO D26-03-B01 완료: 네 구성요소의 반환값을 구분해 dict로 반환합니다.
    # 구성요소 : 출력
    # - Document   : 본문·출처를 보관하는 데이터 객체 (스스로 답을 만들지 않음)
    # - Retriever  : list[Document] (검색 후보 여러 건)
    # - Prompt     : ChatPromptValue (System·Human 메시지 묶음, 아직 답변 아님)
    # - Chat Model : AIMessage (모델 응답. 이 단계에서만 API 요청이 발생)
    # Document는 데이터를 담는 객체이고 Retriever는 여러 Document를 반환하는 역할입니다.
    # Prompt는 질문·Context를 메시지로 구성할 뿐 답변을 생성하지 않습니다.
    # Chat Model을 실행해야 모델 응답을 담은 AIMessage를 얻습니다.
    return {
        "Document": "본문·출처를 보관하는 데이터 객체",
        "Retriever": "list[Document]",
        "Prompt": "ChatPromptValue",
        "Chat Model": "AIMessage",
    }
    # END D26-03-B01

# ============================================================
# 4. 데이터 읽기 vs invoke 실행 구분
# ============================================================
# 값을 "읽는" 구성요소와 invoke()로 "작업을 실행하는" 구성요소를 구분한다.
def access_kinds() -> dict[str, str]:
    # 데이터를 읽는 경우와 invoke로 작업을 실행하는 경우를 구분합니다.
    # BEGIN D26-03-B02
    # TODO D26-03-B02 완료: Document는 data, 실행 구성요소는 invoke로 표시합니다.
    # 구성요소 : 접근 방식
    # - Document   : 'data'   -> doc.page_content, doc.metadata 속성을 읽는다
    # - Retriever  : 'invoke' -> 검색 실행
    # - Prompt     : 'invoke' -> 메시지 구성 (모델 호출 없음)
    # - Chat Model : 'invoke' -> 답변 요청 (실제 API 호출 경계)
    # doc.page_content는 보관된 값을 읽습니다. Document 자체는 답변 생성기가 아닙니다.
    # 같은 invoke라도 Retriever는 검색, Prompt는 메시지 구성, Chat Model은 답변 요청을 합니다.
    # 이 실습은 대응표만 작성하므로 실제 invoke를 호출하지 않습니다.
    return {"Document": "data", "Retriever": "invoke", "Prompt": "invoke", "Chat Model": "invoke"}
    # END D26-03-B02

# ============================================================
# 5. 실행 흐름 (main)
# ============================================================
# 대응표를 만들어 outputs/object_contracts.json 으로 저장한다.
def main():
    outputs = component_outputs()
    access = access_kinds()
    # report 키 : 의미
    # - execution   : execution_record('concept_review') -> db·embedding·llm 모두 False
    # - components  : COMPONENT_INPUTS 순서대로 {component, input, output, access} 4행
    # - observation : 대응표만 작성했고 invoke는 실행하지 않았다는 기록
    report = {
        "execution": execution_record("concept_review"),
        "components": [
            {"component": name, "input": value, "output": outputs[name], "access": access[name]}
            for name, value in COMPONENT_INPUTS.items()
        ],
        "observation": "입력·출력 대응표; Prompt·모델 invoke는 실행하지 않음",
    }
    # HERE/outputs/object_contracts.json 저장 후 경로 반환
    path = save_json(HERE, "object_contracts.json", report)
    show_success("03 기본필수", path)


# ============================================================
# 6. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
