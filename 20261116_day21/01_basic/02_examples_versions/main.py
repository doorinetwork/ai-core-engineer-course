"""Day21 실습 02: Zero/Few-shot·버전. 실행: python main.py"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee(또는 mentor) 폴더. coursekit/·data/가 이 위치에 있다
# sys.path.insert(0, ...) : 어느 작업 폴더에서 실행해도 coursekit 패키지를 import하도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
# 제공 모듈 coursekit/io.py: data/ 경로(DATA)·UTF-8 JSON/텍스트 저장·실행 래퍼(run) 등 파일 도구
from coursekit.io import DATA, read_json, write_json, write_text, run
# 제공 모듈 coursekit/prompting.py: Prompt·문맥·Schema·A/B 규칙을 한 묶음(bundle)으로 만드는 함수
from coursekit.prompting import make_bundle

# 기준: Day21 PPT 14~20장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
# 제공 모듈 coursekit/prompting.py: bundle과 질문 카드로 실제 Prompt 원문·문맥 JSON을 만드는 함수
from coursekit.prompting import render_prompt

# ============================================================
# 2. Few-shot 예제 두 개 작성
# ============================================================
# 평가 질문(D/H)과 분리된 가상 지식 901로 정상·정보 부족 Few-shot 예제 2개를 만든다.
# - 정상 예제   : "어디서 대여하나요?" -> answered, source_ids=[901]
# - 부족 예제   : "대여료는 얼마인가요?" -> insufficient_context, source_ids=[], missing_information="대여료"
# - 반환값      : 예제 dict 2개 목록 (키: question, context, answer)
def build_examples():
    # TODO D21-02-B01: 예제 ID 901을 사용해 정상 답과 정보 부족 답을 완성하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 정상·부족 예제 2개. 평가 질문 D/H를 Few-shot에 사용하지 않습니다.
    # 핵심: 입력 예제를 늘려도 모델 가중치가 바뀌는 것은 아닙니다.
    # BEGIN D21-02-B01
    # 예제 dict 키 : 의미
    # - question : 예제 질문 
    # - context  : 예제 질문에 붙는 지식 (901 한 건)
    # - answer   : 네 필드 계약을 지킨 모범 답 (정상 1개, 정보 부족 1개)
    example_context = [{"id":901,"content":"대여 장소는 안내 데스크입니다."}]
    # TODO: 정상 답에는 [901], 정보 부족에는 []와 부족한 사실 이름을 사용합니다.
    examples = [
        {"question":"어디서 대여하나요?","context":example_context,
         "answer":{"status":"", "answer":"", "source_ids":[901], "missing_information":""}},
        {"question":"대여료는 얼마인가요?","context":example_context,
         "answer":{"status":"insufficient_context", "answer":"제공된 지식으로는 답할 수 없습니다.", "source_ids":[], "missing_information":"대여료"}},
    ]
    # END D21-02-B01
    return examples

# ============================================================
# 3. 실행 흐름 (main)
# ============================================================
# Zero-shot·Few-shot Prompt를 만들고 예제 설계 기록을 outputs/에 저장한다.
# - examples.json / experiment.json : 예제 2개와 변경점·가설·고정 조건
# - zero_prompt.txt / few_prompt.txt : 같은 질문 D01의 예제 없음·있음 Prompt
# - examples_few_v2.json : 정상 예제 질문 표현 1개만 바꾼 새 버전 (응용·심화)
# - example_guard.json / version_guard.json : 평가 질문 누출·같은 버전 덮어쓰기 검사 (심화)
def main():
    examples=build_examples()
    # --------------------------------------------------------
    # 3-1. 같은 질문으로 Zero-shot·Few-shot Prompt 만들기
    # --------------------------------------------------------
    # TODO D21-02-B02: 같은 질문에 Zero-shot과 Few-shot Prompt를 각각 만드세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 두 Prompt가 생성됩니다. 이는 05의 근거 규칙 A/B와 별도 설계 실험입니다.
    # BEGIN D21-02-B02
    # case : 개선용 질문 D01 "파일을 어디에 보관하나요?" (selected_ids=[101])
    # make_bundle(examples=...) : Template·계약·스냅샷·A/B 규칙을 묶은 bundle 생성
    # - examples=[]       : Zero-shot (예제 칸이 "[]")
    # - examples=examples : Few-shot (예제 칸에 901 예제 2개 JSON)
    case = read_json(DATA/"dev_v1.json")[0]
    zero_bundle = make_bundle(examples=[])     # 예제 없는 입력
    few_bundle = make_bundle(examples=examples)      # TODO: examples 사용
    zero_prompt, _ = render_prompt(zero_bundle, case, "A")
    few_prompt, _ = render_prompt(few_bundle, case, "A")
    # TODO: 예제 ID 901의 포함 여부를 확인
    # 확인: 예제 ID 901은 Few-shot Prompt에만 있고 Zero-shot에는 없어야 한다
    # 두 Prompt의 차이는 [예제] 칸 하나뿐이다 (질문·지식·규칙·계약은 같음)
    assert "901" not in zero_prompt and "901" in few_prompt
    # END D21-02-B02

    # --------------------------------------------------------
    # 3-2. 변경점·가설·고정 조건 저장
    # --------------------------------------------------------
    # TODO D21-02-B03: 변경점·가설·고정 조건과 두 Prompt를 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 가설을 실제 성능 결과처럼 기록하지 않습니다.
    # BEGIN D21-02-B03
    # experiment 키 : 의미
    # - experiment   : 실험 이름 (설계 기록일 뿐 모델 실행 결과가 아님)
    # - base_version : 기준 버전 (예제 없음)
    # - new_version  : 비교 버전 (예제 2개)
    # - changed      : 바꾼 요소 한 가지 ("예제 유무")
    # - hypothesis   : 관찰할 가설 (성능 결과처럼 쓰지 않음)
    # - fixed        : 고정한 조건 (질문·knowledge_v1·answer_schema_v1·근거 규칙)
    # - source_mode  : "design_only" = 작성한 설계물
    experiment = {
        "experiment":"examples_design_only", "base_version":"examples_zero_v1", "new_version":"examples_few_v1",
        "changed":"예제 유무", "hypothesis":"정상과 정보 부족의 출력 조합을 잘 따를지 관찰한다.", 
        "fixed":["질문","knowledge_v1","answer_schema_v1","근거 규칙"], "source_mode":"design_only"
    }
    # TODO: 예제 유무를 바꿨다는 기록과 고정 조건을 작성
    write_json(HERE/"outputs"/"examples.json", examples)  # TODO: examples
    write_json(HERE/"outputs"/"experiment.json", experiment)
    write_text(HERE/"outputs"/"zero_prompt.txt", zero_prompt)  # TODO: zero_prompt
    write_text(HERE/"outputs"/"few_prompt.txt", few_prompt)   # TODO: few_prompt
    # END D21-02-B03
    print("[02] 예제·버전 설계 완료. 05는 예제 고정, 근거 규칙만 비교합니다.")

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    run(main)
