"""Day21 실습 03: 작업 분해와 정보 부족. 실행: python main.py"""
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

# 기준: Day21 PPT 21~27장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

# ============================================================
# 2. 판단 카드 분석 (필요 사실 ↔ 지원 사실)
# ============================================================
# 작성한 판단 카드 한 장에서 빠진 사실·상충을 찾아 상태와 출처를 정한다.
# - 모든 필요 사실이 지원되고 상충이 없음 : status="answered", source_ids=지원 ID
# - 빠진 사실(missing)이 하나라도 있음   : status="insufficient_context", source_ids=[]
# - 상충(conflicts)이 하나라도 있음      : status="insufficient_context", source_ids=[]
# - 주의: 자연어 질문의 의미를 자동 판단하지 않는다. 카드에 적힌 required/supported/conflicts만 처리한다
def analyze_card(card):
    # --------------------------------------------------------
    # 2-1. 필요한 사실과 지원되는 사실 구분
    # --------------------------------------------------------
    # TODO D21-03-B01: 질문이 요구하는 사실과, 제공 자료가 지원하는 사실을 구분하세요.
    # 입력·대상: 교사가 작성한 required/supported/conflicts 카드
    # 완료 기준: C01 missing=[], C02 missing=[보관 기간]
    # 핵심: 이 코드는 자유로운 질문의 의미를 자동 이해하지 않습니다. 명시된 판단 카드를 처리합니다.
    # BEGIN D21-03-B01
    # card 키 : 의미
    # - required  : 질문에 답하려면 필요한 사실 이름 목록 (예: C03 -> ["보관 위치", "보관 기간"])
    # - supported : 제공 자료가 지원하는 사실 -> 근거 ID 목록 (예: {"보관 위치": [101]})
    # - conflicts : 자료끼리 서로 다르게 말하는 사실 (예: C04 -> ["보관 위치"])
    # missing : required 중 supported에 없는 사실 (예: C02 -> ["보관 기간"], C01 -> [])
    required = list(card["required"])                               # TODO: card["required"]
    supported = card["supported"]                                   # TODO: card["supported"]
    missing = [fact for fact in required if fact not in supported]  # TODO: supported에 없는 fact인지
    conflicts = list(card["conflicts"])                             # TODO: card["conflicts"]
    # END D21-03-B01

    # --------------------------------------------------------
    # 2-2. 정보 부족 상태 선택과 결과 정리
    # --------------------------------------------------------
    # TODO D21-03-B02: 하나라도 근거가 부족하면 수업의 정보 부족 상태를 선택하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: README의 예상 결과와 출력 파일을 대조합니다.
    # 핵심: 오늘 계약은 부분 답변을 새 상태로 추가하지 않습니다. 충분한 근거가 없으면 명시적으로 멈춥니다.
    # BEGIN D21-03-B02
    status = "insufficient_context" if missing or conflicts else "answered"  # TODO: missing 또는 conflicts가 있으면 부족, 아니면 answered
    source_ids = []
    if status == "answered":
        for fact in required:
            source_ids.extend(supported[fact])  # TODO: supported[fact]
    # dict.fromkeys : 처음 나온 순서를 지키며 중복 ID 제거 (예: [101, 101, 102] -> [101, 102])
    source_ids=list(dict.fromkeys(source_ids))
    result = {"case_id":card["case_id"],"required":required,
              "missing":missing,"conflicts":conflicts,"status":status,"source_ids":source_ids,
              "basis":"teacher_authored_fact_card"}
    # END D21-03-B02
    return result

# ============================================================
# 3. 실행 흐름 (main)
# ============================================================
# 판단 카드를 분석해 정상·정보 부족 판단과 작업 지시를 outputs/에 저장한다.
# - task_cards.json       : C01(answered/[101])·C02(insufficient_context/[]) 결과
# - task_instruction.txt  : 최종 출력 지시 한 문단
#- compound_case.json    : 위치+기간을 함께 묻는 C03 결과 (응용·심화)
#- conflict_review.json  : 상충 자료 C04 결과와 코드·사람 확인 항목 (심화)
def main():
    # cards : 교사 작성 판단 카드 4장 C01~C04 (data/decomposition_cards.json)
    cards=read_json(DATA/"decomposition_cards.json")

    # --------------------------------------------------------
    # 3-1. 정상·정보 부족 판단과 작업 지시 저장
    # --------------------------------------------------------
    # TODO D21-03-B03: 정상·정보 부족의 판단과 최종 출력 지시를 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 2개 결과와 작업 지시를 저장합니다. 모델 추론 관찰 기록이 아닙니다.
    # BEGIN D21-03-B03
    # cards[:2] : C01(보관 위치) · C02(보관 기간) 두 장만 분석
    results = [analyze_card(c) for c in cards[:2]]  # TODO: analyze_card(card)
    # TODO: 사실 대조·부족 처리·최종 JSON 요구(긴 내부 사고 요구 X)
    # instruction : 사실 대조 -> 부족하면 추측 금지 -> 최종 JSON만 반환하라는 작업 지시
    instruction = "질문이 요구하는 사실과 근거를 대조하세요. 부족하면 추측하지 말고 필요한 정보를 적으세요. 내부 사고 기록 대신 최종 JSON만 반환하세요."
    assert results[0]["status"]=="answered" and results[1]["status"]=="insufficient_context"
    write_json(HERE/"outputs"/"task_cards.json", results)
    write_text(HERE/"outputs"/"task_instruction.txt", instruction)
    # END D21-03-B03
    print("[03] 판단 카드 완료. 근거 대조는 사람의 판단이 포함됩니다.")

# ============================================================
# 4. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    run(main)
