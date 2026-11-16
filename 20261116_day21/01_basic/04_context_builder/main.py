"""Day21 실습 04: Context 선택·예산. 실행: python main.py"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
import json

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee(또는 mentor) 폴더. coursekit/·data/가 이 위치에 있다
# sys.path.insert(0, ...) : 어느 작업 폴더에서 실행해도 coursekit 패키지를 import하도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
# 제공 모듈 coursekit/io.py: data/ 경로(DATA)·UTF-8 JSON/텍스트 저장·실행 래퍼(run) 등 파일 도구
from coursekit.io import DATA, read_json, write_json, write_text, run

# 기준: Day21 PPT 28~36장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

# ============================================================
# 2. 선택 ID 순서대로 문맥 고르기·직렬화
# ============================================================
def select_context(rows, selected_ids):
    # 지식 스냅샷에서 선택 ID의 지식을 요청 순서대로 골라 목록과 JSON 문자열로 돌려준다.
    # - 정상 선택          : [102] -> 지식 102 한 건
    # - 중복 선택          : [102, 102, 101] -> [102, 101] (첫 순서 유지, 중복 제거)
    # - 스냅샷에 중복 ID   : ValueError("중복 ID를 확인하세요")
    # - 없는 ID 선택       : [999] -> ValueError("선택한 ID가 없습니다")
    # TODO D21-04-B01: 선택 ID의 순서를 보존해 지식을 골라 JSON으로 직렬화하세요.
    # 입력·대상: knowledge_v1 목록, selected_ids=[102]
    # 완료 기준: 102만 포함한 JSON. 없는 ID는 ValueError. 순서가 바뀌면 문맥도 달라집니다.
    # 핵심: ID 선택과 직렬화일 뿐 의미 검색·RAG를 구현한 것이 아닙니다.
    # BEGIN D21-04-B01
    # by_id : ID -> 지식 dict (예: {101: {...}, 102: {...}, 103: {...}})
    # len(by_id) != len(rows) : 같은 ID가 두 번 있으면 dict 크기가 줄어드므로 스냅샷 중복을 잡아냄
    by_id = {row["id"]:row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("중복 ID를 확인하세요")
    # dict.fromkeys : 요청 순서를 지키며 중복 선택 제거 (예: [102, 102, 101] -> [102, 101])
    ids = list(dict.fromkeys(selected_ids))  # 중복 선택은 첫 순서만 남김
    if any(i not in by_id for i in ids):
        raise ValueError("선택한 ID가 없습니다")
    # chosen       : 선택 순서대로 고른 지식 목록
    # context_json : ensure_ascii=False로 한글을 그대로 둔 JSON 문자열 (Prompt의 [지식 데이터] 칸에 들어감)
    # 예: [{"id": 102, "title": "서버 실행", "content": "프로젝트 폴더에서 서버를 실행합니다.", "category": "environment"}]
    chosen = [by_id[i] for i in ids]                    # TODO: by_id[knowledge_id]
    context_json = json.dumps([], ensure_ascii=False)  # TODO: chosen
    # END D21-04-B01
    return chosen, context_json

# ============================================================
# 3. 가상 token 예산 계산
# ============================================================
def calculate_budget():
    # TODO D21-04-B02: PPT의 가상 token 예산에서 근거 문맥에 남는 공간을 구하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: available_for_context=4936
    # 핵심: 실제 모델 최대 길이나 측정 결과가 아닙니다. 입력 형식을 포함한 실제 token 측정은 별도입니다.
    # BEGIN D21-04-B02
    # parts 키 : 의미 (단위: 가상 token)
    # - window                : 한 요청 전체 창 크기 8192
    # - instructions_question : 지시문·질문 800
    # - examples              : Few-shot 예제 600
    # - output_reserve        : 답변 출력용으로 비워 두는 몫 1600
    # - margin                : 형식·추정 오차 여유 256
    parts = {"window":8192,"instructions_question":800,"examples":600,"output_reserve":1600,"margin":256}
    # TODO: window에서 지시·질문, 예제, 출력 예약, 여유를 뺍니다.
    available = parts["window"] - parts["instructions_question"]-parts["examples"]-parts["output_reserve"]-parts["margin"]  
    result = {"source":"hypothetical_budget_not_model_measurement", "parts":parts, "available_for_context":available}
    # TODO: 결과 4936과 비교. len(text)를 token 수라고 부르지 마세요.
    assert available==4936
    # END D21-04-B02
    return result

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 질문의 문맥을 고르고 포함·제외 기록과 예산을 outputs/에 저장한다.
# - context.json / context_manifest.json : 실제 전달 문맥 원문 / 포함 [102]·제외 101·103·예산
# - selection_checks.json / context_policy.json : 중복·오류 ID 검사 / 선정 정책 (응용·심화)
# - context_hash.json : canonical JSON과 SHA-256 해시 (심화)
# rows : 지식 스냅샷 101~103 (3건)
# select_context(rows, [102]) : 서버 실행 질문에 필요한 102만 선택
def main():
    rows=read_json(DATA/"knowledge_v1.json")
    chosen, context_json=select_context(rows,[102])
    # --------------------------------------------------------
    # 4-1. 포함·제외 ID와 원문·예산 저장
    # --------------------------------------------------------
    budget=calculate_budget()
    # TODO D21-04-B03: 포함·제외 ID와 원문·예산을 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 포함 [102], 제외 101·103. 실제 입력 원문을 보존합니다.
    # BEGIN D21-04-B03
    # included_ids : 포함한 ID [102]
    # excluded     : 선택하지 않은 101·103과 제외 이유
    # manifest 키 : context_version(knowledge_v1) · included_ids · excluded · budget
    # context.json 에는 Prompt에 실제로 들어가는 문맥 문자열을 그대로 저장한다
    included_ids = [r["id"] for r in chosen]  # TODO: chosen의 id 목록
    excluded = [{"id":r["id"],"reason":"이번 서버 실행 질문에서 선택하지 않음"} for r in rows if r["id"] not in included_ids]   # TODO: 제외 이유

    manifest = {"context_version":"knowledge_v1", "included_ids":included_ids, "excluded":excluded, "budget":budget}
    write_text(HERE/"outputs"/"context.json", context_json)  # TODO: context_json
    write_json(HERE/"outputs"/"context_manifest.json", manifest)
    # END D21-04-B03
    print("[04] Context와 가상 예산 저장 완료. token 실측·검색 결과가 아닙니다.")

# ============================================================
# 5. 스크립트 실행
# ============================================================
if __name__ == "__main__":
    run(main)
