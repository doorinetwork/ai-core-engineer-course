"""Day21 실습 07: 실패 분류·최대 2회. 실행: python main.py"""
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
from coursekit.io import DATA, read_json, write_json, sha_text

# 기준: Day21 PPT 56~63장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

# 제공 모듈 coursekit/retry.py: 정해진 순서로 샘플 문자열을 돌려주는 가짜 공급기·4개 시나리오 실행 함수
from coursekit.retry import FixtureProvider, run_fixture_scenarios
# 제공 모듈 coursekit/contracts.py: 완성 검증기(문법→Schema→규칙). 07에서는 이 제공 검증기를 사용
from coursekit.contracts import validate_answer

# ============================================================
# 2. 형식 오류 전용 수정 요청 만들기
# ============================================================
# 수정 요청에는 검증에서 확인한 오류의 안전한 요약만 넣는다. 전체 traceback·경로·비밀 값·이전 원문은 넣지 않는다.
# 출처·상태 규칙 위반(rule_error)은 형식 수정으로 고치지 않고 입력·규칙을 점검한다.
def repair_prompt(original, error):
    # 허용한 형식 오류(json_error·schema_error)에만 원래 Prompt 뒤에 안전한 수정 요청 한 단락을 붙인다.
    # - "json_error"   : "JSON 객체 하나를 코드 블록 없이 출력하세요." 추가
    # - "schema_error" : "필수 필드·자료형·길이를 지키세요. ..." 추가
    # - 그 밖의 오류(unknown_source 등)·임의 문자열 : ValueError (출처 규칙 오류는 형식 수정으로 고치지 않음)
    # - 주의: 원문 응답·예외 전문·경로 같은 비표준 오류 문자열을 모델 지시로 복사하지 않는다
    # 해설 D21-07-I01: 허용한 형식 오류만 안전한 한 문장 요약으로 수정 요청을 만드세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: unknown_source는 자동 형식 수정 요청을 만들지 않습니다.
    # BEGIN D21-07-I01
    # TODO D21-07-I01 완료: 허용한 형식 오류만 안전한 한 문장 요약으로 수정 요청을 만드세요.
    # safe : 오류 이름 -> 미리 정해 둔 한 문장 안내 (허용 목록 방식)
    # repaired 예: 원래 Prompt + "\n[수정 요청] 같은 질문·지식·계약을 유지하세요. JSON 객체 하나를 코드 블록 없이 출력하세요. 부족한 사실은 만들지 마세요."
    safe = {"json_error": "JSON 객체 하나를 코드 블록 없이 출력하세요.",
            "schema_error": "필수 필드·자료형·길이를 지키세요. source_ids는 양의 정수 목록입니다."}
    if error not in safe:
        raise ValueError("형식 오류만 수정 요청을 만듭니다")
    repaired = original + "\n[수정 요청] 같은 질문·지식·계약을 유지하세요. " + safe[error] + " 부족한 사실은 만들지 마세요."
    # END D21-07-I01
    return repaired

# ============================================================
# 3. 재시도: 최초 1회 + 수정 1회 제한
# ============================================================
# 교안의 run_with_retry()와 같은 분기다. 이 코드는 history에 raw_text·prompt_sha256을 더 남긴다.
# 정상 insufficient_context는 1회로 종료한다. answered가 나올 때까지 반복하지 않는다.
def run_with_retry(generate, prompt, validate):
    # 응답 생성기를 최대 2회 호출하며 검증을 통과한 객체와 시도 기록을 돌려준다.
    # - 1회차 통과 (정상 답·정상 정보 부족) : (객체, 기록 1건) 즉시 종료
    # - 1회차 형식 오류 -> 2회차 통과        : (객체, 기록 2건)
    # - 2회 모두 형식 오류                   : (None, 기록 2건). 3회차는 없다
    # - 규칙 오류 (unknown_source 등)         : (None, 기록 1건). 형식 재시도 대상이 아니다
    # - generate : prompt -> 응답 문자열 (여기서는 FixtureProvider, 실제 LLM 아님)
    # - validate : 응답 문자열 -> (객체, 오류 이름)
    # 해설 D21-07-I02: 최초 1회+수정 1회까지만 반복하고 통과하지 못하면 None을 반환하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 호출 수가 최대 2. 정상 정보 부족도 1회에 종료.
    # 핵심: 성공 반환은 자동 구조·참조 통과일 뿐입니다. 근거를 맞게 읽었는지 사람 검토가 남습니다.
    # BEGIN D21-07-I02
    # TODO D21-07-I02 완료: 최초 1회+수정 1회까지만 반복하고 통과하지 못하면 None을 반환하세요.
    # history 원소 키 : 의미
    # - attempt       : 시도 번호 1 또는 2
    # - error         : 그 시도의 검증 결과 (통과면 None)
    # - raw_text      : 받은 응답 원문 그대로 (수정하지 않음)
    # - prompt_sha256 : 그 시도에 보낸 Prompt의 SHA-256 (수정 요청이 붙으면 값이 달라짐)
    history = []
    for attempt in (1, 2):
        raw = generate(prompt)
        obj, error = validate(raw)
        history.append({"attempt": attempt, "error": error, "raw_text": raw, "prompt_sha256": sha_text(prompt)})
        if error is None:
            return obj, history
        # 형식 오류가 아니거나 이미 2회차면 더 반복하지 않고 종료 -> 최대 호출 수 2
        # 형식 오류이고 1회차면 수정 요청을 붙인 Prompt로 한 번만 다시 호출
        if error not in {"json_error", "schema_error"} or attempt == 2:
            break
        prompt = repair_prompt(prompt, error)
    return None, history
    # END D21-07-I02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
def main():
    # 작성한 fixture로 재시도 제어 흐름을 실행하고 결과를 outputs/에 저장한다.
    # - retry_results.json   : 4개 시나리오 호출 수 [2, 2, 1, 1]·자동 통과 여부 [True, False, False, True]
    # - first_vs_final.json  : 첫 응답 오류와 최종 자동 통과를 다른 필드로 기록 (응용·심화)
    # - termination.json     : 연속 실패 뒤 수정 요청이 더 만들어지지 않는지 (응용·심화)
    # - repair_safety.json   : 비표준 오류 문자열 거부 확인 (심화)
    fixtures = read_json(DATA / "validation_fixtures.json")

    # --------------------------------------------------------
    # 4-1. 네 시나리오 호출 수 확인·기록
    # --------------------------------------------------------
    # 깨진->정상 2회 통과 / 깨진->깨진 2회 실패 / 없는 출처 1회 중단 / 정상 정보 부족 1회 종료
    # fixture 통과는 실제 모델 복구 성공률이 아니다.
    # 해설 D21-07-I03: 네 fixture 순서의 호출 수 2·2·1·1을 확인하고 기록하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 복구 성공·연속 실패·규칙 중단·부족 정상 종료 네 경우.
    # BEGIN D21-07-I03
    # TODO D21-07-I03 완료: 네 fixture 순서의 호출 수 2·2·1·1을 확인하고 기록하세요.
    # run_fixture_scenarios(fixtures, runner=run_with_retry) : 내가 만든 재시도 함수로 4개 시나리오 실행
    # - repair_success    : broken_json -> valid_answer      : 호출 2, 자동 통과
    # - repair_failure    : broken_json -> broken_json       : 호출 2, None
    # - rule_stop         : unknown_source                   : 호출 1, None (형식 재시도 안 함)
    # - insufficient_stop : valid_insufficient               : 호출 1, 정상 정보 부족 객체
    # 주의: 호출 수는 제어 흐름 확인값이며 실제 모델 복구율(actual_model_recovery_rate)은 None으로 남긴다
    results = run_fixture_scenarios(fixtures, runner = run_with_retry)
    assert [x["calls"] for x in results] == [2, 2, 1, 1]
    write_json(HERE / "outputs" / "retry_results.json", {"source_mode": "fixture", "results": results})
    # END D21-07-I03

    # --------------------------------------------------------
    # 4-2. 첫 응답과 재시도 후 결과 분리 집계
    # --------------------------------------------------------
    # 재시도 후 회복은 첫 응답 성적과 섞지 않는다. 첫 실패를 수정 응답의 성공으로 덮어쓰지 않는다.
    # 해설 D21-07-I04: 첫 응답과 재시도 후 결과를 다른 필드로 집계하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 처음 실패→나중 통과를 첫 응답 통과로 집계하지 않습니다.
    # BEGIN D21-07-I04
    # TODO D21-07-I04 완료: 첫 응답과 재시도 후 결과를 다른 필드로 집계하세요.
    # rows 원소 키 : 의미
    # - scenario      : 시나리오 이름
    # - first_error   : 첫 응답의 오류 (예: repair_success -> "json_error")
    # - attempts      : 실제 시도 수 (history 길이)
    # - final_auto_ok : 재시도까지 포함한 최종 자동 통과 여부
    # 주의: repair_success는 첫 응답 실패 + 최종 통과다. 첫 응답 통과로 세지 않는다
    rows = []
    for item in results:
        history = item["history"]
        rows.append({"scenario": item["scenario"], "first_error": history[0]["error"],
                     "attempts": len(history), "final_auto_ok": item["auto_accepted"], "source_mode": "fixture"})
    write_json(HERE / "outputs" / "first_vs_final.json", rows)
    # END D21-07-I04

    # --------------------------------------------------------
    # 4-3. 마지막 실패 뒤 수정 요청 중단 확인
    # --------------------------------------------------------
    # 마지막 실패 뒤에는 재요청 문구도 만들지 않는다.
    # 해설 D21-07-I05: 마지막 실패 뒤에는 수정 요청이 더 만들어지지 않는지 확인하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: README의 예상 결과와 출력 파일을 대조합니다.
    # 핵심: 이 범위는 완료된 문자열을 받은 뒤입니다. 공급자 거절·통신 재시도는 내일 다룹니다.
    # BEGIN D21-07-I05
    # TODO D21-07-I05 완료: 마지막 실패 뒤에는 수정 요청이 더 만들어지지 않는지 확인하세요.
    # bad : 깨진 JSON 샘플 '{"status":'
    # FixtureProvider([bad, bad]) : 두 번 모두 깨진 JSON을 돌려주는 가짜 공급기 (받은 prompt를 prompts에 기록)
    # 확인 1: 결과 None, 호출 2회
    # 확인 2: 보낸 Prompt가 2개뿐이고 두 번째 Prompt에 "[수정 요청]"이 정확히 1번만 있음 (수정 요청이 중첩되지 않음)
    bad = next(x["raw_text"] for x in fixtures if x["fixture_id"] == "broken_json")
    provider = FixtureProvider([bad, bad])
    obj, history = run_with_retry(provider, "original", lambda raw: validate_answer(raw, [101]))
    assert obj is None and provider.calls == 2
    assert len(provider.prompts) == 2
    assert provider.prompts[1].count("[수정 요청]") == 1
    write_json(HERE / "outputs" / "termination.json", {"calls": provider.calls, "success_object": None})
    # END D21-07-I05
    # 콘솔 확인 메시지
    print("[07] 제어 흐름만 검증했습니다. 실제 모델 복구율은 측정하지 않았습니다.")
