"""Day21 실습 06: JSON·Schema·참조 검증. 실행: python main.py"""
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
from coursekit.io import DATA, read_json, write_json

# 기준: Day21 PPT 46~55장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

# ============================================================
# 2. 검증용 모듈 import
# ============================================================
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
# 제공 모듈 coursekit/contracts.py: 크기·중복 키·NaN을 거부하는 JSON 파서·정보 부족 고정문
from coursekit.contracts import strict_json, INSUFFICIENT

# ============================================================
# 3. 엄격한 네 필드 계약(Pydantic 모델) 선언
# ============================================================
# status 2종·answer 1~500자·source_ids 양의 정수 최대 3개·missing_information 최대 200자, 추가 필드 금지
# schema-field 예시: [101] 통과, ["101"]·[True]·[101.0]·[0] 실패. [999]는 구조상 통과하므로 출처 범위는 규칙 검사가 맡는다
def make_model():
    # 해설 D21-06-I01: PPT 50장의 엄격한 네 필드 계약을 선언하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 문자열 ID "101", bool true, 추가 필드, 501자 답을 거부합니다.
    # 핵심: 모델 정의는 로컬 검증입니다. 실제 LLM의 생성 형식을 자동 제약하는 API 설정은 아닙니다.
    # BEGIN D21-06-I01
    # TODO D21-06-I01 완료: PPT 50장의 엄격한 네 필드 계약을 선언하세요.
    # SourceID : 출처 ID 한 개의 자료형
    # - int         : 정수만
    # - strict=True : "101"·True를 101·1로 바꾸지 않고 거부 (bool은 int의 하위형이지만 strict에서 거부)
    # - gt=0        : 0과 음수 거부 (예: 0 -> 오류)
    SourceID = Annotated[int, Field(strict = True, gt = 0)]
    # KnowledgeAnswer 필드 : 규칙
    # - status              : "answered" 또는 "insufficient_context" 두 값만 (Literal)
    # - answer              : 1~500자 문자열
    # - source_ids          : SourceID 목록, 최대 3개
    # - missing_information : 최대 200자 문자열
    # 기본값을 주지 않았으므로 네 필드 모두 필수 (빠지면 schema_error)
    class KnowledgeAnswer(BaseModel):
        # extra="forbid" : 정의하지 않은 키가 있으면 거부 / strict=True : 모든 필드에서 자동 형변환 금지
        model_config = ConfigDict(extra = "forbid", strict = True)
        status: Literal["answered", "insufficient_context"]
        answer: str = Field(min_length = 1, max_length = 500)
        source_ids: list[SourceID] = Field(max_length = 3)
        missing_information: str = Field(max_length = 200)
    # END D21-06-I01
    return KnowledgeAnswer

# ============================================================
# 4. JSON 문법 실패와 Schema 실패 분리
# ============================================================
# ValueError(JSONDecodeError 포함) -> json_error, ValidationError -> schema_error
# 중복 키·NaN/Infinity 거부 hook의 ValueError도 문법 단계(json_error)로 분류한다. 이 실습 파서는 크기 제한도 함께 둔다.
def parse_answer(raw):
    # 응답 원문 문자열을 JSON으로 읽고 모델로 검증해 (객체, 오류 이름)을 돌려준다.
    # - 깨진 JSON·코드 블록·중복 키·NaN·20,000 bytes 초과 : (None, "json_error")
    # - JSON은 맞지만 필드·자료형·길이 위반 ({}, 최상위 [], 문자열 ID "101") : (None, "schema_error")
    # - 둘 다 통과 : (KnowledgeAnswer 객체, None)
    # Model : make_model()이 만든 네 필드 계약 클래스 (호출할 때마다 새로 정의)
    Model = make_model()
    # 해설 D21-06-I02: JSON 문법 실패와 Pydantic Schema 실패를 분리하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 깨진 JSON→json_error, {} 또는 문자열 출처→schema_error.
    # 핵심: 형변환을 허용해 잘못된 출처를 억지로 정수로 만들지 않습니다.
    # BEGIN D21-06-I02
    # TODO D21-06-I02 완료: JSON 문법 실패와 Pydantic Schema 실패를 분리하세요.
    try:
        # strict_json(raw) : json.loads에 중복 키 거부·NaN/Infinity 거부·크기 제한을 더한 제공 파서
        # 예: '{"status":' -> ValueError(JSONDecodeError) -> json_error
        # RecursionError : 지나치게 깊게 중첩된 JSON도 문법 실패로 분류
        data = strict_json(raw)
    except (ValueError, RecursionError):
        return None, "json_error"
    try:
        # Model.model_validate(data) : dict를 계약대로 검사. 위반하면 ValidationError -> schema_error
        # 예: {"status":"answered","answer":"...","source_ids":["101"],"missing_information":""} -> schema_error
        obj = Model.model_validate(data)
    except ValidationError:
        return None, "schema_error"
    # END D21-06-I02
    return obj, None

# ============================================================
# 5. 출처 범위·상태 조합 규칙 검사
# ============================================================
# check_rules(중복·범위·상태 조합) + final_rules(blank_answer·부족 고정문)를 한 함수에 합친 형태다.
# 정수 목록이라도 이번에 제공한 ID인지, 본문이 주장을 뒷받침하는지는 따로 확인한다.
def check_rules(obj, provided_ids):
    # Schema를 통과한 객체가 이번 문맥의 출처 범위와 상태별 필드 조합을 지키는지 검사한다.
    # - 같은 ID 중복 [101, 101]                 : "duplicate_source"
    # - 이번에 제공하지 않은 ID [999]·예제 901 : "unknown_source"
    # - answered인데 출처 없음·missing 있음     : "status_rule"
    # - 부족인데 출처 있음·missing 공백·고정문 아님 : "status_rule"
    # - answer가 공백뿐                         : "blank_answer"
    # - 모두 통과 : None (주의: 근거 문장이 주장을 실제로 뒷받침하는지는 사람이 평가)

    # --------------------------------------------------------
    # 5-1. 중복 출처와 문맥 밖 출처 구분
    # --------------------------------------------------------
    # 해설 D21-06-I03: 중복 출처와 이번에 제공하지 않은 출처를 서로 구분하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: README의 예상 결과와 출력 파일을 대조합니다.
    # BEGIN D21-06-I03
    # TODO D21-06-I03 완료: 중복 출처와 이번에 제공하지 않은 출처를 서로 구분하세요.
    # len(ids) != len(set(ids)) : 같은 ID가 두 번 이상이면 집합 크기가 줄어듦 -> duplicate_source
    # set(ids) <= set(provided_ids) : 모든 출처가 이번에 실제 전달한 ID 안에 있는가 (부분집합)
    # 예: provided_ids=[101], source_ids=[102] -> unknown_source
    # 주의: 전체 knowledge의 ID가 아니라 이번 Prompt에 넣은 provided_ids만 기준으로 삼는다
    ids = obj.source_ids
    if len(ids) != len(set(ids)):
        return "duplicate_source"
    if not set(ids) <= set(provided_ids):
        return "unknown_source"
    # END D21-06-I03

    # --------------------------------------------------------
    # 5-2. 상태별 필드 조합·공백·고정문 검사
    # --------------------------------------------------------
    # 해설 D21-06-I04: 상태별 필드 조합과 공백·부족 고정문을 검사하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 정상 부족 응답은 오류가 아닙니다. 과도한 포기는 사람이 평가합니다.
    # BEGIN D21-06-I04
    # TODO D21-06-I04 완료: 상태별 필드 조합과 공백·부족 고정문을 검사하세요.
    # answered            : 출처가 1개 이상이고 missing_information == ""
    # insufficient_context : 출처 [] 이고 missing_information에 부족한 사실 이름 (공백만이면 위반)
    if obj.status == "answered":
        if not ids or obj.missing_information != "":
            return "status_rule"
    else:
        if ids or not obj.missing_information.strip():
            return "status_rule"
    # 공백만 있는 answer -> blank_answer (min_length=1은 공백 1글자를 통과시키므로 따로 검사)
    if not obj.answer.strip():
        return "blank_answer"
    # 부족 상태의 answer는 고정문 "제공된 지식으로는 답할 수 없습니다."와 정확히 같아야 한다
    if obj.status == "insufficient_context" and obj.answer != INSUFFICIENT:
        return "status_rule"
    # END D21-06-I04
    return None

# ============================================================
# 6. 검사 단계 연결 (문법 → Schema → 규칙)
# ============================================================
# 앞 단계가 실패하면 뒤 규칙 검사를 하지 않는다. 통과 객체도 evidence_ok는 아직 미평가다.
def validate_answer(raw, provided_ids):
    # 문법·Schema 검사(parse_answer) 뒤 규칙 검사(check_rules)를 이어 실행한다. 앞 단계에서 실패하면 바로 반환한다.
    # - 1단계 parse_answer 실패 : (None, "json_error" 또는 "schema_error")
    # - 2단계 check_rules 실패  : (None, "duplicate_source" / "unknown_source" / "status_rule" / "blank_answer")
    # - 전부 통과              : (KnowledgeAnswer 객체, None)
    obj, error = parse_answer(raw)
    if error:
        return None, error
    error = check_rules(obj, provided_ids)
    return (None, error) if error else (obj, None)

# ============================================================
# 7. 실행 흐름 (main)
# ============================================================
def main():
    # 제공 fixture 26개를 자신의 검사기로 분류하고 Schema와 검토 기록을 outputs/에 저장한다.
    # - fixture_results.json : fixture별 error·expected_error·matches_expected (26/26 일치해야 함)
    # - answer_schema.json   : 모델 정의에서 내보낸 JSON Schema
    # - semantic_review.json : 자동 통과하지만 근거가 맞지 않는 반례의 사람 검토 해설
    # - boundary_checks.json : 중복 키·NaN·크기·길이·bool ID 경계 검사
    # fixtures : 교사가 작성한 응답 샘플 26개 (실제 ChatGPT 응답 아님)
    # fixture 키 : fixture_id · raw_text(응답 원문) · provided_ids(이번 문맥 ID) · expected_error(기대 오류, 통과면 null)
    fixtures = read_json(DATA / "validation_fixtures.json")

    # --------------------------------------------------------
    # 7-1. fixture 26개 분류와 Schema 내보내기
    # --------------------------------------------------------
    # model_json_schema()는 명세 생성, model_validate()는 데이터 검사다.
    # fixture는 사람이 의도적으로 만든 입력이다. 통과 수를 실제 모델 답변 품질 점수로 바꾸지 않는다.
    # 해설 D21-06-I05: 제공 fixture를 분류하고 모델 정의에서 Schema를 내보내세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 26개 fixture가 기대 오류와 일치. semantic_counterexample은 자동 통과하지만 사람 검토 필요.
    # BEGIN D21-06-I05
    # TODO D21-06-I05 완료: 제공 fixture를 분류하고 모델 정의에서 Schema를 내보내세요.
    # results 원소 키 : 의미
    # - fixture_id       : 샘플 이름 (예: "string_id")
    # - source_mode      : "fixture" (작성한 샘플)
    # - error            : 내 검사기의 판정 (예: "schema_error", 통과면 None)
    # - expected_error   : 기대 판정
    # - matches_expected : 두 값이 같은가
    # - evidence_ok      : None = 사람 근거 평가 미실시
    results = []
    for item in fixtures:
        obj, error = validate_answer(item["raw_text"], item["provided_ids"])
        results.append({"fixture_id": item["fixture_id"], "source_mode": "fixture", "error": error,
                        "expected_error": item["expected_error"], "matches_expected": error == item["expected_error"], "evidence_ok": None})
    # 확인: 26개 모두 기대 오류와 일치해야 저장 단계로 진행
    # make_model().model_json_schema() : 모델 정의에서 JSON Schema 생성 (additionalProperties=false, required 4개)
    assert all(r["matches_expected"] for r in results)
    write_json(HERE / "outputs" / "fixture_results.json", results)
    write_json(HERE / "outputs" / "answer_schema.json", make_model().model_json_schema())
    # END D21-06-I05

    # --------------------------------------------------------
    # 7-2. 자동 검사만 통과하는 의미 반례 확인
    # --------------------------------------------------------
    # 해설 D21-06-I06: ID 101을 붙인 서버 답변이 자동 검사만 통과하는 반례를 확인하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: README의 예상 결과와 출력 파일을 대조합니다.
    # 핵심: 이 사람 예상 판단은 작성된 반례의 해설입니다. 실제 모델 평가 점수를 대신 채우지 않습니다.
    # BEGIN D21-06-I06
    # TODO D21-06-I06 완료: ID 101을 붙인 서버 답변이 자동 검사만 통과하는 반례를 확인하세요.
    # semantic_counterexample : 서버 실행 위치 답에 출처 101(파일 보관 지식)을 붙인 샘플
    # 자동 검사(문법·Schema·출처 범위)는 통과(error=None)하지만 101은 서버 실행 위치를 뒷받침하지 않는다
    # review 키 : source_mode · auto_ok(True) · human_expected("근거 부적합") · reason · model_performance_claim(False)
    # 주의: 작성된 반례의 해설이며 실제 모델 평가 점수를 대신 채우는 값이 아니다
    counter = next(x for x in fixtures if x["fixture_id"] == "semantic_counterexample")
    obj, error = validate_answer(counter["raw_text"], counter["provided_ids"])
    assert error is None
    review = {"source_mode": "fixture", "auto_ok": True, "human_expected": "근거 부적합",
              "reason": "101은 보관 위치만 설명한다. 서버 실행 위치는 102에서 확인한다.", "model_performance_claim": False}
    write_json(HERE / "outputs" / "semantic_review.json", review)
    # END D21-06-I06
    # 콘솔 확인 메시지
    print("[06] 로컬 검증 완료. 구조·참조 검사와 실제 근거 평가는 다릅니다.")
