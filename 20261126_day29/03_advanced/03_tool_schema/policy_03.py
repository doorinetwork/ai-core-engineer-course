"""
경계값·오류/권한 정책: Advanced · Day29.

@tool 로 등록한 Tool 객체에서 모델에게 노출되는 JSON Schema 를 꺼내 확인
"""

# ============================================================
# Schema 입력 필드 이름 : schema_fields()
# ============================================================
# schema_fields : 모델에게 노출되는 Tool 입력 필드 이름을 목록으로 반환합니다.
# 메서드 체인
# - tool_obj.args_schema   : @tool 이 함수 타입 힌트로 만든 Pydantic 모델 클래스
# - .model_json_schema()   : 그 모델을 JSON Schema dict 로 변환 ({'properties': {...}, 'required': [...], ...})
# 예: register_status_tool() 결과 -> ['report_id']
def schema_fields(tool_obj) -> list[str]:
    # 해설 D29-03-A-03: 모델에게 노출될 인수 Schema와 실제 Python Tool의 실행을 별도 관찰합니다.
    # 검수 기준: report_id는 실제 DB의 양의 정수이며 task_id가 아닙니다.
    schema = tool_obj.args_schema.model_json_schema()
    # sorted(dict) : properties dict 의 키(필드 이름)만 정렬해 목록으로 만듦
    return sorted(schema.get('properties', {}))


# ============================================================
# 필수 인수 강제 : enforce_explicit_schema() 
# ============================================================
# enforce_explicit_schema : Tool JSON Schema에 report_id가 required인지 점검합니다.
# - 'report_id' 가 schema['required'] 에 없음 : ValueError('report_id 필수 인수 누락')
# - 있음 : {'name': 'lookup_status', 'required': ['report_id']}
# 주의: required 에 없으면 모델이 report_id 를 빼고 호출해도 Schema 단계에서 막지 못함
def enforce_explicit_schema(tool_obj) -> dict:
    # 해설 D29-03-A-04: 모델에게 노출될 인수 Schema와 실제 Python Tool의 실행을 별도 관찰합니다.
    # 검수 기준: report_id는 실제 DB의 양의 정수이며 task_id가 아닙니다.
    schema = tool_obj.args_schema.model_json_schema()
    if 'report_id' not in schema.get('required', []):
        raise ValueError('report_id 필수 인수 누락')
    return {'name': tool_obj.name, 'required': schema['required']}
