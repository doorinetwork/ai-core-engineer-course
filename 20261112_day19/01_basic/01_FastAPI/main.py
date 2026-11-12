"""
    Day19 오전 실습 1 · 기본필수 지식 CRUD API. 
    실행: python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""
# ============================================================
# 1. 라이브러리·모듈 불러오기
# ============================================================
from fastapi import FastAPI, HTTPException, Response
from storage import LOCK, read_data, write_data

from pydantic import BaseModel, Field

# ============================================================
# 2. 입력·응답 모델 정의
# ============================================================
# KnowledgeWrite : 입력 모델
# 제약조건
#   title    : 1~80자   
#   content  : 1~2000자 
#   category : 1~30자   
class KnowledgeWrite(BaseModel):
    # 세 필드는 모두 필수입니다. 길이 규칙은 공백 제거 규칙과 다릅니다.
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

# Knowledge : 응답 모델 = 입력 세 필드 + 서버가 발급한 id
# 예: {"id": 4, "title": "보충 학습", "content": "...", "category": "learning"}
class Knowledge(KnowledgeWrite):
    id: int

# ============================================================
# 3. FastAPI 앱 생성과 상태 확인 경로
# ============================================================
# title="Day19 AM Knowledge API - Basic" : /docs 상단에 보이는 API 이름
# app : uvicorn main:app 의 "app"
app = FastAPI(title="Day19 AM Knowledge API - Basic")

# GET /health : 서버 상태 확인
@app.get("/health")
def health():
    # 서버 응답 여부 반환
    return {"status": "ok"}

# ============================================================
# 4. 조회 (GET)
# ============================================================
# response_model=list[Knowledge] : 응답을 Knowledge 목록 형태로 /docs 스키마에 표시
@app.get("/api/knowledge", response_model=list[Knowledge])
# 저장 파일의 지식 목록을 id 오름차순으로 반환한다.
def list_knowledge():
    # 읽기 -> 수정 -> 저장을 같은 LOCK 안에서 수행해 중간에 다른 요청이 끼어들지 않게 한다.
    with LOCK:
        data = read_data()
    # BEGIN B-API-01
    # TODO B-API-01: data의 entries를 id 오름차순으로 반환하세요. sorted(..., key=lambda e:
    # e["id"])를 사용할 수 있습니다. next_id는 응답하지 않습니다.
    # data["entries"]를 id 기준으로 정렬해 반환
    # sorted(iterable, key=...)
    #   iterable=data["entries"]  : 지식 dict 목록
    #   key=lambda e: e["id"]     : 각 항목의 id 값으로 비교 -> [3, 1, 2] 순서여도 [1, 2, 3]으로 응답
    return sorted(data["entries"], key = lambda e: e["id"])
    # END B-API-01

# knowledge_id: int : Path 값을 정수로 변환
@app.get("/api/knowledge/{knowledge_id}", response_model=Knowledge)
# 요청 ID와 같은 지식 한 건을 반환한다.
def get_knowledge(knowledge_id: int):
    # 읽기 -> 수정 -> 저장을 같은 LOCK 안에서 수행해 중간에 다른 요청이 끼어들지 않게 한다.
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        # BEGIN B-API-02
        # TODO B-API-02: 반복 중인 entry의 id가 knowledge_id와 같을 때 entry를 반환하세요. 목록 인덱스와 ID를
        # 혼동하지 마세요.
        # entry["id"] == knowledge_id 일 때 바로 반환
        # id가 아닌 index값으로 찾지 않도록 주의한다.
        if entry["id"] == knowledge_id:
            return entry
        # END B-API-02
    raise HTTPException(404, "Knowledge not found")

# ============================================================
# 5. 생성 (POST)
# ============================================================
# status_code=201 : 생성 성공 상태코드 선언
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
# 새 지식을 저장하고 서버가 발급한 id와 함께 반환한다.
def create_knowledge(payload: KnowledgeWrite):
    # 읽기 -> 수정 -> 저장을 같은 LOCK 안에서 수행해 중간에 다른 요청이 끼어들지 않게 한다.
    with LOCK:
        data = read_data()
        # BEGIN B-API-03
        # TODO B-API-03: payload를 dict로 변환하여 entry에 넣고, data의 next_id 값을 entry의 id로
        # 넣으세요.
        # payload.model_dump() : Pydantic 모델을 dict로 변환
        # 입력 모델 -> dict 변환 후 id 부여
        entry = payload.model_dump()
        entry["id"] = data["next_id"]
        # END B-API-03
        # BEGIN B-API-04
        # TODO B-API-04: entry를 entries에 추가하고 next_id를 1 늘린 뒤 write_data(data)로 저장하세요.
        # 목록 추가 -> next_id 1 증가 -> 파일 저장
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
        # END B-API-04
    return entry

# ============================================================
# 6. 수정 (PUT)
# ============================================================
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
# 같은 ID의 지식 세 필드를 교체해 저장하고 반환한다.
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    # 읽기 -> 수정 -> 저장을 같은 LOCK 안에서 수행해 중간에 다른 요청이 끼어들지 않게 한다.
    with LOCK:
        data = read_data()
        # --- 같은 id의 위치 i 찾기 ---
        # enumerate(data["entries"]) : (위치 i, 항목 entry) 쌍을 순서대로 꺼냄
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                # 입력 세 필드 + 기존 id로 교체용 dict 생성
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                # BEGIN B-API-05
                # TODO B-API-05: 찾은 위치 i의 항목을 updated로 교체한 뒤 저장하세요. 같은 ID를 유지하고 다른 항목은 변경하지
                # 않습니다.
                # data["entries"][i] 한 칸만 교체 후 저장
                data["entries"][i] = updated
                write_data(data)
                # END B-API-05
                return updated
    raise HTTPException(404, "Knowledge not found")

# ============================================================
# 7. 삭제 (DELETE)
# ============================================================
# status_code=204 : 본문 없는 성공 응답
@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
# 같은 ID의 지식 하나만 삭제하고 본문 없는 204를 반환한다.
#   - 있는 ID: 204, 응답 본문 길이 0
#   - 없는 ID: 404, {"detail": "Knowledge not found"}, 파일 변화 없음
def delete_knowledge(knowledge_id: int):
    # 읽기 -> 수정 -> 저장을 같은 LOCK 안에서 수행해 중간에 다른 요청이 끼어들지 않게 한다.
    with LOCK:
        data = read_data()
        # --- 같은 id의 위치 i 찾기 ---
        # enumerate(data["entries"]) : (위치 i, 항목 entry) 쌍을 순서대로 꺼냄
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                # BEGIN B-API-06
                # TODO B-API-06: 찾은 위치 i의 항목 하나만 삭제하고 저장하세요. next_id는 줄이지 않습니다.
                # 위치 i 한 건만 제거 후 저장
                # list.pop(i) : i번째 항목 하나를 목록에서 제거
                # 예: id 4 삭제 -> entries는 id 1~3, next_id는 5 유지 (다음 생성은 5)
                data["entries"].pop(i)
                write_data(data)
                # END B-API-06
                return Response(status_code=204)
    raise HTTPException(404, "Knowledge not found")
