# 실습 1 · 기본필수 API 보고서

> **보고서 ID:** D19-B01-FST-API  
> **작성 기준일:** 2026-11-12  
> **검증 범위:** `01_basic/01_FastAPI/main.py`의 TODO 6개 완성본, 명세서.md·예상결과.md, 실제 uvicorn 서버 실행 결과
> **판정:** 실습 산출물 검증 통과 (PASS)

## 0. 계약과 시작 데이터, 서버 구동

### 0.1 입력·계약
| 필드 | 요청 자료형 | 필수 | 규칙 |
|---|---|---|---|
| title | 문자열 | 예 | 1~80자 |
| content | 문자열 | 예 | 1~2000자 |
| category | 문자열 | 예 | 1~30자, 특정 목록으로 제한하지 않음 |
| id | 서버 응답의 정수 | 요청에서 작성하지 않음 | 서버가 next_id로 생성 |

### 0.2 시작 데이터 (`data/knowledge.json`)

| id | title | content | category | content 글자 수 |
|--:|:--|:--|:--|--:|
| 1 | 실습 파일 보관 | 실습 파일은 개인 폴더에 보관합니다. | learning | 20 |
| 2 | 서버 실행 | 프로젝트 폴더에서 서버를 실행합니다. | environment | 20 |
| 3 | 질문 작성 | 오류 메시지와 시도한 작업을 함께 적습니다. | learning | 24 |  

시작 상태에서 지식은 3개, `next_id`는 4로 설정되어 있다.    

### 0.3 서버 구동 명령

`python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload`

## 1. 목록(GET) /api/knowledge (status: 200)
GET 메서드를 사용해 현재 등록된 지식 리스트를 조회한다. 요청이 들어오면 `main.py`의 `@app.get("/api/knowledge")` 경로에 연결된 `list_knowledge` 함수가 호출된다. 등록된 모든 지식들의 리스트를 보여준다.  
```
[
    {
        "title": "실습 파일 보관",
        "content": "실습 파일은 개인 폴더에 보관합니다.",
        "category": "learning",
        "id": 1
    },
    {
        "title": "서버 실행",
        "content": "프로젝트 폴더에서 서버를 실행합니다.",
        "category": "environment",
        "id": 2
    },
    {
        "title": "질문 작성",
        "content": "오류 메시지와 시도한 작업을 함께 적습니다.",
        "category": "learning",
        "id": 3
    }
]
```
이 결과는 주어진 `01_initial_list.json`과 일치한다.  

## 2. 상세(GET) /api/knowledge/2 (status: 200)
GET 메서드를 사용해 `id = 2`에 해당하는 지식을 조회한다. 요청이 들어오면 `main.py`의 `@app.get("/api/knowledge/{knowledge_id}")` 경로에 연결된 `get_knowledge` 함수가 호출된다. 해당 id에 해당하는 지식의 id, 제목, 내용, 분류를 보여준다.  
```
{
    "title": "서버 실행",
    "content": "프로젝트 폴더에서 서버를 실행합니다.",
    "category": "environment",
    "id": 2
}
```
이 결과는 주어진 `02_detail.json`과 일치한다.  

## 3. 생성(POST) /api/knowledge/ (status: 201)
POST 메서드를 사용해 지식을 생성한다. 이 때 입력은 `requests/POST_valid.json`에 제시된 값을 사용한다. 요청이 들어오면 `main.py`의 `@app.post("/api/knowledge")` 경로에 연결된 `create_knowledge` 함수가 호출된다. 
```
{
    "title": "보충 학습",
    "content": "복습 자료는 개인 폴더에서 확인합니다.",
    "category": "learning",
    "id": 4
}
```
생성하도록 입력했던 제목, 내용, 분류와 함께 새로 발급된 id가 함께 반환된다. 4는 시작 상태에서 next_id였던 값이다. 이 생성 작업을 마치고 next_id는 하나 증가된 5로 변경된다.  
이 결과는 주어진 `03_created_example.json`에 제시된 결과와 필드의 순서는 다르나, 내용은 일치한다.  
*한계*: `create_knowledge`에서 반환되는 필드들을 id, title, content, category 순으로 정렬되도록 지정하지 않았다.

## 4. 수정(PUT) /api/knowledge/4 (status: 200)
PUT 메서드를 사용해 id 4번 지식을 수정한다. 이 때 입력은 `requests/PUT_valid.json`에 제시된 값을 사용한다. 요청이 들어오면 `main.py`의 `@app.put("/api/knowledge/{knowledge_id}"` 경로에 연결된 `update_knowledge` 함수가 호출된다.  
```
{
    "title": "보충 학습",
    "content": "복습 자료와 예제 코드를 함께 확인합니다.",
    "category": "learning",
    "id": 4
}
```
수정하도록 입력한 제목, 내용, 분류와 함께 해당 id가 함께 반환된다.  
이 결과는 주어진 `04_updated_example.json`에 제시된 결과와 필드의 순서는 다르나, 내용은 일치한다.  
*한계*: `update_knowledge`에서 반환되는 필드들을 id, title, content, category 순으로 정렬되도록 지정하지 않았다.

## 5. 삭제(DELETE) /api/knowledge/4 (status: 204 No Content)
DELETE 메서드를 사용해 id 4번 지식을 삭제한다. 요청이 들어오면 `main.py`의 `@app.delete("/api/knowledge/{knowledge_id}"` 경로에 연결된 `delete_knowledge` 함수가 호출된다.  
삭제가 성공하면 status code 204를 반환하며 별도의 데이터는 반환하지 않는다.  
추가로 `list_knowledge` 함수 호출을 해 현재 지식 목록의 상태를 확인하면 다음과 같다.  
```
[
    {
        "title": "실습 파일 보관",
        "content": "실습 파일은 개인 폴더에 보관합니다.",
        "category": "learning",
        "id": 1
    },
    {
        "title": "서버 실행",
        "content": "프로젝트 폴더에서 서버를 실행합니다.",
        "category": "environment",
        "id": 2
    },
    {
        "title": "질문 작성",
        "content": "오류 메시지와 시도한 작업을 함께 적습니다.",
        "category": "learning",
        "id": 3
    }
]
```
id 4번 지식이 삭제된 것을 확인할 수 있다. 이 결과는 주어진 `05_final_list.json`과 일치한다.

## 6. Status code 404·422 체크

### GET /api/knowledge/5 (status: 404)
GET 메서드를 통해 **현재 지식 목록에 없는 id**를 조회하려 하면 404 상태를 확인할 수 있다.
```
{
    "detail": "Knowledge not found"
}
```

### POST /api/knowledge/ (status: 422)
`content` **필드가 비어있는** 새로운 값을 POST하려 하면 422 상태를 확인할 수 있다. 이 때 입력은 `POST_empty_content.json`을 사용한다.
```
{
    "detail": [
        {
            "type": "string_too_short",
            "loc": [
                "body",
                "content"
            ],
            "msg": "String should have at least 1 character",
            "input": "",
            "ctx": {
                "min_length": 1
            }
        }
    ]
}
``` 
`content` 필드가 최소 1 length 이상의 문자열을 가져아 한다는 메시지가 출력된다. 이는 `KnowledgeWrite` 클래스에서 `min_length=1`로 정해둔 설정과도 일치한다.

### PUT /api/knowledge/3 (status: 422)
`content` 필드를 제외한 나머지 `title`, `category` **두 필수입력 필드가 비어있는** 값을 PUT하려 하면 422 상태를 확인할 수 있다. 이 때 입력은 `PUT_missing_fields.json`을 사용한다.  
```
{
    "detail": [
        {
            "type": "missing",
            "loc": [
                "body",
                "title"
            ],
            "msg": "Field required",
            "input": {
                "content": "복습 자료와 예제 코드를 함께 확인합니다."
            }
        },
        {
            "type": "missing",
            "loc": [
                "body",
                "category"
            ],
            "msg": "Field required",
            "input": {
                "content": "복습 자료와 예제 코드를 함께 확인합니다."
            }
        }
    ]
}
```
`title`, `category` 필드 값이 비어있다는 메시지가 출력된다. 이는 `KnowledgeWrite` 클래스에서 `min_length=1`로 정해둔 설정과도 일치한다.

## 7. 변경 내용 저장 유지
`data/knowledge.json` 파일을 시작 상태로 되돌린 뒤, 서버를 종료 후 재시작하여 동일 작업을 수행해도 같은 결과가 재현된다.  
지시된 일련의 작업을 수행한 뒤 `data/knowledge.json`의 내용은 다음과 같다.
```
{
  "next_id": 5,
  "entries": [
    {
      "id": 1,
      "title": "실습 파일 보관",
      "content": "실습 파일은 개인 폴더에 보관합니다.",
      "category": "learning"
    },
    {
      "id": 2,
      "title": "서버 실행",
      "content": "프로젝트 폴더에서 서버를 실행합니다.",
      "category": "environment"
    },
    {
      "id": 3,
      "title": "질문 작성",
      "content": "오류 메시지와 시도한 작업을 함께 적습니다.",
      "category": "learning"
    }
  ]
}
``` 
이는 주어진 예상 결과 `06_final_storage_example.json` 파일과 일치함이 확인된다.