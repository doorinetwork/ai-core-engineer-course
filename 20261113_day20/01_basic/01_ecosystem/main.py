"""API·DB·LLM·응답(응용: 검색·검증, 심화: 권한·신뢰 경계)의 역할과 연결을 개념도로 저장한다.

- roles          : 구성요소 이름 -> 한 문장 역할
- edges          : (출발, 도착, 전달 데이터) 화살표 목록
- responsibility : 저장 책임과 생성 책임 구분
- trust_boundary : 신뢰하지 않는 입력과 권한 규칙
- 결과: outputs/architecture.md (Mermaid), outputs/architecture.json
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 폴더 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 수준 공통 모듈 common/support.py: JSON·Markdown 저장 도구
from common.support import write_json, write_md

# ============================================================
# 2. 실습 본문: AI 서비스 구성요소 개념도 작성
# ============================================================
# 강의 PPT 15장: 개념도에 실제 구현한 것과 향후 구현할 것을 구분합니다.
# LLM을 실제 호출하지 않습니다. diagram.md는 그림을 그리는 Mermaid 텍스트입니다.
def main():
    # --------------------------------------------------------
    # 2-1. 구성요소 역할 정의
    # --------------------------------------------------------
    # BEGIN D20-01-B01
    # TODO D20-01-B01 완료: FastAPI·PostgreSQL·LLM·응답의 역할 네 가지를 짧은 문장으로 작성하세요.
    # roles 키 : 의미
    # - FastAPI    : HTTP 요청 검증·함수 연결
    # - PostgreSQL : 원본 지식·분류 저장
    # - LLM        : 문맥을 받아 다음 Token 생성
    # - Response   : Client에 JSON 결과 전달
    roles = {
        "FastAPI": "HTTP 요청을 검증하고 함수·응답을 연결한다",
        "PostgreSQL": "원본 지식·분류를 관계형 테이블에 저장한다",
        "LLM": "문맥과 입력을 바탕으로 다음 Token을 생성한다",
        "Response": "호출한 Client에 JSON 결과를 전달한다",
    }
    # 해설: DB는 다음 토큰을 계산하지 않고, LLM은 영구 DB를 대신하지 않습니다.
    # END D20-01-B01

    # --------------------------------------------------------
    # 2-2. 요청부터 응답까지 화살표 연결
    # --------------------------------------------------------
    # BEGIN D20-01-B02
    # TODO D20-01-B02 완료: 요청→API→데이터·LLM→응답 연결을 올바른 순서의 화살표로 표현하세요.
    # edges 원소 = (출발 노드, 도착 노드, 화살표가 전달하는 데이터)
    # 예: ("Client", "FastAPI", "HTTP 요청") -> Client가 FastAPI로 HTTP 요청을 보냄
    edges = [
        ("Client", "FastAPI", "HTTP 요청"),
        ("FastAPI", "PostgreSQL", "원본 지식 조회·수정"),
        ("PostgreSQL", "FastAPI", "원본 데이터"),
        ("FastAPI", "LLM", "입력·문맥(후속 구현)"),
        ("LLM", "Response", "생성 결과(후속 구현)"),
        ("Response", "Client", "JSON"),
    ]
    # 해설: 이 선들은 역할의 개념 관계이며 오늘 구현하는 실제 파이프라인은 아닙니다.
    # END D20-01-B02
    # responsibility: 저장/생성 책임 두 줄 기록
    responsibility = {"storage":"DB는 원본 저장", "generation":"LLM은 언어 생성"}
    # trust_boundary: 범위 기록
    trust_boundary = {"scope":"후속 구현 개념도"}

    # --------------------------------------------------------
    # 2-3. Mermaid 개념도와 JSON 요약 저장
    # --------------------------------------------------------
    # BEGIN D20-01-B03
    # TODO D20-01-B03 완료: 완성한 연결 관계를 Mermaid 그림과 JSON 요약으로 저장하세요.
    # 해설: Mermaid는 파일을 열어 그릴 수 있는 표기법이며 네트워크·API 호출이 아닙니다.
    # node_names: 화살표 양 끝에 나온 노드 이름을 중복 없이 모아 가나다/알파벳순 정렬
    # node_id   : 이름 -> Mermaid 노드 ID (예: "Client" -> "N0")
    node_names = sorted({n for a,b,_ in edges for n in (a,b)})
    node_id = {name:f"N{i}" for i,name in enumerate(node_names)}
    # Mermaid 문법 줄 만들기
    # - "flowchart LR"              : 왼쪽 -> 오른쪽 방향 흐름도
    # - N0["Client"]                : 노드 선언 (ID와 표시 이름)
    # - N0 -->|"HTTP 요청"| N1      : 라벨이 붙은 화살표
    lines = ["flowchart LR"]
    lines += [f'  {node_id[name]}["{name}"]' for name in node_names]
    lines += [f'  {node_id[a]} -->|"{label}"| {node_id[b]}' for a,b,label in edges]
    # architecture.md   : Mermaid 코드 블록을 담은 Markdown
    # architecture.json : roles·edges·responsibility·trust_boundary 원본 값
    # 주의: JSON은 튜플을 저장할 수 없으므로 edges를 list로 바꿔 기록
    write_md(HERE/"outputs"/"architecture.md", "# AI 지식 플랫폼 개념도\n\n```mermaid\n"+"\n".join(lines)+"\n```\n")
    write_json(HERE/"outputs"/"architecture.json", {
        "roles":roles,"edges":[list(x) for x in edges],
        "responsibility":responsibility,"trust_boundary":trust_boundary,
    })
    # END D20-01-B03
    # 실행 완료 메시지 출력
    print("[01] architecture.md / architecture.json 생성 완료")

# ============================================================
# 3. 스크립트 실행
# ============================================================
if __name__=="__main__":
    main()
