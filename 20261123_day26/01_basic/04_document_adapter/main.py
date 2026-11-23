"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다.

실습 04 Document Adapter · 기본필수
- 역할: 검색 Hit(dict)를 LangChain Document로 바꾸며 본문은 page_content, 출처 여섯 값은 metadata로 보존한다
- 기능: to_document()(D26-04-B01) Hit 1건 -> Document, hits_to_documents()(D26-04-B02) 목록 변환
- 결과: outputs/document_adapter.json (body_preserved·metadata_preserved = True, citation_id_type = 'int')
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
from langchain_core.documents import Document
import argparse

# ============================================================
# 2. Metadata 계약
# ============================================================
# Document.metadata에 넣을 여섯 키. 모두 입력 Hit에서 그대로 가져온다
# - citation_id    : 출처 번호 (int, 예: 201)
# - chunk_id       : 청크 식별자 (예: 'demo-attend-v2-sec1')
# - doc_id         : 문서 ID (예: 'DOC-ATTEND')
# - source_version : 문서 버전 (예: 'v2')
# - source_locator : 원문 위치 (예: 'attendance_v2.md#sec-1')
# - category       : 분류 (학사·시설·지원)
# 오늘의 최소 Metadata 계약. 여섯 값 모두 입력 Hit에서 가져옵니다.
METADATA_KEYS = (
    "citation_id",
    "chunk_id",
    "doc_id",
    "source_version",
    "source_locator",
    "category",
)

# ============================================================
# 3. Hit 한 건 -> Document
# ============================================================
# 본문과 출처를 분리해 LangChain Document 하나로 만든다.
# - page_content : hit['body'] 원문 그대로 (요약·자르기 없음)
# - metadata     : 여섯 출처 필드 (citation_id는 int 그대로, 문자열로 바뀌지 않음)
# 주의: Document 생성은 데이터 보관일 뿐 Embedding·LLM 실행이 아니다
def to_document(hit: dict) -> Document:
    # 한 Hit의 본문과 여섯 Metadata 값을 Document로 옮깁니다.
    # BEGIN D26-04-B01
    # TODO D26-04-B01 완료: 본문을 page_content로, 제공된 여섯 Key를 metadata로 옮깁니다.
    # METADATA_KEYS 여섯 키만 복사. 하나라도 없으면 KeyError
    # page_content에는 본문을, metadata에는 여섯 출처 필드를 둡니다.
    # 정수 citation_id는 입력 값을 그대로 옮기므로 문자열로 바뀌지 않습니다.
    # hit[key]는 필수 Key가 없으면 KeyError를 내며 누락된 출처를 가짜 값으로 채우지 않습니다.
    metadata = {key: hit[key] for key in METADATA_KEYS}
    # Document 인자
    # - page_content = hit["body"] : 원문 본문
    # - metadata     = …           : 출처 여섯 값
    return Document(page_content = hit["body"], metadata = metadata)
    # END D26-04-B01


# ============================================================
# 4. Hit 목록 -> Document 목록
# ============================================================
# 입력 순서를 유지해 Hit마다 Document 하나를 만든다.
# - Hit 3건    : Document 3개 (검색 순서 유지)
# - 빈 목록    : [] (정상적인 빈 결과)
# - 필드 누락  : to_document()의 KeyError가 그대로 전달됨 (빈 결과로 숨기지 않음)
def hits_to_documents(hits: list[dict]) -> list[Document]:
    # 순서대로 변환한 Document 목록을 반환합니다. 빈 입력은 빈 목록입니다.
    # BEGIN D26-04-B02
    # TODO D26-04-B02 완료: 입력 순서대로 to_document를 호출해 목록을 반환합니다.
    # 검색 순서를 그대로 유지해 각 Hit를 한 Document로 바꿉니다.
    # 빈 입력에는 반복이 한 번도 실행되지 않으므로 빈 목록을 반환합니다.
    # 변환은 검색이 아니며 입력의 문서가 질문에 충분한지는 별도로 판단해야 합니다.
    documents = []
    for hit in hits:
        documents.append(to_document(hit))
    return documents
    # END D26-04-B02

# ============================================================
# 5. Metadata 보존 확인
# ============================================================
# 여섯 키 집합이 정확히 같고 값도 모두 같을 때만 True.
# - 키 일부만 복사 : False (집합 비교에서 걸림)
# - 값이 바뀜      : False (예: citation_id 201 -> '201')
def metadata_is_preserved(hit: dict, metadata: dict) -> bool:
    # 제공된 관찰 함수: 여섯 필드가 모두 있고 각 값이 같은지 확인합니다.
    required = {"citation_id", "chunk_id", "doc_id", "source_version", "source_locator", "category"}
    # 일부 필드만 복사한 결과를 '보존 완료'로 표시하지 않도록 Key 집합부터 확인합니다.
    return set(metadata) == required and all(metadata[key] == hit[key] for key in required)

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# Hit를 Document로 바꾸고 원문·출처 보존 여부를 비교해 outputs/document_adapter.json 으로 저장한다.
def main():
    # ----------------------------------------------------
    # 6-1. 명령행 인자 정의
    # ----------------------------------------------------
    parser = argparse.ArgumentParser(description = "Hit → 실제 LangChain Document 변환")
    # 명령행 인자 : 의미
    # - --live-db     : 실제 E5 질의 Embedding + PostgreSQL 검색 결과를 입력으로 사용
    # - --question    : 검색 질문 (기본 '15분 늦으면 어떻게 기록되나요?')
    # - --category    : 검색 분류 (학사·시설·지원 중 하나, 기본 '학사')
    # - --candidate-k : 검색 후보 수 (기본 3)
    parser.add_argument(
        "--live-db",
        action = "store_true",
        help = "실제 E5·PostgreSQL 검색 결과를 입력으로 사용",
    )
    parser.add_argument("--question", default = "15분 늦으면 어떻게 기록되나요?")
    parser.add_argument("--category", default = "학사", choices = ["학사", "시설", "지원"])
    parser.add_argument("--candidate-k", type = int, default = 3)
    args = parser.parse_args()

    # ----------------------------------------------------
    # 6-2. 입력 Hit 준비 (실제 검색 또는 강의 고정 예시)
    # ----------------------------------------------------
    # - --live-db : make_search_hits()로 E5 모델·DB 연결 함수를 준비해 실제 검색 (미실행 경로)
    # - 기본      : example_hit() 1건, mode = 'local_adapter'
    if args.live_db:
        # --live-db를 선택한 경우에만 실제 모델·DB 준비 함수를 불러옵니다.
        # 제공 모듈 coursekit/live.py: 실제 CPU E5 모델·PostgreSQL 연결과 검색 함수 (--live-db·실제 경로에서만)
        from coursekit.live import make_search_hits
        search_hits = make_search_hits()
        hits = search_hits(args.question, args.category, k = args.candidate_k)
        mode, input_source = ("live_db", "E5_PostgreSQL")
    else:
        hits = [example_hit()]
        mode, input_source = ("local_adapter", "lecture_fixture")

    # ----------------------------------------------------
    # 6-3. Document 변환과 보존 비교
    # ----------------------------------------------------
    documents = hits_to_documents(hits)
    comparisons = []
    # 비교 행 키 : 의미
    # - citation_id        : Document metadata의 출처 번호
    # - body_preserved     : page_content == 원문 body
    # - metadata_preserved : metadata_is_preserved() 결과
    # - citation_id_type   : 자료형 이름 ('int' 이어야 함)
    # - page_content       : 변환된 본문
    # - metadata           : 변환된 Metadata
    for hit, doc in zip(hits, documents):
        comparisons.append({
            "citation_id": doc.metadata["citation_id"],
            "body_preserved": doc.page_content == hit["body"],
            "metadata_preserved": metadata_is_preserved(hit, doc.metadata),
            "citation_id_type": type(doc.metadata["citation_id"]).__name__,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })

    # ----------------------------------------------------
    # 6-4. 결과 저장
    # ----------------------------------------------------
    # report 키 : 의미
    # - execution      : execution_record(mode, db=…, embedding=…) (기본 실행은 모두 False)
    # - input_source   : 'lecture_fixture' 또는 'E5_PostgreSQL'
    # - input_count    : 입력 Hit 수 / document_count : 변환된 Document 수
    # - documents      : 위 비교 행 목록
    report = {
        "execution": execution_record(mode, db = args.live_db, embedding = args.live_db),
        "input_source": input_source,
        "input_count": len(hits),
        "document_count": len(documents),
        "documents": comparisons,
    }
    path = save_json(HERE, "document_adapter.json", report)
    show_success("04 기본필수", path)

# ============================================================
# 7. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == "__main__":
    main()
