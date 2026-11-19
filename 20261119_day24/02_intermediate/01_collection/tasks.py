"""실습 01 Intermediate · run via python main.py

실습 01 Collection · 응용
- 역할: manifest.json 의 문서 6건을 하나도 빠뜨리지 않고 수집 상태(collected/unsupported/read_error)로 기록한다
- 기능: sort_rows()(I01) ID 정렬·중복 검사, classify()(I02) 지원 형식 판정,
        collect_one()(I03) 1건 수집, collect_all()(I04) 전체 수집과 상태별 건수 대조
- 결과: outputs/manifest_report.json (total 6, collected 5, unsupported 1, read_error 0)
"""

# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
# HERE : 이 파일이 있는 실습 폴더 (outputs/ 저장 기준)
HERE = Path(__file__).resolve().parent
# HERE.parents[1] : <수준>/<실습>_주석 기준 두 단계 위 = mentee 폴더
# -> mentee 를 import 경로 맨 앞에 넣어 coursekit 패키지를 불러온다
sys.path.insert(0, str(HERE.parents[1]))
# 제공 모듈 coursekit/io.py: 공통 경로(DATA)·JSON 읽기/저장·완료 메시지
from coursekit.io import DATA, save_json, show_success, load_json
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import raw_path, sha_bytes

# ============================================================
# 2. 문서 목록 정렬·중복 ID 검사 (TODO D24-01-I01)
# ============================================================
# manifest 행을 doc_id 순으로 정렬한다. 같은 doc_id가 두 번 있으면 오류로 중단한다.
# - 6건 고유 ID : 'DOC-ASSIGN', 'DOC-ATTEND', 'DOC-CAREER', ... 사전순 목록
# - 중복 ID     : ValueError('duplicate doc ID')
def sort_rows(rows: list[dict]) -> list[dict]:
    # BEGIN D24-01-I01
    # 개념·실무 해설(mentor 원문): 원본 수집 시 경로가 존재하는지 확인하고 read_bytes()를 사용합니다. text decode는 다음 실습에서 수행하므로 이 단계에서 원문 바이트를 바꾸지 않습니다.
    # TODO D24-01-I01 완료: 문서 목록을 doc_id 순으로 정렬합니다. 동일 ID가 있으면 오류로 분류합니다.
    # ids : 모든 행의 doc_id 목록
    ids = [r['doc_id'] for r in rows]
    # set 으로 중복을 없앤 개수가 줄면 중복 ID가 있다는 뜻
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate doc ID')
    # key = lambda r: r['doc_id'] : 각 행의 doc_id 문자열로 정렬 (재실행해도 같은 순서)
    return sorted(rows, key = lambda r: r['doc_id'])
    # END D24-01-I01

# ============================================================
# 3. 지원 형식 판정 (TODO D24-01-I02)
# ============================================================
# 확장자가 허용 집합 {.txt, .md, .html, .pdf} 안이면 'supported', 아니면 'unsupported'.
# - 'career.pdf'      -> 'supported'
# - 'unsupported.xyz' -> 'unsupported'
# - 'NOTE.TXT'        -> 'supported' (suffix.lower() 로 대소문자 무시)
def classify(path: Path) -> str:
    # BEGIN D24-01-I02
    # 개념·실무 해설(mentor 원문): len(raw)은 Byte 길이이며 len(text)와 같지 않습니다. SHA-256은 같은 바이트를 다시 확인하기 위한 식별 단서이지 문서의 내용이 진실임을 증명하지 않습니다.
    # TODO D24-01-I02 완료: 확장자와 허용 집합을 비교하여 supported/unsupported를 반환합니다.
    return 'supported' if path.suffix.lower() in {'.txt', '.md', '.html', '.pdf'} else 'unsupported'
    # END D24-01-I02

# ============================================================
# 4. 문서 1건 수집 (TODO D24-01-I03)
# ============================================================
# 문서 1건의 원문 크기·해시 또는 실패 이유를 기록한 dict를 만든다.
# - 미지원 확장자 : status 'unsupported' (파일을 읽지 않음)
# - 읽기 실패     : status 'read_error', reason 예외 이름 (예: 'FileNotFoundError')
# - 정상          : status 'collected', size_bytes·raw_sha256 추가
def collect_one(doc: dict) -> dict:
    # BEGIN D24-01-I03
    # 개념·실무 해설(mentor 원문): 문서 ID와 파일명은 다른 계약입니다. 원본 이름이 변경되어도 동일 문서의 이력을 doc_id로 이어갈 수 있어야 합니다.
    # TODO D24-01-I03 완료: 파일별 원문 크기·해시 또는 읽기 실패 이유를 안전하게 기록합니다.
    # raw_path : data/raw/ 아래 경로로 만들고, data/raw 밖을 가리키면 ValueError
    path = raw_path(doc)
    # result 기본 키 : 의미
    # - doc_id   : 문서 ID
    # - file     : 원문 파일명
    # - version  : 원문 판 (source_version)
    # - category : 분류
    # - status   : classify() 결과로 시작 ('supported' 는 아래에서 collected/read_error 로 바뀜)
    result = {
        'doc_id': doc['doc_id'],
        'file': doc['file'],
        'version': doc['source_version'],
        'category': doc['category'],
        'status': classify(path),
    }
    if result['status'] == 'unsupported':
        return result

    # ----------------------------------------------------
    # 4-1. 원문 읽기 (실패도 기록으로 남김)
    # ----------------------------------------------------
    try:
        raw = path.read_bytes()
    except OSError as exc:
        # OSError : 파일 없음·권한 없음 등. 예외 이름만 기록하고 다음 문서로 진행
        result.update(status = 'read_error', reason = type(exc).__name__)
    else:
        # else 블록 : 예외가 없을 때만 실행. 바이트 수와 SHA-256 기록
        result.update(status = 'collected', size_bytes = len(raw), raw_sha256 = sha_bytes(raw))
    return result
    # END D24-01-I03

# ============================================================
# 5. 전체 수집과 상태별 건수 대조 (TODO D24-01-I04)
# ============================================================
# 모든 문서를 제외 없이 수집하고 상태별 건수의 합이 입력 수와 같은지 확인한다.
# - 6건 입력 -> counts {'collected': 5, 'unsupported': 1, 'read_error': 0}, 합 6
def collect_all(rows: list[dict]) -> dict:
    # BEGIN D24-01-I04
    # 개념·실무 해설(mentor 원문): 지원 밖 확장자를 조용히 제외하면 입력 총수가 줄어듭니다. unsupported와 read_error는 별도의 상태로 남깁니다.
    # TODO D24-01-I04 완료: 모든 입력을 제외 없이 기록하고 상태별 건수 합이 입력 수와 일치하는지 검사합니다.
    # sort_rows 로 정렬·중복 검사 후 문서마다 collect_one 실행
    items = [collect_one(r) for r in sort_rows(rows)]
    # state 별로 status 가 같은 항목 수를 센다 (True=1, False=0 합산)
    counts = {
        state: sum(r['status'] == state for r in items)
        for state in ('collected', 'unsupported', 'read_error')
    }
    # 주의: 세 상태 외의 status 가 생기면 합이 맞지 않아 중단된다 (누락 탐지)
    assert sum(counts.values()) == len(items)
    # 반환 dict 키 : 의미
    # - total  : 입력 문서 수
    # - counts : 상태별 건수
    # - items  : 문서별 수집 기록
    return {'total': len(items), 'counts': counts, 'items': items}
    # END D24-01-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# manifest.json 전체를 수집해 outputs/manifest_report.json 으로 저장한다.
def main():
    result = collect_all(load_json(DATA / 'manifest.json'))
    out = save_json(HERE, 'manifest_report.json', result)
    show_success('01 Intermediate', out)
