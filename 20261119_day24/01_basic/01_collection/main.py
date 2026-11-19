"""실습 01 Basic · Python 파일 하나로 원문을 수집합니다.

실습 01 Collection · 기본필수
- 역할: 원문 파일 1건(data/raw/assignment.txt)을 바이트 그대로 읽어 문서 ID·파일명·바이트 수·SHA-256을 기록한다
- 기능: read_raw()(D24-01-B01)로 원문 bytes 읽기, build_record()(D24-01-B02)로 수집 기록 dict 생성
- 결과: outputs/collection.json
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
from coursekit.io import DATA, save_json, show_success
# 제공 모듈 coursekit/documents.py: manifest 조회·원문 경로·SHA-256·형식별 Parser(TXT/Markdown/HTML/PDF)
from coursekit.documents import sha_bytes


# ============================================================
# 2. 원문 파일을 bytes로 읽기 (TODO D24-01-B01)
# ============================================================
# 원문 파일을 변경 없이 bytes로 읽어 돌려준다.
# - 파일이 있음   : path.read_bytes() 결과 반환 (예: assignment.txt -> 267 bytes)
# - 파일이 없음   : FileNotFoundError로 즉시 중단 (빈 bytes로 대신하지 않음)
# - 폴더 경로     : is_file()이 False이므로 FileNotFoundError
def read_raw(path: Path) -> bytes:
    # 입력: Path / 출력: bytes. 원본 파일은 수정하지 않습니다.
    # BEGIN D24-01-B01
    # 개념·실무 해설(mentor 원문): 원본 수집 시 경로가 존재하는지 확인하고 read_bytes()를 사용합니다. 
    # text decode는 다음 실습에서 수행하므로 이 단계에서 원문 바이트를 바꾸지 않습니다.
    # TODO D24-01-B01 완료: 입력 경로의 파일을 bytes로 읽어 반환합니다. 없는 파일은 명확히 실패해야 합니다.
    # path.is_file() : 경로가 실제로 존재하는 "일반 파일"일 때만 True
    # 주의: 없는 파일을 b''로 바꿔 진행하면 다음 단계에서 "빈 문서"로 오인된다
    if not path.is_file():
        raise FileNotFoundError(path)
    # read_bytes() : 디코딩하지 않은 원본 바이트. 텍스트 변환(decode)은 실습 02 Parsing에서 수행
    return path.read_bytes()
    # END D24-01-B01

# ============================================================
# 3. 수집 기록 dict 만들기 (TODO D24-01-B02)
# ============================================================
# 문서 ID·파일명·바이트 수·SHA-256을 한 건의 수집 기록으로 묶는다.
# - doc_id : manifest의 문서 식별자 (예: 'DOC-ASSIGN'). 파일명이 바뀌어도 이어지는 ID
# - path   : 원문 경로. 기록에는 파일 이름(path.name)만 남김
# - raw    : read_raw()가 읽은 원본 bytes
def build_record(doc_id: str, path: Path, raw: bytes) -> dict:
    # 파일명은 식별자가 아닙니다. 문서 ID와 원본 바이트 해시를 별도로 보관합니다.
    # BEGIN D24-01-B02
    # 개념·실무 해설(mentor 원문): len(raw)은 Byte 길이이며 len(text)와 같지 않습니다. 
    # SHA-256은 같은 바이트를 다시 확인하기 위한 식별 단서이지 문서의 내용이 진실임을 증명하지 않습니다.
    # TODO D24-01-B02 완료: 문서 ID·원문 파일명·바이트 수·SHA-256을 한 딕셔너리에 묶습니다.
    # 반환 dict 키 : 의미
    # - doc_id     : 문서 ID (예: 'DOC-ASSIGN')
    # - filename   : 원문 파일 이름 (예: 'assignment.txt')
    # - size_bytes : 바이트 수 len(raw). 글자 수 len(text)와 다름 (한글 1자 = UTF-8 3바이트)
    # - raw_sha256 : 원본 바이트의 SHA-256 16진 문자열 64자. 같은 바이트인지 다시 확인하는 지문
    # 주의: SHA-256은 "같은 파일인가"만 알려 주며 내용이 옳다는 증명이 아니다
    return {
        'doc_id': doc_id,
        'filename': path.name,
        'size_bytes': len(raw),
        'raw_sha256': sha_bytes(raw),
    }
    # END D24-01-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# assignment.txt 1건을 수집해 outputs/collection.json 으로 저장한다.
def main():
    # DATA : coursekit/io.py가 정한 mentee/data 폴더 (__file__ 기준 절대 경로)
    path = DATA / 'raw' / 'assignment.txt'
    raw = read_raw(path)
    # build_record 인자
    # - 'DOC-ASSIGN' : manifest의 과제 안내 문서 ID
    # - path         : 원문 경로
    # - raw          : 읽은 원본 bytes
    record = build_record('DOC-ASSIGN', path, raw)
    # 기록한 바이트 수가 실제 읽은 바이트 수와 같은지 마지막으로 확인
    assert record['size_bytes'] == len(raw)
    # save_json(폴더, 파일명, dict) : 폴더/outputs/파일명 에 JSON 저장 후 경로 반환
    out = save_json(HERE, 'collection.json', record)
    show_success('01 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
