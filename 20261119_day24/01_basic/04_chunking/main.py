"""실습 04 Basic · 종료 조건이 중요한 문자 창.

실습 04 Chunking · 기본필수
- 역할: 본문을 Chunk Size·Overlap 규칙의 문자 창 [start, end) 목록으로 나눈다
- 기능: check_params()(D24-04-B01) 크기·겹침 검사, make_windows()(D24-04-B02) 창 생성과 종료 조건
- 결과: outputs/chunks.json (71자 가상 입력 -> [0,40), [30,70), [60,71))
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

# ============================================================
# 2. Chunk Size·Overlap 검사 (TODO D24-04-B01)
# ============================================================
# size·overlap이 정수이고 0 <= overlap < size 인지 검사한다.
# - (40, 10)            : 통과
# - (40, 40)            : ValueError (다음 시작 = end - 40 -> 앞으로 나아가지 못해 무한 반복)
# - (40, -1)            : ValueError (음수 겹침)
# - (True, 0) / (40.0, 10) : ValueError (bool·float는 정수 크기로 인정하지 않음)
def check_params(size: int, overlap: int) -> None:
    # BEGIN D24-04-B01
    # 개념·실무 해설(mentor 원문): 0 이상인 Overlap이 Chunk Size보다 작아야 다음 시작 위치가 앞으로 진행합니다. bool은 int의 하위 타입이라 type()으로 따로 거부합니다.
    # TODO D24-04-B01 완료: size와 overlap을 정수로 검사하고 0 <= overlap < size를 보장합니다.
    # type(x) is not int : bool은 int의 하위 타입이라 isinstance(True, int)가 True이므로 type()으로 엄격히 비교
    if type(size) is not int or type(overlap) is not int or (not 0 <= overlap < size):
        raise ValueError('invalid chunk size or overlap')
    # END D24-04-B01

# ============================================================
# 3. 문자 창 목록 만들기 (TODO D24-04-B02)
# ============================================================
# text를 size 길이 창으로 자르고, 다음 창은 overlap 글자만큼 겹쳐 시작한다.
# - 창 범위 : [char_start, char_end) 시작 포함·끝 제외 (Python 슬라이스와 같음)
# - 종료    : 창 끝이 본문 끝(len(text))에 닿으면 즉시 종료
# - 예      : 71자, size=40, overlap=10 -> [0,40), [30,70), [60,71)
def make_windows(text: str, size: int, overlap: int) -> list[dict]:
    # 본문 끝에서 반복문을 종료하지 않으면 겹친 꼬리 청크가 생깁니다.
    # BEGIN D24-04-B02
    # 개념·실무 해설(mentor 원문): Chunk Size·Overlap은 0≤overlap<size를 만족해야 합니다. 창이 끝에 닿은 순간 종료해야 중복 꼬리 Chunk가 생기지 않습니다.
    # TODO D24-04-B02 완료: start·end·text가 있는 Chunk 목록을 만들고 본문 끝에 닿으면 종료합니다.
    # 잘못된 크기면 반복문에 들어가기 전에 중단
    check_params(size, overlap)
    spans = []
    start = 0

    # ----------------------------------------------------
    # 3-1. 창 반복 생성
    # ----------------------------------------------------
    while start < len(text):
        # end : 창 끝. 본문 끝을 넘지 않도록 min으로 자름 (예: start=60 -> min(100, 71) = 71)
        end = min(start + size, len(text))
        # 창 dict 키 : 의미
        # - char_start : 시작 위치 (포함)
        # - char_end   : 끝 위치 (제외)
        # - text       : text[start:end] 원문 그대로의 조각
        spans.append({'char_start': start, 'char_end': end, 'text': text[start:end]})
        # 종료 조건: 마지막 창을 넣은 직후 멈춤
        # 주의: 이 검사가 없으면 start = 71 - 10 = 61 에서 [61,71) 꼬리 창이 한 번 더 생긴다
        if end == len(text):
            break
        # 다음 시작 = 이번 끝 - 겹침 (예: 40 - 10 = 30)
        start = end - overlap
    return spans
    # END D24-04-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# '가' 71자 fixture로 창 3개가 나오는지 확인하고 저장한다.
def main():
    text = '가' * 71
    # make_windows 인자
    # - text       : '가' * 71 (71자 가상 입력)
    # - size = 40    : 창 최대 길이 40자
    # - overlap = 10 : 인접 창이 공유하는 글자 수 10자
    chunks = make_windows(text, size = 40, overlap = 10)
    # 기대 범위 [(0, 40), (30, 70), (60, 71)] 과 정확히 같은지 확인
    assert [(c['char_start'], c['char_end']) for c in chunks] == [(0, 40), (30, 70), (60, 71)]
    # 저장 dict 키 : 의미
    # - source : 입력 출처 표시 '71-character teaching fixture' (실제 문서가 아님)
    # - chunks : 창 3개 목록
    out = save_json(
        HERE,
        'chunks.json',
        {'source': '71-character teaching fixture', 'chunks': chunks},
    )
    show_success('04 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
