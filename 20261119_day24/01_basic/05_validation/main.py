"""실습 05 Basic · 유효한 위치인지 먼저 판단.

실습 05 Validation · 기본필수
- 역할: Chunk의 위치(Span)가 유효한지, 저장된 본문이 clean_text의 그 위치와 같은지 검사한다
- 기능: check_bounds()(D24-05-B01) 범위 검사, check_content()(D24-05-B02) 본문 일치·빈 청크 검사
- 결과: outputs/span_check.json ({"good": "pass", "bad": "expected failure"})
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
from coursekit.io import save_json, show_success

# ============================================================
# 2. Span 범위 검사 (TODO D24-05-B01)
# ============================================================
# 0 <= char_start < char_end <= len(text) 이고 두 값이 정수인지 검사한다.
# - (0, 20) / 20자 본문 : 통과
# - (5, 5)              : ValueError (길이 0 구간)
# - (0, 21) / 20자 본문 : ValueError (본문 밖)
# - ('0', 5)            : ValueError (문자열 위치)
def check_bounds(chunk: dict, text: str) -> None:
    # BEGIN D24-05-B01
    # 개념·실무 해설(mentor 원문): Schema 검증은 필수 필드, 올바른 타입, 양의 위치 등을 확인하는 최소 Gate입니다. 누락을 정상 데이터로 바꾸지 않습니다.
    # TODO D24-05-B01 완료: 범위 0 <= start < end <= len(clean_text)를 검사합니다.
    # a : char_start, b : char_end
    a, b = (chunk['char_start'], chunk['char_end'])
    if type(a) is not int or type(b) is not int or (not 0 <= a < b <= len(text)):
        raise ValueError('invalid span')
    # END D24-05-B01

# ============================================================
# 3. 본문 일치·빈 청크 검사 (TODO D24-05-B02)
# ============================================================
# 범위가 맞은 뒤, 저장된 chunk['text']가 text[start:end] 와 완전히 같은지 확인한다.
# - 같음          : 통과 (Citation이 원문 구간을 정확히 가리킴)
# - 다름          : ValueError('span text mismatch')
# - 공백만 있음   : ValueError('empty content')
def check_content(chunk: dict, text: str) -> None:
    # BEGIN D24-05-B02
    # 개념·실무 해설(mentor 원문): clean_text[start:end]와 저장된 Chunk 본문이 완전히 일치하는지 확인해야 Citation이 다시 원문 구간을 가리킬 수 있습니다.
    # TODO D24-05-B02 완료: chunk.text와 clean_text[start:end]가 같고 공백만 있는 청크가 아닌지 검사합니다.
    # 범위 검사를 먼저 해야 슬라이스가 엉뚱한 구간을 비교하지 않는다
    check_bounds(chunk, text)
    # 저장 본문 vs 원문 슬라이스 비교 (한 글자라도 다르면 실패)
    if chunk['text'] != text[chunk['char_start']:chunk['char_end']]:
        raise ValueError('span text mismatch')
    # 위치는 맞아도 공백뿐인 청크는 검색·인용 가치가 없으므로 거부
    if not chunk['text'].strip():
        raise ValueError('empty content')
    # END D24-05-B02

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 정상 청크는 통과, 본문이 바뀐 청크는 실패하는지 확인한다.
def main():
    text = '10분 초과 시 결석으로 기록합니다.'
    # good : 본문 전체 [0, 20) 과 같은 text를 가진 정상 청크
    good = {'char_start': 0, 'char_end': len(text), 'text': text}
    # bad : 위치는 같지만 text가 '잘못된 내용'으로 바뀐 손상 청크
    bad = {'char_start': 0, 'char_end': len(text), 'text': '잘못된 내용'}
    check_content(good, text)

    # ----------------------------------------------------
    # 4-1. 손상 청크는 반드시 실패해야 함
    # ----------------------------------------------------
    try:
        check_content(bad, text)
    except ValueError:
        result = 'expected failure'
    else:
        # 예외가 나지 않으면 검증 로직이 잘못된 것이므로 프로그램을 실패시킨다
        raise AssertionError('a corrupted chunk must fail')
    out = save_json(HERE, 'span_check.json', {'good': 'pass', 'bad': result})
    show_success('05 Basic', out)

# ============================================================
# 5. 스크립트 직접 실행 시 main() 호출
# ============================================================
if __name__ == '__main__':
    main()
