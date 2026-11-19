"""실습 05 Advanced · 검증 Gate와 Regression.

실습 05 Validation · 심화
- 역할: 청크 검증 Gate(Schema·본문 일치·중복 ID)로 통과/격리를 나누고, 오류 수정 후 정상 입력이 다시 통과하는지 회귀 검사한다
- 기능: schema_check()(A01) 필수 필드·자료형, text_match()(A02) 본문 일치, duplicated_ids()(A03) 중복 ID,
        coverage_stats()(A04) Coverage·Gap·Repeat, gate()(A05) 통과·격리 분리, regression()(A06) 회귀 검사
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
# 제공 모듈 coursekit/processing.py: 정제·문자 창 Chunking·Span 검증·Coverage·안정 chunk_id·E5 입력 문자열
from coursekit.processing import covered_positions

# ============================================================
# 2. Schema 검사 (TODO D24-05-A01)
# ============================================================
# 문자열 필수 필드 5개와 정수 Span 필드 2개를 검사한다.
# - chunk_id, doc_id, source_version, unit_id, body 중 빈 값 : ValueError('missing_<필드>')
# - char_start/char_end 가 정수 아님                       : ValueError('invalid_span_type')
def schema_check(chunk: dict) -> None:
    # BEGIN D24-05-A01
    # 개념·실무 해설(mentor 원문): Schema 검증은 필수 필드, 올바른 타입, 양의 위치 등을 확인하는 최소 Gate입니다. 누락을 정상 데이터로 바꾸지 않습니다.
    # TODO D24-05-A01 완료: 필수 필드·자료형·문자 범위를 검증합니다.
    for name in ('chunk_id', 'doc_id', 'source_version', 'unit_id', 'body'):
        if not isinstance(chunk.get(name), str) or not chunk[name].strip():
            raise ValueError(f'missing_{name}')
    if type(chunk.get('char_start')) is not int or type(chunk.get('char_end')) is not int:
        raise ValueError('invalid_span_type')
    # END D24-05-A01

# ============================================================
# 3. 본문 일치 검사 (TODO D24-05-A02)
# ============================================================
# Schema 통과 후 0 <= a < b <= len(clean) 이고 body == clean[a:b] 인지 확인한다.
# - 범위 밖   : ValueError('invalid_span')
# - 본문 다름 : ValueError('text_mismatch')
def text_match(chunk: dict, clean: str) -> None:
    # BEGIN D24-05-A02
    # 개념·실무 해설(mentor 원문): clean_text[start:end]와 저장된 Chunk 본문이 완전히 일치하는지 확인해야 Citation이 다시 원문 구간을 가리킬 수 있습니다.
    # TODO D24-05-A02 완료: 청크 본문이 기준 clean_text 슬라이스와 일치하는지 확인합니다.
    schema_check(chunk)
    a, b = (chunk['char_start'], chunk['char_end'])
    if not 0 <= a < b <= len(clean):
        raise ValueError('invalid_span')
    if chunk['body'] != clean[a:b]:
        raise ValueError('text_mismatch')
    # END D24-05-A02

# ============================================================
# 4. 중복 chunk ID 찾기 (TODO D24-05-A03)
# ============================================================
# 두 번 이상 나온 chunk_id 를 정렬해 돌려준다.
# - ['C1', 'C2', 'C1', 'C2'] -> ['C1', 'C2']
# 주의: 서로 다른 ID 의 Overlap(겹친 글자)은 여기서 오류로 보지 않는다
def duplicated_ids(chunks: list[dict]) -> list[str]:
    # BEGIN D24-05-A03
    # 개념·실무 해설(mentor 원문): Overlap으로 일부 문자가 반복되는 것은 의도적일 수 있지만 동일 Chunk ID를 두 번 생성하는 것은 별도 오류입니다.
    # TODO D24-05-A03 완료: 같은 chunk_id 중복과 의도한 overlap을 구분합니다.
    seen = set()
    duplicates = set()
    for c in chunks:
        cid = c.get('chunk_id')
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)
    return sorted(duplicates)
    # END D24-05-A03

# ============================================================
# 5. Coverage·Gap·Repeat (TODO D24-05-A04)
# ============================================================
# 통과 청크가 clean_text 를 얼마나 덮는지 계산한다.
# - clean 이 빈 문자열 : coverage None
# - [(0,10), (10,20)] / 20자 : coverage 1.0, gaps [], repeat_extra 0
def coverage_stats(chunks: list[dict], clean: str) -> dict:
    # BEGIN D24-05-A04
    # 개념·실무 해설(mentor 원문): Coverage의 분모는 해당 clean_text 문자 수입니다. 중복 위치를 여러 번 더하지 않으며 원본 PDF 전체가 보존됐다는 뜻도 아닙니다.
    # TODO D24-05-A04 완료: 고유 문자 위치 Coverage와 Gap·Repeat 양을 계산합니다.
    if not clean:
        # 반환 dict 키 : 의미
        # - coverage     : 덮인 고유 글자 수 / 전체 글자 수
        # - gaps         : 덮이지 않은 위치 목록 (위치 하나씩)
        # - repeat_extra : Span 길이 합 - 고유 글자 수 (겹쳐 저장된 글자 수)
        return {'coverage': None, 'gaps': [], 'repeat_extra': 0}
    spans = [(c['char_start'], c['char_end']) for c in chunks]
    n, missing = covered_positions(spans, len(clean))
    return {
        'coverage': n / len(clean),
        'gaps': missing,
        'repeat_extra': sum(b - a for a, b in spans) - n,
    }
    # END D24-05-A04

# ============================================================
# 6. 검증 Gate: 통과·격리 분리 (TODO D24-05-A05)
# ============================================================
# 청크마다 본문 일치·중복 ID 를 검사해 통과(passed)와 격리(quarantine, 사유 포함)로 나눈다.
# - 격리 사유 예: 'text_mismatch', 'duplicate_id', 'missing_body', 'invalid_span'
# - 격리 청크도 버리지 않고 chunk_id·reason 을 남김
def gate(chunks: list[dict], clean: str) -> dict:
    # BEGIN D24-05-A05
    # 개념·실무 해설(mentor 원문): 격리한 입력은 사유·위치·ID를 남깁니다. 실패한 Chunk를 버리면 공개 전 검수에서 무엇이 누락됐는지 알 수 없습니다.
    # TODO D24-05-A05 완료: 입력 오류별 reason code를 기록하고 통과 청크·격리 청크를 분리합니다.
    good = []
    quarantine = []
    seen = set()
    for c in chunks:
        # ------------------------------------------------
        # 6-1. 청크별 검사 (실패는 격리 목록으로)
        # ------------------------------------------------
        try:
            text_match(c, clean)
            # 앞에서 이미 통과한 chunk_id 와 같으면 중복으로 격리
            if c['chunk_id'] in seen:
                raise ValueError('duplicate_id')
            seen.add(c['chunk_id'])
        # KeyError·TypeError : 필드가 아예 없거나 자료형이 틀린 입력도 프로그램을 멈추지 않고 격리
        except (ValueError, KeyError, TypeError) as exc:
            quarantine.append({'chunk_id': c.get('chunk_id'), 'reason': str(exc)})
        else:
            good.append(c)
    # 반환 dict 키 : passed(통과 청크), quarantine(격리 목록), stats(통과 청크 기준 coverage_stats)
    return {'passed': good, 'quarantine': quarantine, 'stats': coverage_stats(good, clean)}
    # END D24-05-A05

# ============================================================
# 7. 오류 수정 후 회귀 검사 (TODO D24-05-A06)
# ============================================================
# 손상 입력은 정확히 2건 격리되고, 정상 입력 2건은 모두 통과(Coverage 1.0)하는지 확인한다.
def regression() -> dict:
    # BEGIN D24-05-A06
    # 개념·실무 해설(mentor 원문): 실패 사례 수정 후 정상 입력까지 다시 실행해야 합니다. 예외 검사를 강화하느라 정상 입력을 전부 차단하면 품질이 악화됩니다.
    # TODO D24-05-A06 완료: 오류 수정 후 정상 회귀 사례를 다시 확인합니다.
    # clean : 20자 기준 본문
    clean = '01234567890123456789'
    # good : [0,10) C1 + [10,20) C2 로 본문 전체를 정확히 덮는 정상 청크 2건
    good = [
        {
            'chunk_id': 'C1',
            'doc_id': 'DOC-TEST',
            'source_version': 'v1',
            'unit_id': 't1',
            'char_start': 0,
            'char_end': 10,
            'body': clean[:10],
        },
        {
            'chunk_id': 'C2',
            'doc_id': 'DOC-TEST',
            'source_version': 'v1',
            'unit_id': 't1',
            'char_start': 10,
            'char_end': 20,
            'body': clean[10:],
        },
    ]
    # broken : 정상 2건 + 본문이 'corrupted' 로 바뀐 C1 + 같은 C2 한 번 더
    # -> 앞의 C1·C2 는 통과, 손상 C1 은 text_mismatch, 두 번째 C2 는 duplicate_id 로 격리
    broken = [*good, {**good[0], 'body': 'corrupted'}, good[1]]
    result = gate(broken, clean)
    # 격리 2건이 아니면 Gate 가 오류를 놓친 것
    if len(result['quarantine']) != 2:
        raise AssertionError('error cases not isolated')
    # 수정된(정상) 입력 재실행: 통과 2건·Coverage 1.0 이어야 함 (정상 입력까지 막는 과잉 차단 확인)
    fixed = gate(good, clean)
    if len(fixed['passed']) != 2 or fixed['stats']['coverage'] != 1.0:
        raise AssertionError('normal regression failed')
    return {'failure_reasons': result['quarantine'], 'fixed_passed': len(fixed['passed'])}
    # END D24-05-A06

# ============================================================
# 8. 실행 흐름 (main)
# ============================================================
# 회귀 검사 결과를 outputs/validation_gate.json 으로 저장한다.
def main():
    out = save_json(HERE, 'validation_gate.json', regression())
    show_success('05 Advanced', out)
