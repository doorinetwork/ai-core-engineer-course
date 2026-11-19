"""실습 05 Intermediate · 설명용 20자 경계 세트.

실습 05 Validation · 응용
- 역할: 20자 가상 본문의 Span 세트 3종(overlap·missing·duplicate)으로 Coverage·Gap·중복 ID 를 계산해 구분한다
- 기능: validate_ranges()(I01) 범위 검사, coverage()(I02) 합집합 Coverage,
        detect_gaps_and_duplicates()(I03) 빠진 구간·중복 ID, make_report()(I04) 판정 보고서
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
# 2. Span 범위 검사 (TODO D24-05-I01)
# ============================================================
# 모든 (a, b) 가 0 <= a < b <= length 인지 검사한다.
# - length 20, [(0, 8), (6, 14)] : 통과
# - (8, 8) 또는 (15, 21)        : ValueError('invalid range')
def validate_ranges(length: int, spans: list[tuple[int, int]]) -> None:
    # BEGIN D24-05-I01
    # 개념·실무 해설(mentor 원문): Schema 검증은 필수 필드, 올바른 타입, 양의 위치 등을 확인하는 최소 Gate입니다. 누락을 정상 데이터로 바꾸지 않습니다.
    # TODO D24-05-I01 완료: 청크별 Span이 기준 본문의 범위 안인지 검증합니다.
    for a, b in spans:
        if not 0 <= a < b <= length:
            raise ValueError('invalid range')
    # END D24-05-I01

# ============================================================
# 3. 합집합 Coverage (TODO D24-05-I02)
# ============================================================
# Span 들이 덮은 고유 문자 수 / 전체 문자 수 를 계산한다.
# - length 0                       : None (0 으로 나누지 않고 "계산 불가" 표시)
# - [(0, 8), (10, 20)] / 20        : 18 / 20 = 0.9
# - [(0, 8), (6, 14), (12, 20)] / 20 : 겹친 6~8·12~14 는 한 번만 세어 20 / 20 = 1.0
def coverage(length: int, spans: list[tuple[int, int]]) -> float | None:
    # BEGIN D24-05-I02
    # 개념·실무 해설(mentor 원문): clean_text[start:end]와 저장된 Chunk 본문이 완전히 일치하는지 확인해야 Citation이 다시 원문 구간을 가리킬 수 있습니다.
    # TODO D24-05-I02 완료: 합집합으로 덮은 문자 수와 Coverage를 계산합니다.
    if length == 0:
        return None
    validate_ranges(length, spans)
    # covered_positions : (덮인 위치 수, 덮이지 않은 위치 목록) 반환. 겹침은 한 번만 셈
    n, _ = covered_positions(spans, length)
    return n / length
    # END D24-05-I02

# ============================================================
# 4. 빠진 구간·중복 ID 찾기 (TODO D24-05-I03)
# ============================================================
# 덮이지 않은 위치를 연속 구간으로 묶고, 두 번 이상 나온 chunk ID 를 찾는다.
# - missing 위치 [8, 9]  -> gaps [(8, 10)]
# - ids ['A', 'A', 'C']  -> duplicate_ids ['A']
def detect_gaps_and_duplicates(length: int, spans: list[tuple[int, int]], ids: list[str]) -> dict:
    # BEGIN D24-05-I03
    # 개념·실무 해설(mentor 원문): Overlap으로 일부 문자가 반복되는 것은 의도적일 수 있지만 동일 Chunk ID를 두 번 생성하는 것은 별도 오류입니다.
    # TODO D24-05-I03 완료: 빠진 위치의 연속 구간 및 중복 Chunk ID를 구합니다.
    _, missing = covered_positions(spans, length)

    # ----------------------------------------------------
    # 4-1. 빠진 위치를 연속 구간으로 묶기
    # ----------------------------------------------------
    gaps = []
    for pos in missing:
        # 직전 구간의 끝과 이어지면 끝을 한 칸 늘림 (예: (8, 9) + 9 -> (8, 10))
        if gaps and gaps[-1][1] == pos:
            gaps[-1] = (gaps[-1][0], pos + 1)
        else:
            gaps.append((pos, pos + 1))

    # ----------------------------------------------------
    # 4-2. 중복 chunk ID 찾기
    # ----------------------------------------------------
    seen = set()
    dups = []
    for cid in ids:
        if cid in seen:
            dups.append(cid)
        seen.add(cid)
    # duplicate_ids 는 sorted(set(...)) 로 중복 없이 정렬 (재실행해도 같은 순서)
    return {'gaps': gaps, 'duplicate_ids': sorted(set(dups))}
    # END D24-05-I03

# ============================================================
# 5. 판정 보고서 (TODO D24-05-I04)
# ============================================================
# Coverage·덮인 글자 수·반복 글자 수·오류를 모아 valid 여부를 판정한다.
# - 의도한 Overlap 만 있음 : valid True (반복은 오류가 아님)
# - Gap 또는 중복 ID 있음  : valid False
def make_report(length: int, spans: list[tuple[int, int]], ids: list[str]) -> dict:
    # BEGIN D24-05-I04
    # 개념·실무 해설(mentor 원문): Coverage의 분모는 해당 clean_text 문자 수입니다. 중복 위치를 여러 번 더하지 않으며 원본 PDF 전체가 보존됐다는 뜻도 아닙니다.
    # TODO D24-05-I04 완료: 통과·오류·반복량·Coverage를 구분한 보고서를 작성합니다.
    validate_ranges(length, spans)
    covered, _ = covered_positions(spans, length)
    errors = detect_gaps_and_duplicates(length, spans, ids)
    # 반환 dict 키 : 의미
    # - coverage           : 합집합 Coverage (0.0 ~ 1.0)
    # - covered_characters : 덮인 고유 글자 수
    # - repeat_extra_chars : Span 길이 합 - 고유 글자 수 = 겹쳐서 더 저장된 글자 수 (예: overlap 세트 24 - 20 = 4)
    # - errors             : gaps·duplicate_ids
    # - valid              : Gap 과 중복 ID 가 모두 없을 때만 True
    # 주의: Coverage 1.0 이어도 duplicate 세트처럼 valid 는 False 일 수 있다
    return {
        'coverage': coverage(length, spans),
        'covered_characters': covered,
        'repeat_extra_chars': sum(b - a for a, b in spans) - covered,
        'errors': errors,
        'valid': not errors['gaps'] and (not errors['duplicate_ids']),
    }
    # END D24-05-I04

# ============================================================
# 6. 실행 흐름 (main)
# ============================================================
# 설명용 20자 경계 세트 3종을 판정하고 Coverage 가 [1.0, 0.9, 1.0] 인지 확인한다.
def main():
    # samples 키 : (Span 목록, chunk ID 목록)
    # - overlap   : 의도한 겹침 (0~8, 6~14, 12~20) -> 정상
    # - missing   : 8~10 두 글자가 빠짐 -> Gap
    # - duplicate : 같은 ID 'A' 가 두 번 -> 중복 ID 오류
    samples = {
        'overlap': ([(0, 8), (6, 14), (12, 20)], ['A', 'B', 'C']),
        'missing': ([(0, 8), (10, 20)], ['A', 'B']),
        'duplicate': ([(0, 8), (0, 8), (8, 20)], ['A', 'A', 'C']),
    }
    # make_report(20, spans, ids) : *parts 로 (spans, ids) 튜플을 두 인자로 펼침
    reports = {name: make_report(20, *parts) for name, parts in samples.items()}
    assert [reports[n]['coverage'] for n in samples] == [1.0, 0.9, 1.0]
    out = save_json(HERE, 'coverage_report.json', reports)
    show_success('05 Intermediate', out)
