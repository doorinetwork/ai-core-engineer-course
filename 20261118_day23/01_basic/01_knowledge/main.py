"""Day23 실습 01 · 검색 대상과 ID (기본필수, mentor).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 01 기본필수: 지식 101~103의 ID·제목·본문을 확인하고 어떤 필드를 Embedding 입력으로 사용할지 표시합니다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
import argparse
import sys
from pathlib import Path
# 공통 모듈 폴더를 import 경로 맨 앞에 추가
# Path(__file__).resolve().parents[2] : <수준>/<교시>_주석/main.py 기준 두 단계 위 = mentee 폴더
# -> mentee/common 을 넣어 day23kit 패키지를 import 할 수 있게 함
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (load_knowledge)
from day23kit.data import load_knowledge
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result


# ============================================================
# 2. 원래 지식 101~103 선택
# ============================================================
# BEGIN D23-01-B01
def select_original(rows):
    # 전체 지식 목록에서 기존 수업 지식 101·102·103만 복사해 ID 오름차순으로 돌려준다.
    # - rows   : load_knowledge()가 읽은 지식 dict 목록 (101~109, 9건)
    # - 반환값 : 새 dict 3개 목록, ID 순서 [101, 102, 103]
    # - 원본 rows의 dict는 수정하지 않는다
    # [강사용 정답] D23-01-B01 · select_original
    #     기존 수업에서 사용한 101·102·103만 ID 오름차순으로 선택합니다.
    #     입력: 전체 지식 목록(list[dict]). 출력: 원래 지식 3개를 담은 list.
    #     원본 딕셔너리의 id/title/content를 수정하지 마세요.
    #     개념: 원문 ID 고정·검색할 자료의 범위.
    # 해설 D23-01-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # ① 기존 지식의 ID만 고르고 ② 순서 고정: 앞 차시의 ID/원문과 대응합니다.
    # (dict(r) for r in rows if ...) : 조건에 맞는 행만 얕은 복사 -> 원본 스냅샷 보존
    # key=lambda r: r['id']          : ID 오름차순 정렬 (입력 순서가 섞여 있어도 [101, 102, 103])
    return sorted((dict(r) for r in rows if r['id'] in (101, 102, 103)), key = lambda r: r['id'])
# END D23-01-B01

# ============================================================
# 3. E5 문서 입력 문자열 만들기
# ============================================================
# BEGIN D23-01-B02
def make_passage_input(row):
    # 지식 한 행을 E5 문서 입력 형식 "passage: 제목 본문" 문자열로 만든다.
    # - row    : 지식 dict 한 개 (title·content 사용)
    # - 반환값 : 'passage: ' + 제목 + 공백 1개 + 본문
    # 예: 102 -> "passage: 서버 실행 프로젝트 폴더에서 서버를 실행합니다."
    # [강사용 정답] D23-01-B02 · make_passage_input
    #     E5 문서 입력 텍스트를 "passage: 제목 본문" 형식으로 만드세요.
    #     입력: 지식 한 행. 출력: 문자열. 제목과 본문을 공백 하나로 구분하세요.
    #     개념: FAQ 질문과 달리 지식 본문을 검색 대상으로 사용합니다.
    # 해설 D23-01-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # E5 비대칭 검색은 문서에는 passage: 접두어가 필요합니다.
    # 주의: 질문 쪽은 'query: ' 접두어를 쓴다. 문서와 질문의 접두어를 바꿔 쓰면 E5 검색 계약이 깨진다
    return 'passage: ' + row['title'] + ' ' + row['content']
# END D23-01-B02

# ============================================================
# 4. ID·원문·버전·passage 입력 대응표 만들기
# ============================================================
# BEGIN D23-01-B03
def make_mapping(rows):
    # 지식마다 ID·원문·분류·버전과 E5 입력 문자열을 한 dict로 묶은 대응표를 만든다.
    # - rows   : 대응표를 만들 지식 목록 (기본필수는 101~103, 응용·심화는 101~109)
    # - 반환값 : dict 목록, 입력과 같은 순서
    # 대응표 키 : 의미
    # - id             : 지식 ID (벡터 행 번호가 아니라 원문을 찾는 열쇠)
    # - title/content  : 원문 제목·본문
    # - category       : 분류 (course / development / support)
    # - source_version : 원문 버전 (101~103은 knowledge_v1, 104~109는 knowledge_search_v2)
    # - passage_input  : make_passage_input()이 만든 E5 문서 입력
    # [강사용 정답] D23-01-B03 · make_mapping
    #     각 지식의 id/title/content/category/source_version 및 passage_input을 기록하세요.
    #     입력: 선택된 지식 목록. 출력: 딕셔너리 목록.
    #     개념: 벡터를 생성하기 전에도 원문·ID·버전을 추적해야 합니다.
    # 해설 D23-01-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 각 ID의 원문과 Embedding 입력을 함께 보존합니다.
    return [dict(id = r['id'], title = r['title'], content = r['content'],
                 category = r['category'], source_version = r['source_version'],
                 passage_input = make_passage_input(r)) for r in rows]
# END D23-01-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # 지식 스냅샷을 읽어 TODO 함수 결과를 결과 dict 하나로 모은다.
    # - mode   : 'fixture' 또는 'live' (이 실습은 DB·모델을 쓰지 않아 두 경로의 동작이 같다)
    # - level  : 1=기본필수, 2=응용, 3=심화. 수준이 높을수록 결과 키가 늘어난다
    # - 반환값 : outputs/exercise_01.json 에 저장될 dict
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 응용·심화 함수가 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. 지식 읽기와 TODO 함수 적용
    # ----------------------------------------------------
    # rows : common/data/knowledge_search_v2.json 의 지식 9건 (101~109)
    rows = load_knowledge()
    # selected : 101·102·103 세 건 / passage : 두 번째 지식 102의 E5 입력
    selected = select_original(rows)
    passage = make_passage_input(selected[1])
    # 대응표 대상 : 기본필수(level 1)는 3건, 응용·심화는 9건 전체
    mapping = make_mapping(selected if level == 1 else rows)
    # ----------------------------------------------------
    # 5-2. 결과 dict 구성
    # ----------------------------------------------------
    # result 키 : 의미
    # - exercise / level   : 실습 번호 1, 수준 번호
    # - source_mode        : 'local_snapshot' = 로컬 JSON 스냅샷만 읽음 (모델·DB 미사용)
    # - selected_ids       : [101, 102, 103]
    # - example_passage    : "passage: 서버 실행 프로젝트 폴더에서 서버를 실행합니다."
    # - total_mapped       : 대응표 행 수 (3 또는 9)
    # - mapping            : 대응표 전체
    result = dict(exercise = 1, level = level, source_mode = 'local_snapshot',
                  selected_ids = [r['id'] for r in selected], example_passage = passage,
                  total_mapped = len(mapping), mapping = mapping)
    # 응용 이상 : unique_ids = 중복 없는 ID 개수(9), all_fields_valid = 필수 필드 검사 결과(True)
    if level >= 2:
        result['unique_ids'] = len(reject_duplicate_ids(rows))
        result['all_fields_valid'] = reject_missing_fields(rows)
    # 심화 : versions = 버전별 ID 목록
    if level >= 3:
        result['versions'] = audit_versions(rows)
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_01.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 01 · 검색 대상과 ID')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_01.json', result)
    import json
    # json.dumps 인자
    # - ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 출력
    # - indent=2           : 사람이 읽기 쉽게 두 칸 들여쓰기
    # - default=str        : JSON으로 바꿀 수 없는 값(Path 등)은 문자열로 변환
    print(json.dumps(result, ensure_ascii = False, indent = 2, default = str))
    print('결과 저장:', path)

# ============================================================
# 7. 스크립트 진입점
# ============================================================
if __name__ == '__main__':
    main()
