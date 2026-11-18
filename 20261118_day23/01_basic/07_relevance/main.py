"""Day23 실습 07 · 검색 결과 정성평가.

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 07 기본필수: 질문 3개에 대한 Top-k 결과의 원문을 읽고 관련·무관을 판단합니다.
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
# 제공 모듈 common/day23kit/fixtures.py: 교육용 2차원 모의 벡터와 PostgreSQL 없이 도는 모의 Top-k 검색 (fixture_search)
from day23kit.fixtures import fixture_search
# 제공 모듈 common/day23kit/embedding.py: Encoder 선택(fixture 모의 / live 실제 E5)·문서·질문 인코딩·벡터 계약 검사 (get_encoder, encode_question)
from day23kit.embedding import get_encoder, encode_question
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (connect_db, exact_topk, check_data_ready)
from day23kit.db import connect_db, exact_topk, check_data_ready
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result

# ============================================================
# 2. 평가 질문 3종(casebook)
# ============================================================
# BEGIN D23-07-B01
def make_casebook():
    # 강의의 질문 3종과 사람이 확인할 근거 지식 ID를 고정 목록으로 돌려준다.
    # casebook 키 : 의미
    # - case_id       : 질문 이름 (server / retention / error)
    # - question      : 검색 질문 원문
    # - reference_ids : 원문을 직접 근거로 볼 지식 ID ([102] / [] / [103, 108])
    # 주의: retention(보관 기간)은 지식 101에 보관 '위치'만 있고 '기간'이 없어 직접 근거가 없다 -> []
    # [강사용 정답] D23-07-B01 · make_casebook
    #     강의의 질문 3종과 확인할 지식 ID를 구성하세요.
    #     server(102), retention(101: 보관 기간 없음), error(103/108).
    #     자동 생성한 정답이 아니라 사람이 평가할 고정 기준 샘플임을 표시합니다.
    # 해설 D23-07-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return [
       {'case_id': 'server', 'question': '서버는 어디에서 실행하나요?', 'reference_ids': [102]},
       {'case_id': 'retention', 'question': '실습 파일은 몇 년 동안 보관하나요?', 'reference_ids': []},
       {'case_id': 'error', 'question': '오류를 질문할 때 무엇을 적어야 하나요?', 'reference_ids': [103, 108]},
    ]
# END D23-07-B01

# ============================================================
# 3. 교육용 기준표로 관련성 라벨 붙이기
# ============================================================
# BEGIN D23-07-B02
def judge_training_reference(case_id, doc_id):
    # 사람이 미리 만든 기준표(gold)에서 (질문, 문서 ID)의 관련성 라벨을 찾아 돌려준다.
    # - 기준표에 있는 쌍 : '관련' 또는 '부분 관련'  예: ('server', 102) -> '관련', ('server', 106) -> '부분 관련'
    # - 기준표에 없는 쌍 : '무관'                   예: ('server', 107) -> '무관'
    # - 모르는 case_id   : ValueError('unknown_case')
    # 주의: 모델 점수가 아니라 교육용 수동 기준이다
    # [강사용 정답] D23-07-B02 · judge_training_reference
    #     교안에 미리 주어진 원문 근거로 관련/부분 관련/무관을 수동 기준표로 판정하세요.
    #     server:102 관련,106 부분 관련. retention:101 부분 관련(기간 없음).
    #     error:103,108 관련. 그 밖에는 무관.
    #     *이 표는 사람이 만든 교육용 기준이며 모델 검색 평가점수가 아닙니다.*
    # 해설 D23-07-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # gold : {질문 이름: {문서 ID: 라벨}}
    gold = {'server': {102: '관련', 106: '부분 관련'},
            'retention': {101: '부분 관련'},
            'error': {103: '관련', 108: '관련'}}
    if case_id not in gold:
        raise ValueError('unknown_case')
    # .get(doc_id, '무관') : 기준표에 없는 문서는 무관
    return gold[case_id].get(doc_id, '무관')
# END D23-07-B02

# ============================================================
# 4. 검토 행 한 개 만들기
# ============================================================
# BEGIN D23-07-B03
def make_review_row(case, hit):
    # 질문 하나와 검색 후보 하나를 사람이 검토할 표의 한 행으로 만든다.
    # 검토 행 키 : 의미
    # - case_id / query     : 질문 이름·원문
    # - retrieved_id        : 검색된 지식 ID
    # - content             : 검색된 원문 (요약·수정하지 않음)
    # - distance            : Cosine distance (정답 확률이 아님, 작을수록 가까울 뿐)
    # - teaching_reference  : 교육용 기준 라벨 (관련 / 부분 관련 / 무관)
    # - human_reviewed      : False (실제 사람 검토 전)
    # [강사용 정답] D23-07-B03 · make_review_row
    #     질문 ID·검색된 ID·본문·거리·교재 기준 라벨을 한 행으로 만드세요.
    #     distance는 정답 확률로 표현하지 않습니다.
    #     human_reviewed는 False로 두어 실제 수강생의 근거 검토 전임을 표시합니다.
    # 해설 D23-07-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return dict(case_id = case['case_id'], query = case['question'],
                retrieved_id = hit['id'], content = hit['content'],
                distance = round(hit['distance'], 6),
                teaching_reference = judge_training_reference(case['case_id'], hit['id']),
                human_reviewed = False)
# END D23-07-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # 질문 3종을 각각 Top-3 검색하고, 후보마다 교육용 기준 라벨을 붙인 검토 행 9개를 만든다.
    # - mode='fixture' : fixture_search() 모의 검색 (교육용 2차원 좌표)
    # - mode='live'    : 실제 E5 질문 Embedding + pgvector Exact Top-k (실습 05 live 데이터 필요)
    # - level          : 2 이상은 검색 상태 판정·필터 효과·빈 결과/오류 구분, 3은 검토 행 무결성 검사
    # - human_evidence_ok=None : 사람의 근거 검토가 아직 없다는 뜻 (True로 만들지 않음)
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 classify_search_state 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. 질문별 검색과 검토 행 만들기
    # ----------------------------------------------------
    cases = make_casebook()
    reviews = []
    search_state = []
    for case in cases:
        # 검색 자체는 fixture의 수학 예시 또는 실제 E5/pgvector.
        if mode == 'fixture':
            hits = fixture_search(case['question'], k = 3)
        else:
            # live 첫 질문에서만 Encoder 로드·DB 연결을 한 번 만든다 (_live_cache 로 재사용 표시)
            if '_live_cache' not in locals():
                encoder = get_encoder('live')
                conn = connect_db(Path(__file__).with_name('.env'))
                check_data_ready(conn)
                _live_cache = True
            qvec = encode_question(encoder, case['question'])
            hits = exact_topk(conn, qvec, 3)
        # 검색 상태 : 응용 이상은 classify_search_state(), 기본필수는 후보 유무만 간단히 판정
        search_state.append({'case_id': case['case_id'], 'state': classify_search_state(hits) if level >= 2 else ('candidates' if hits else 'empty')})
        for hit in hits:
            row = make_review_row(case, hit)
            # source_version : 무결성 검사(심화)에서 스냅샷 버전과 대조하려고 행에 함께 기록
            row['source_version'] = hit['source_version']
            reviews.append(row)
    # ----------------------------------------------------
    # 5-2. 결과 dict 구성
    # ----------------------------------------------------
    # result 키 : 의미
    # - review_rows       : 검토 행 9개 (질문 3 × 후보 3)
    # - search_states     : 질문별 상태 (fixture에서는 셋 다 'candidates')
    # - human_evidence_ok : None (사람 검토 전)
    result = dict(exercise = 7, level = level, source_mode = 'simulated' if mode == 'fixture' else 'pgvector_exact',
                  review_rows = reviews, search_states = search_state,
                  human_evidence_ok = None)
    # ----------------------------------------------------
    # 5-3. 응용·심화: 필터 효과와 상태 구분
    # ----------------------------------------------------
    if level >= 2:
        # baseline : 필터 없는 Top-3 / filtered : category='development' Top-3
        if mode == 'live':
            qvec = encode_question(encoder, '서버는 어디에서 실행하나요?')
            baseline = [h['id'] for h in exact_topk(conn, qvec, k = 3)]
            filtered = [h['id'] for h in exact_topk(conn, qvec, k = 3, category = 'development')]
        else:
            baseline = [h['id'] for h in fixture_search('서버는 어디에서 실행하나요?', k = 3)]
            filtered = [h['id'] for h in fixture_search('서버는 어디에서 실행하나요?', k = 3, category = 'development')]
        result['filter_effect'] = review_filter_effect(baseline, filtered)
        # empty_is_not_error : 빈 결과는 'empty', 오류 문자열이 있으면 'error' 로 서로 다르게 나오는지 확인 -> True
        result['empty_is_not_error'] = classify_search_state([]) == 'empty' and classify_search_state([], error = 'DB 연결 오류') == 'error'
    if level >= 3:
        result['evaluation_integrity'] = check_evaluation_integrity(reviews)
    # live에서 연 DB 연결을 닫는다
    if mode == 'live':
        conn.close()
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_07.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 07 · 검색 결과 정성평가')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_07.json', result)
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
