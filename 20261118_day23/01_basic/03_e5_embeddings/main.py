"""Day23 실습 03 · E5 문장 Embedding.

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 03 기본필수: 모델을 로드해 간단한 지식과 질문을 인코딩하고 shape를 확인합니다.
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
import numpy as np
# 제공 모듈 common/day23kit/data.py: 지식 스냅샷 읽기·E5 입력 문자열·검증과 공통 상수(E5_MODEL_ID·DIMS·TABLE_NAME) (load_knowledge, passage_text, E5_MODEL_ID)
from day23kit.data import load_knowledge, passage_text, E5_MODEL_ID
# 제공 모듈 common/day23kit/embedding.py: Encoder 선택(fixture 모의 / live 실제 E5)·문서·질문 인코딩·벡터 계약 검사 (get_encoder)
from day23kit.embedding import get_encoder
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result


# ============================================================
# 2. 문서 입력 문자열 목록 만들기
# ============================================================
# BEGIN D23-03-B01
def make_passages(rows):
    # 지식 목록을 같은 순서의 E5 문서 입력 문자열 목록으로 바꾼다.
    # - rows   : 지식 dict 목록 (기본필수 3건, 응용·심화 9건)
    # - 반환값 : 행마다 문자열 1개, 순서 유지 -> i번째 문자열 = rows[i]
    # 예: 101 -> "passage: 실습 파일 보관 실습 파일은 개인 폴더에 보관합니다."
    # [강사용 정답] D23-03-B01 · make_passages
    #     지식 목록을 E5 문서 입력 문자열 목록으로 변환하세요.
    #     문서에는 'passage: ' + 제목 + ' ' + 본문을 사용합니다.
    #     한 행마다 정확히 한 문자열을 만들어 순서를 유지하세요.
    # 해설 D23-03-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 문서마다 고정된 순서로 passage를 구성하여 ID와 행 인덱스를 보존합니다.
    # passage_text(row) : 제공 함수. 'passage: ' + 제목 + ' ' + 본문
    return [passage_text(row) for row in rows]
# END D23-03-B01

# ============================================================
# 3. 질문 입력 문자열 목록 만들기
# ============================================================
# BEGIN D23-03-B02
def make_queries(questions):
    # 질문 목록의 각 질문 앞에 'query: ' 접두어를 붙인다.
    # - questions : 질문 문자열 목록
    # - 반환값    : 같은 순서의 'query: 질문' 목록
    # 예: ['서버는 어디에서 실행하나요?'] -> ['query: 서버는 어디에서 실행하나요?']
    # 주의: 입력에 없는 질문(빈 질문 등)을 끼워 넣지 않는다
    # [강사용 정답] D23-03-B02 · make_queries
    #     질문 목록 각 요소 앞에 query: 공백을 붙이세요.
    #     출력: 입력과 동일한 순서의 문자열 리스트.
    #     빈 질문을 추가해 그럴듯한 벡터를 만들지 마세요.
    # 해설 D23-03-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return ['query: ' + q for q in questions]
# END D23-03-B02

# ============================================================
# 4. 같은 Encoder로 문서·질문 인코딩
# ============================================================
# BEGIN D23-03-B03
def encode_both(encoder, passages, queries):
    # 하나의 Encoder로 문서 입력과 질문 입력을 따로 인코딩해 float32 배열 두 개를 돌려준다.
    # - encoder  : fixture면 SimulatedEncoder(모의, E5 아님), live면 SentenceTransformer E5-small
    # - passages : make_passages() 결과 (n개)
    # - queries  : make_queries() 결과 (1개)
    # - 반환값   : (doc_vectors (n, 384), query_vectors (1, 384))
    # [강사용 정답] D23-03-B03 · encode_both
    #     같은 Encoder에 문서와 질문을 나누어 encode하세요.
    #     반드시 normalize_embeddings=True, convert_to_numpy=True를 전달합니다.
    #     결과: (doc_vectors, query_vectors). 기본필수의 모의 모드와
    #     --mode live의 E5 실측을 구분하여 같은 API를 사용합니다.
    # 해설 D23-03-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 동일 모델로 두 역할을 인코딩합니다. 인코딩은 모델을 새로 학습하지 않습니다.
    # encoder.encode 인자
    # - normalize_embeddings=True : 각 벡터 길이를 1로 맞춤 -> Cosine = 내적
    # - convert_to_numpy=True     : 결과를 NumPy 배열로 받음
    # - batch_size=8              : 문서를 8개씩 묶어 인코딩 (9건이면 2번)
    docs = encoder.encode(passages, normalize_embeddings = True, convert_to_numpy = True, batch_size = 8)
    queries_vec = encoder.encode(queries, normalize_embeddings = True, convert_to_numpy = True)
    # np.asarray(..., dtype=np.float32) : pgvector vector 타입과 같은 32비트 실수로 통일
    return np.asarray(docs, dtype = np.float32), np.asarray(queries_vec, dtype = np.float32)
# END D23-03-B03

# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # 지식을 E5 입력 문자열로 바꾸고 Encoder로 인코딩해 벡터 shape와 입력 예시를 기록한다.
    # - mode='fixture' : SimulatedEncoder (2차원 교육용 좌표를 384차원으로 채운 모의 벡터, E5 아님)
    # - mode='live'    : 실제 intfloat/multilingual-e5-small (최초 실행 시 모델 다운로드)
    # - level          : 1=문서 3건, 2·3=문서 9건 + 대응·계약 검사, 3=모델 정보 기록 추가
    # - 반환값         : outputs/exercise_03.json 에 저장될 dict
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 verify_alignment 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. 입력 준비와 인코딩
    # ----------------------------------------------------
    # rows : 기본필수는 앞 3건(101~103), 응용·심화는 9건 전체
    rows = load_knowledge()[:3] if level == 1 else load_knowledge()
    # get_encoder(mode) : 제공 함수. fixture -> SimulatedEncoder, live -> SentenceTransformer(E5, CPU)
    encoder = get_encoder(mode)
    passages = make_passages(rows)
    questions = make_queries(['서버는 어디에서 실행하나요?'])
    # docs : (3, 384) 또는 (9, 384) / queries : (1, 384)
    docs, queries = encode_both(encoder, passages, questions)
    # ----------------------------------------------------
    # 5-2. 결과 dict 구성
    # ----------------------------------------------------
    # result 키 : 의미
    # - source_mode     : 'e5_live'(실제 E5) 또는 'simulated'(모의)
    # - model_id        : 실제 모델 ID 또는 SIMULATED_2D_PADDED_TO_384_NOT_E5
    # - document_shape  : [3, 384] 또는 [9, 384]
    # - query_shape     : [1, 384]
    # - example_passage : 첫 문서 입력 "passage: 실습 파일 보관 실습 파일은 개인 폴더에 보관합니다."
    # - example_query   : "query: 서버는 어디에서 실행하나요?"
    result = dict(exercise = 3, level = level, source_mode = 'e5_live' if mode == 'live' else 'simulated',
                  model_id = E5_MODEL_ID if mode == 'live' else encoder.model_id,
                  document_shape = list(docs.shape), query_shape = list(queries.shape),
                  example_passage = passages[0], example_query = questions[0])
    # ----------------------------------------------------
    # 5-3. 응용·심화: 대응·계약 검사와 모델 정보
    # ----------------------------------------------------
    # alignment : [(101, 0), ..., (109, 8)] / vectors_valid : True
    if level >= 2:
        result['alignment'] = verify_alignment(rows, docs)
        result['vectors_valid'] = check_shapes_vectors(docs, queries, len(rows))
    # model_contract : make_model_stamp() 결과 dict
    if level >= 3:
        result['model_contract'] = make_model_stamp(encoder, mode)
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_03.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 03 · E5 문장 Embedding')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_03.json', result)
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
