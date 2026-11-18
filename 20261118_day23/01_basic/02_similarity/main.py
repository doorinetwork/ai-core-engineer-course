"""Day23 실습 02 · Vector·Similarity·Distance (기본필수, mentor).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 02 기본필수: (1,0)과 (0.8,0.6)의 내적·Cosine·L2를 직접 계산합니다.
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
# 제공 모듈 common/day23kit/mathops.py: 2차원 벡터 검사와 Cosine distance 계산 도우미 (cosine_distance)
from day23kit.mathops import cosine_distance
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result

# ============================================================
# 2. 내적(Dot product)
# ============================================================
# BEGIN D23-02-B01
def dot(a, b):
    # 같은 위치의 좌표끼리 곱해 모두 더한 내적을 float으로 돌려준다.
    # - a, b   : 같은 길이의 NumPy 1차원 배열
    # - 반환값 : float
    # 예: (1, 0)·(0.8, 0.6) = 1×0.8 + 0×0.6 = 0.8
    # [강사용 정답] D23-02-B01 · dot
    #     같은 위치의 숫자를 곱해 합산하는 내적을 구현하세요.
    #     예: (1,0)·(0.8,0.6)=0.8. 입력 NumPy 배열 2개, 출력 float.
    # 해설 D23-02-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 좌표별 곱을 모두 합산. 내적은 벡터 크기에도 영향을 받습니다.
    # float(...) : NumPy 스칼라(np.float64)를 파이썬 float으로 바꿔 JSON 저장이 쉽게 함
    return float(np.dot(a, b))
# END D23-02-B01

# ============================================================
# 3. Cosine similarity
# ============================================================
# BEGIN D23-02-B02
def cosine(a, b):
    # 내적을 두 벡터 길이의 곱으로 나눈 Cosine similarity를 돌려준다.
    # - 정상 입력        : -1 ~ 1 사이 float (1에 가까울수록 방향이 같음)
    # - 영벡터가 하나라도 : ValueError('zero_vector') (길이 0으로 나눌 수 없음)
    # 예: q=(1, 0), A=(0.8, 0.6) -> 0.8 / (1 × 1) = 0.8
    # [강사용 정답] D23-02-B02 · cosine
    #     내적을 두 벡터 길이의 곱으로 나누어 Cosine similarity를 구하세요.
    #     q=(1,0), A=(0.8,0.6)라면 0.8입니다.
    #     입력: NumPy 배열 2개. 출력: float. 영벡터는 무조건 거부합니다.
    # 해설 D23-02-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # ① 길이 ② 분모 확인 ③ 정규화한 내적.
    # np.linalg.norm(x) : 벡터 길이 sqrt(x1² + x2² + ...) -> (0.8, 0.6)의 길이는 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    # 1e-12 이하 : 사실상 0인 길이로 보고 거부 (부동소수점 오차 고려)
    if na <= 1e-12 or nb <= 1e-12:
        raise ValueError('zero_vector')
    return dot(a, b) / (na * nb)
# END D23-02-B02

# ============================================================
# 4. L2(유클리드) 거리
# ============================================================
# BEGIN D23-02-B03
def l2(a, b):
    # 두 벡터의 좌표 차이 벡터의 길이(L2 거리)를 돌려준다. 값이 작을수록 가깝다.
    # - a, b   : 같은 길이의 NumPy 1차원 배열
    # - 반환값 : 0 이상의 float
    # 예: (1, 0) - (0.8, 0.6) = (0.2, -0.6) -> sqrt(0.04 + 0.36) = sqrt(0.4) ≈ 0.632456
    # [강사용 정답] D23-02-B03 · l2
    #     두 벡터의 좌표 차이로 L2 거리를 구하세요. 출력 float.
    #     q=(1,0), A=(0.8,0.6) → sqrt(0.4) ≈ 0.6325.
    # 해설 D23-02-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 좌표 간 차이의 길이: 값이 작을수록 가깝습니다.
    return float(np.linalg.norm(a - b))
# END D23-02-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # 예제 벡터로 Dot·Cosine·L2·Cosine distance를 계산해 결과 dict로 모은다.
    # - mode   : 'fixture' 또는 'live' (이 실습은 로컬 수학 계산만 하므로 두 경로가 같다)
    # - level  : 1=기본필수, 2=응용(정규화·순위), 3=심화(입력 검증·오류 구분)
    # - 반환값 : outputs/exercise_02.json 에 저장될 dict
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 normalize 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. 예제 벡터 준비
    # ----------------------------------------------------
    # q : 질문 벡터 (1, 0) / a : 문서 A (0.8, 0.6)
    # u : 길이 1인 (1, 0) / v : 길이가 큰 (10, 10) -> Dot과 Cosine 순위 차이 확인용
    q = np.array([1., 0.])
    a = np.array([.8, .6])
    u = np.array([1., 0.])
    v = np.array([10., 10.])
    # ----------------------------------------------------
    # 5-2. 기본 계산 결과
    # ----------------------------------------------------
    # result 키 : 의미
    # - source_mode     : 'local_math' = 로컬 수학 계산 (모델·DB 미사용)
    # - dot             : 0.8
    # - cosine          : 0.8 (클수록 가깝다)
    # - l2              : 0.632456 (작을수록 가깝다)
    # - cosine_distance : 1 - cosine = 0.2 (pgvector <=> 연산자가 돌려주는 값과 같은 정의, 작을수록 가깝다)
    result = dict(exercise = 2, level = level, source_mode = 'local_math',
                  dot = round(dot(q, a), 6), cosine = round(cosine(q, a), 6), l2 = round(l2(q, a), 6),
                  cosine_distance = round(1 - cosine(q, a), 6))
    # ----------------------------------------------------
    # 5-3. 응용: 정규화와 순위 비교
    # ----------------------------------------------------
    # normalized_v : (10, 10) -> [0.707107, 0.707107]
    # rankings     : {'dot': ['V', 'U'], 'cosine': ['U', 'V']}
    if level >= 2:
        result['normalized_v'] = normalize(v).round(6).tolist()
        result['rankings'] = rank_dot_vs_cosine(q, {'U': u, 'V': v})
    # ----------------------------------------------------
    # 5-4. 심화: 잘못된 입력 세 가지를 일부러 넣어 오류 이름 기록
    # ----------------------------------------------------
    if level >= 3:
        result['validated'] = validate_pair(q, a)
        errors = {}
        # (라벨, 첫 벡터, 둘째 벡터) 목록
        # - dimension_mismatch : 2차원 q 와 3차원 [1, 0, 2]
        # - nonfinite          : NaN 포함 [nan, 1]
        # - zero_vector        : 영벡터 [0, 0]
        for label, x, y in [('dimension_mismatch', q, np.array([1., 0., 2.])),
                            ('nonfinite', q, np.array([np.nan, 1.])),
                            ('zero_vector', q, np.array([0., 0.]))]:
            # 오류 메시지를 라벨별로 저장 -> {'dimension_mismatch': 'dimension_mismatch', ...}
            try:
                validate_pair(x, y)
            except ValueError as e:
                errors[label] = str(e)
        result['invalid_inputs'] = errors
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_02.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 02 · Vector·Similarity·Distance')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_02.json', result)
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
