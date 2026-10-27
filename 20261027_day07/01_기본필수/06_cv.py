"""실습 06: train 12건 안에서만 3-fold 교차검증합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 데이터를 train / test / valid로 나누기 위해 외부 라이브러리 scikit-learn의 train_test_split을 import한다.
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다.
raw = pd.read_csv(HERE / "requests.csv")

# -----------------------------------------------------------------------------
# train (60%) / valid (20%) / test (20%) 분할
# -----------------------------------------------------------------------------
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.

# -----------------------------------------------------------------------------
# 입력(X)과 정답(y) 분리
# -----------------------------------------------------------------------------
# train에서 input_chars 열을 X_train으로 저장한다.
X_train = train[["input_chars"]]
# valid에서 input_chars 열을 X_valid으로 저장한다.
X_valid = valid[["input_chars"]]
# train에서 failed 열을 y_train으로 저장한다.
y_train = train["failed"]
# valid에서 failed 열을 y_valid으로 저장한다.
y_valid = valid["failed"]

# -----------------------------------------------------------------------------
# 모델 도구 불러오기
# -----------------------------------------------------------------------------
# 단일 규칙 기반의 의사결정트리 분류기 모델을 생성하기 위해 DecisionTreeClassifier 클래스를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 교차검증을 수행하여 각 fold의 평가지표 점수를 계산하기 위해 cross_val_score 함수를 가져온다
from sklearn.model_selection import StratifiedKFold, cross_val_score

# train 12건의 클래스 비율을 유지하면서 3개 fold(학습 8건, 평가 4건)로 분할하도록 설정한다.
folds = StratifiedKFold(
    n_splits=3,         # 전체 데이터를 3개의 Fold(부분 집합)로 분할
                        # 3번의 교차검증을 수행하며, 매번 2개 Fold는 학습용,
                        # 나머지 1개 Fold는 검증용으로 사용
                        # 각 Fold가 한 번씩 검증 데이터로 사용됨

    shuffle=True,       # 데이터를 Fold로 나누기 전에 무작위로 섞음
                        # 원본 데이터의 정렬 순서가 분할 결과에 미치는 영향을 줄임
                        # 단, 각 클래스의 비율은 최대한 유지하도록 분할

    random_state=42     # 데이터 셔플에 사용할 난수 시드 고정
                        # 코드를 반복 실행해도 동일한 Fold 구성이 생성되도록 설정
                        # shuffle=True일 때 적용됨
)
# 최대 깊이를 3으로 제한하고 난수 시드를 고정한 결정 트리 모델을 생성한다.
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# cross_val_score는 fold마다 model을 복제해 학습 8건으로 전처리기와 분류기를 함께 fit하고, 평가 4건은 변환·예측만 한다. 
# 그래서 평가 4건의 값이 중앙값·최빈값 계산에 섞이지 않는다.
# 전처리만 train 12건으로 미리 fit하고 모델만 교차검증하면 평가 fold의 정보가 전처리 통계에 새어 들어가므로, Pipeline 전체를 넘겨야 한다.
# cv=folds는 위에서 고정한 같은 fold 5개를 세 후보에 똑같이 쓴다. scoring="f1"은 실패(1)를 양성으로 본 F1을 점수로 쓴다. 점수 순서는 folds 순서와 같다.
scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
# 3개 fold 각각에서 계산된 F1 점수 리스트를 소수점 셋째 자리까지 출력한다.
print("fold별 F1:", scores.round(3).tolist())
# 3개 fold 점수의 산술 평균값을 출력한다.
print("평균 F1:", round(scores.mean(), 3))
# fold 점수들의 기술적 산포를 나타내는 표준편차를 출력한다.
print("표준편차:", round(scores.std(), 3))
