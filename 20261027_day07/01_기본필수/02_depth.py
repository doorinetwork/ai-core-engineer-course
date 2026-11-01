"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""
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
# valid에서 input_chars 열을 X_train으로 저장한다.
X_valid = valid[["input_chars"]]
# train에서 failed 열을 X_train으로 저장한다.
y_train = train["failed"]
# valid에서 failed 열을 X_train으로 저장한다.
y_valid = valid["failed"]

# -----------------------------------------------------------------------------
# 모델 도구 불러오기
# -----------------------------------------------------------------------------
# Decision Tree를 만들기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 모델 성능을 f1 score로 재기 위해 외부 라이브러리 scikit-learn에서 f1_score를 가져온다.
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# 깊이별 모델 생성 / 학습 / 평가
# -----------------------------------------------------------------------------
# 비교할 깊이 1과 3을 정해두고 하나씩 꺼내 아래 학습 및 평가를 반복한다.
for depth in [1, 3]:
    # depth를 전달하여 해당 깊이에 맞는 decision tree를 생성한다.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # 학습
    model.fit(X_train, y_train)
    # 예측 후 train과 valid의 f1 score 생성
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")
