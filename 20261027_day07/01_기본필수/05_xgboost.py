"""실습 05: XGBoost도 같은 fit·predict 흐름으로 실행합니다."""
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
# 부스팅 분류 후보를 만들기 위해 외부 라이브러리 xgboost에서 XGBClassifier를 가져온다. 
# scikit-learn처럼 fit·predict·predict_proba를 제공해 Pipeline에 그대로 넣을 수 있다.
from xgboost import XGBClassifier
# 모델 성능을 f1 score로 재기 위해 외부 라이브러리 scikit-learn에서 f1_score를 가져온다.
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# 모델 생성 / 학습 / 평가
# -----------------------------------------------------------------------------
# 사전 고정된 하이퍼파라미터(회차 80, 깊이 2, 학습률 0.05, hist 방식, CPU)로 XGBoost 분류기를 생성한다.
# n_estimators=80은 보정 트리를 80회 차례로 쌓는다. 
# max_depth=2는 보정 트리 하나가 질문을 최대 2번까지만 하게 해 복잡도를 낮춘다. 
# learning_rate=0.05는 한 회차의 보정을 5%만 반영해 천천히 고친다.
# objective="binary:logistic"은 0/1 이진 분류용 손실을 학습 목표로 삼아 실패 확률을 출력하게 한다. 
# eval_metric="logloss"는 학습 진단용 손실 지표이며, 후보 선택 기준인 F1과는 다른 값이다.
# tree_method="hist"는 값을 구간(히스토그램)으로 묶어 분할 후보를 빠르게 찾는다. 
# random_state=42는 subsample의 무작위 행 선택을 고정해 결과를 재현한다. 
# n_jobs=1은 CPU 1개만 써서 환경이 달라도 같은 결과를 낸다.
model = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=42, n_jobs=1)
# 학습
model.fit(X_train, y_train)
# 예측
prediction = model.predict(X_valid)

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
print("valid 예측:", prediction.tolist())
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
print("학습 목적: 실패 분류 / 확인 지표: F1")
