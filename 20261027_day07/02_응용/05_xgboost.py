"""실습 05 응용: PPT와 같은 XGBoost 80회차 분류 모델입니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다. 인덱스는 request_id로 정한다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# data/split_manifest.csv에 정의된 data 분할 규칙과 position 순서를 읽어 manifest에 저장한다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# -----------------------------------------------------------------------------
# 데이터 분할
# -----------------------------------------------------------------------------
# train에 해당하는 request_id를 읽어 position순으로 정렬해 train_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid에 해당하는 request_id를 읽어 position순으로 정렬해 valid_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# raw로부터 train_ids 순서로 할당된 train 데이터를 꺼낸다.
train = raw.loc[train_ids]
# raw로부터 valid_ids 순서로 할당된 valid 데이터를 꺼낸다.
valid = raw.loc[valid_ids]
# 모델 입력에 쓸 열들을 FEATURES에 리스트로 저장한다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# 각 train, valid 데이터에서 입력에 쓸 열들만 추려서 X_train, X_valid로 저장한다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# 각 train, valid 데이터에서 정답에 쓸 failed 열을 추려서 y_train, y_valid로 저장한다.
y_train, y_valid = train["failed"], valid["failed"]
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.

# -----------------------------------------------------------------------------
# 경로 지정
# -----------------------------------------------------------------------------
# 현재의 Python 파일이 존재하는 경로 아래에 outputs 디렉토리를 만드는 경로를 OUT에 저장한다.
OUT = HERE / "outputs"
# 디렉토리가 이미 존재한다면 무시한다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 전처리기
# -----------------------------------------------------------------------------
# 결측치를 채우기 위해 외부 라이브러리 scikit-learn에서 SimpleImputer를 가져온다.
from sklearn.impute import SimpleImputer
# 범주형 데이터의 One-hot 인코딩을 위해 외부 라이브러리 scikit-learn에서 OneHotEncoder를 가져온다.
from sklearn.preprocessing import OneHotEncoder
# 각 열마다 다른 전처리를 적용하기 위해 외부 라이브러리 scikit-learn에서 ColumnTransformer를 가져온다.
from sklearn.compose import ColumnTransformer
# 전처리와 모델을 한 흐름으로 묶기 위해 외부 라이브러리 scikit-learn에서 Pipeline을 가져온다.
from sklearn.pipeline import Pipeline
# 파이프라인 내부의 전처리기 상태를 초기화된 새 객체로 복제하기 위해 외부 라이브러리 scikit-learn에서 clone을 가져온다.
from sklearn.base import clone

# 각 열마다 다른 전처리 기법을 사용해 pre에 저장한다.
# 수치형 2열의 결측을 train 중앙값으로 채운다. 범주형은 최빈값 대체를 사용한다.
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않는다. 각 후보·fold마다 새 전처리기를 학습한다.

# -----------------------------------------------------------------------------
# 학습 · 예측 · 평가
# -----------------------------------------------------------------------------
# 부스팅 분류 후보를 만들기 위해 외부 라이브러리 xgboost에서 XGBClassifier를 가져온다. 
# scikit-learn처럼 fit·predict·predict_proba를 제공해 Pipeline에 그대로 넣을 수 있다.
from xgboost import XGBClassifier
# 모델 성능 측정을 위해 외부 라이브러리 scikit-learn에서 평가 함수 3개를 가져온다. 
# f1_score: 정밀도와 재현율의 조화평균
# confusion_matrix: TN·FP·FN·TP 개수
# log_loss: 예측 확률이 정답에서 얼마나 벗어났는가
from sklearn.metrics import f1_score, confusion_matrix, log_loss

# 사전 고정된 하이퍼파라미터(회차 80, 깊이 2, 학습률 0.05, hist 방식, CPU)로 XGBoost 분류기를 생성한다.
# n_estimators=80은 보정 트리를 80회 차례로 쌓는다. 
# max_depth=2는 보정 트리 하나가 질문을 최대 2번까지만 하게 해 복잡도를 낮춘다. 
# learning_rate=0.05는 한 회차의 보정을 5%만 반영해 천천히 고친다.
# subsample=0.8은 회차마다 train 행의 80%만 무작위로 써서 과적합을 줄인다. 
# colsample_bytree=1.0은 트리마다 특성을 전부 쓴다. 
# reg_lambda=1.0은 잎 가중치에 L2 규제를 걸어 값이 지나치게 커지지 않게 한다.
# objective="binary:logistic"은 0/1 이진 분류용 손실을 학습 목표로 삼아 실패 확률을 출력하게 한다. 
# eval_metric="logloss"는 학습 진단용 손실 지표이며, 후보 선택 기준인 F1과는 다른 값이다.
# tree_method="hist"는 값을 구간(히스토그램)으로 묶어 분할 후보를 빠르게 찾는다. 
# device="cpu"는 GPU 없이 CPU로 학습한다. 
# random_state=42는 subsample의 무작위 행 선택을 고정해 결과를 재현한다. 
# n_jobs=1은 CPU 1개만 써서 환경이 달라도 같은 결과를 낸다.
estimator = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1)
# 전처리기를 독립 상태로 복제하여 XGBoost 모델과 함께 Pipeline으로 묶는다.
# clone(pre)를 쓰므로 다른 실습·fold에서 학습된 통계가 섞이지 않는다. 
model = Pipeline([("pre", clone(pre)), ("model", estimator)])
# 학습
model.fit(X_train, y_train)
# 예측
prediction = model.predict(X_valid)
# predict_proba를 호출하여 클래스 1(실패)에 대한 예측 확률 배열을 추출한다.
# predict_proba는 [0일 확률, 1일 확률] 두 열을 돌려주고, [:, 1]은 그중 두 번째 열인 실패 확률을 고른다.
probability = model.predict_proba(X_valid)[:, 1]  # classes_ = [0, 1] 확인 후 실패1의 열 사용
# 모델이 기억한 클래스 순서를 출력한다. 
print("클래스 순서:", model.classes_.tolist())
# 혼동행렬을 통해 오차 종류(FP, FN)를 분리하여 집계한다.
tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
# 결과 지표 한 행을 result에 저장한다. train_F1은 학습한 데이터를 다시 맞힌 점수로, valid_F1과의 차이를 보고 과적합을 진단한다. 
# zero_division=0은 실패를 하나도 예측하지 않아 분모가 0일 때 경고 대신 0을 쓴다. 
# valid_F1은 처음 보는 데이터의 점수로 후보 선택의 주 지표이다. FP는 성공을 실패로 잘못 경보한 수, FN은 실제 실패를 놓친 수이다.
# valid_logloss는 실패 확률이 실제 정답과 얼마나 어긋났는지를 잰 진단용 손실이다. 작을수록 좋다. 
# labels=[0, 1]은 valid에 한 클래스만 있어도 두 클래스를 기준으로 계산하게 한다.
result = pd.DataFrame([{"train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                       "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn,
                       "valid_logloss": log_loss(y_valid, probability, labels=[0, 1])}])

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
result.to_csv(OUT / "05_xgboost.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.")
