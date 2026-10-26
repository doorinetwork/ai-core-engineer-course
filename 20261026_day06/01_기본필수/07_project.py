"""실습 07~08 기본필수: 같은 파일을 완성하고 다시 실행합니다."""
# Path 메소드를 사용하기 위한 pathlib 라이브러리 import
from pathlib import Path
# 외부 라이브러리 pandas import
import pandas as pd
# 외부 라이브러리 scikit-learn import
from sklearn.model_selection import train_test_split

# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 data에 저장한다.
raw = pd.read_csv(HERE / "requests.csv")

# 1차 분할: test 20% 
develop, test = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["failed"])
# 2차 분할: test를 뺀 나머지(develop)에서 25%를 valid에 할당: 
# (data 20행일 때) 최종적으로 train 12, valid 4, test 4의 구성이 된다.
train, valid = train_test_split(develop, test_size=0.25, random_state=42, stratify=develop["failed"])

# 분할 이후 학습 및 비교를 위한 scikit-learn import
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score

# feature 선택: 학습에 사용할 독립변수들의 목록 (type: list)
# elapsed_ms와 failed는 정답이므로 제외하지 않으면 데이터 누수가 발생
features = ["input_chars", "task_type"]
# 훈련용 입력 (type:DataFrame)
X_train = train[features]
# 검증용 입력 (type:DataFrame)
X_valid = valid[features]

# 전처리기
# ColumnTransformer를 통해 수치형 input_chars는 StandardScaler (평균 0, 표준편차 1),
# 범주형 task_type은 One-hot 인코딩으로 변환한다.
pre = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

# 뒤의 코드에서 append해주기 위해 미리 빈 리스트를 생성한다.
regression_rows = []
classification_rows = []
# copy()를 사용해 valid 원본이 변경되는 것을 막는다.
result = valid[["request_id", "elapsed_ms", "failed"]].copy()
# 회귀: 실행시간 예측
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 실행시간 elapsed_ms로 학습한다. (수치형)
    model.fit(X_train, train["elapsed_ms"])
    # X_valid를 입력받아 예측값을 prediction에 저장한다. valid는 학습에 사용하지 않고 예측에 사용한다.
    prediction = model.predict(X_valid)
    # MAE(mean absolute error)값을 regression_rows에 저장한다. [회귀]
    regression_rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction)})
    result[name + "_ms"] = prediction
# 분류: 성공/실패 여부 예측
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(max_iter=1000))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 실행시간 failed로 학습한다. (범주형)
    model.fit(X_train, train["failed"])
    # X_valid를 입력받아 예측값을 prediction에 저장한다. valid는 학습에 사용하지 않고 예측에 사용한다.
    prediction = model.predict(X_valid)
    # F1 score 값을 classification_rows에 저장한다. [분류]
    classification_rows.append({"model": name, "F1": f1_score(valid["failed"], prediction, zero_division=0)})
    result[name + "_failed"] = prediction

# 회귀와 분류의 값을 저장
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)

# 표로 (CSV) 변환하여 지정된 각 파일에 저장한다.
regression.to_csv(HERE / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
result.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print("회귀 비교 / 낮은 MAE가 좋음")
print(regression.round(3).to_string(index=False))
print("분류 비교 / 높은 F1이 좋음")
print(classification.round(3).to_string(index=False))
print("저장: regression_comparison.csv, classification_comparison.csv, validation_predictions.csv")
print("test는 학습·예측하지 않았습니다.")
