"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
# Path 메소드를 사용하기 위한 pathlib 라이브러리 import
from pathlib import Path
# 외부 라이브러리 pandas import
import pandas as pd

# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다. 인덱스열은 request_id로 한다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 분할을 위한 분할표를 읽어 split에 저장한다. (type:DataFrame)
# 열: request_id, 
#     split(train/valid/test), 
#     position(분할 안에서의 행 순서)
# split에 저장된 train / valid / test 비율에 따라 추후 분할을 수행한다.
split = pd.read_csv(HERE / "data/split_manifest.csv")

# split이 "train"인 행만 골라 position 순으로 정렬한 뒤, 
# request_id 열만 꺼내 train_ids로 저장한다.
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
# split이 "valid"인 행만 골라 position 순으로 정렬한 뒤, 
# request_id 열만 꺼내 valid_ids로 저장한다.
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
# raw로부터 train_ids에 해당하는 행만 뽑아 train으로 저장한다.
train = raw.loc[train_ids]
# raw로부터 valid_ids에 해당하는 행만 뽑아 valid로 저장한다.
valid = raw.loc[valid_ids]
# 모델에 입력으로 쓸 열 4개를 features에 리스트로 저장한다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]
# train에서 위의 features 입력 4열만 골라 X_train에 저장한다.
X_train = train[features]
# valid에서 위의 features 입력 4열만 골라 X_valid에 저장한다.
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.

# 분할 이후 학습 및 비교를 위한 scikit-learn import
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# 열마다 다른 전처리를 한 번에 적용하는 ColumnTransformer를 import한다.
from sklearn.compose import ColumnTransformer
# 전처리와 모델을 하나의 순서로 묶는 Pipeline을 import한다.
from sklearn.pipeline import Pipeline
# 학습 상태를 복사하지 않고 설정만 새로 복제하는 clone을 import한다.
from sklearn.base import clone

# baseline 모델: 분류용 DummyClassifier, 회귀용 DummyRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
# 선형 모델: 회귀용 LinearRegression, 분류용 LogisticRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
# K-최근접 이웃 모델: 분류용, 회귀용
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# 평가 지표: MAE, RMSE(회귀) / Precision, Recall, F1, 혼동행렬(분류)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 수치형 Pipeline: 결측은 train 중앙값으로 채운 뒤 평균 0, 표준편차 1로 표준화한다. (type: Pipeline)
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
# 범주형 Pipeline: 결측은 train 최빈값으로 채운 뒤 범주마다 0/1 열로 바꾼다. (type: Pipeline)
# handle_unknown="ignore": train에 없던 범주가 valid에 나와도 오류 대신 모든 열을 0으로 처리한다.
# sparse_output=False: 희소행렬이 아니라 일반 numpy 배열로 출력한다.
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
# ColumnTransformer는 열마다 다른 전처리기법ㄹ을 사용한다:
# features[:2] = 수치 2열, features[2:] = 범주 2열. 변환 결과는 수치 2 + 원핫 6 = 8열이다.
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 회귀 비교표의 행을 모으기 위해 빈 리스트를 생성한다.
regression_rows = []
# 분류 비교표의 행을 모으기 위해 빈 리스트를 생성한다.
classification_rows = []
# 모델별 valid 예측표를 모으기 위해 빈 리스트를 생성한다.
predictions = []

# -----------------------------------------------------------------------------
# 회귀: 실행시간(elapsed_ms) 예측
# -----------------------------------------------------------------------------
# (이름, 모델) 3개를 차례로 돈다. 
# Dummy는 train 평균만 내놓는 기준선이라, 나머지 모델이 이보다 낮은 오차를 내야 의미가 있다.
# KNN15의 15는 실습06에서 valid RMSE가 가장 낮았던 K이다.
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    # 전처리와 모델을 하나로 묶은 Pipeline을 만든다. (type: Pipeline)
    # clone(pre): 모델마다 전처리기를 새로 복제해, 앞 모델의 학습 통계가 다음 모델로 새지 않게 한다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 입력(X_train)과 정답(elapsed_ms)으로 학습한다.
    model.fit(X_train, train["elapsed_ms"])
    # 학습에 쓰지 않은 valid 입력으로 예측한다.
    # 전처리기는 transform만 하고 통계를 다시 계산하지 않는다. (valid 정보가 학습에 새는 것을 막음)
    prediction = model.predict(X_valid)
    # 회귀 비교표에 한 행을 append한다. (type: dict)
    # MAE_ms: 평균 절대 오차(ms). 오차의 평균 크기라 이해하기 쉽다.
    # RMSE_ms: 제곱 후 평균의 제곱근(ms). 큰 오차에 더 큰 벌점을 주며, 이 프로젝트의 회귀 선택 기준이다.
    # 두 함수 모두 인자 순서가 (실제값, 예측값)이다.
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
    # 이 모델의 예측을 ID와 함께 표로 만들어 모아 둔다.
    # valid.index는 request_id이다. (위에서 set_index를 했기 때문)
    # "task"와 "model"에 문자열 하나를 주면 모든 행에 같은 값이 채워진다.
    # .to_numpy(): Series를 배열로 바꿔 index 정렬 없이 위치 순서대로 넣는다.
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

# -----------------------------------------------------------------------------
# 분류: 실패 여부(failed, 0/1) 예측
# -----------------------------------------------------------------------------
# Dummy(most_frequent)는 항상 다수 클래스만 찍는 기준선이다. 실패가 적으면 거의 전부 0을 예측해 F1이 0이 된다.
# C=1.0: 정규화 세기(기본값). max_iter=1000: 수렴 전에 멈춰 경고가 나지 않도록 반복 횟수를 늘린다.
# KNN3의 3은 실습06에서 valid F1이 가장 높았던 K이다.
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    # 회귀와 같은 방식으로 모델마다 새 전처리기를 가진 Pipeline을 만든다. (type: Pipeline)
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 정답은 0/1 정수인 failed(Series 60개)이다. 흐름은 회귀와 같고 정답 열만 다르다.
    model.fit(X_train, train["failed"])
    # 학습에 쓰이지 않은 valid 입력으로 예측한다.
    prediction = model.predict(X_valid)
    # 혼동행렬을 1차원으로 펴서 네 값으로 풀어 담는다. 
    # labels=[0, 1]: valid에 한 클래스만 있어도 항상 2×2로 만들어 풀기(unpack) 오류를 막는다.
    # 순서는 TN(실제0·예측0), FP(실제0·예측1: 오탐), FN(실제1·예측0: 놓친 실패), TP(실제1·예측1)이다.
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    # 분류 비교표에 한 행을 append한다.
    # F1: 정밀도와 재현율의 조화평균. 이 프로젝트의 분류 선택 기준이며 실패(1)를 양성으로 본다.
    # Precision: 실패라고 예측한 것 중 실제 실패 비율 / Recall: 실제 실패 중 맞힌 비율
    # zero_division=0: 양성 예측이 하나도 없어 분모가 0이면 경고 대신 0으로 처리한다.
    classification_rows.append({"model": name, "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    # 이 모델의 valid 예측표를 모아 둔다.
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

# -----------------------------------------------------------------------------
# 결과 저장
# -----------------------------------------------------------------------------
# dict 리스트를 DataFrame으로 바꾼다. 
# 회귀(MAE·RMSE: 낮을수록 좋음)와 분류(F1: 높을수록 좋음)는 방향이 달라 표를 섞지 않고 따로 둔다.
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
# 결과 폴더 "outputs"가 들어갈 경로를 만든다. (type: Path) 
out = HERE / "outputs"
# 폴더가 없으면 만든다. exist_ok=True: 이미 있어도 오류를 내지 않는다.
out.mkdir(exist_ok=True)

# index=False: 행 번호 열은 저장하지 않는다.
# encoding="utf-8-sig": 엑셀에서 열어도 한글이 깨지지 않도록 BOM이 붙은 UTF-8로 저장한다.
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# predictions의 DataFrame 6개를 세로로 이어 붙여 하나의 표로 저장한다.
# ignore_index=True: 합친 뒤 행 번호를 0부터 새로 매긴다.
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
print("회귀 / valid 20건")
# 소수 셋째 자리까지 반올림하고 행 번호 없이 표 형태로 출력한다.
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
# 소수 셋째 자리까지 반올림하고 행 번호 없이 표 형태로 출력한다.
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
