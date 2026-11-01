"""실습 03 심화: 회귀 Pipeline 두 개를 같은 KFold로 비교합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다. 인덱스는 request_id로 정한다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# data/split_manifest.csv에 정의된 data 분할 규칙과 position 순서를 읽어 manifest에 저장한다.
# manifest (type: DataFrame): 각 request_id가 train인지 valid인지 적어 둔 분할표이다.
#   - 분할을 코드에서 매번 새로 하지 않고 파일로 고정해 두면, 누가 언제 실행해도 같은 train/valid로 실험하게 된다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# -----------------------------------------------------------------------------
# 데이터 분할
# -----------------------------------------------------------------------------
# train에 해당하는 request_id를 읽어 position순으로 정렬해 train_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
# valid에 해당하는 request_id를 읽어 position순으로 정렬해 valid_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
# raw에서 train_ids와 valid_ids에 해당하는 행만 꺼내 각각 train, valid에 저장한다.
# train, valid: copy()로 원본 raw와 분리된 복사본을 만들어, 이후 수정이 raw에 영향을 주지 않게 한다.
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
# 모델 입력에 쓸 열들을 FEATURES에 리스트로 저장한다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# 수치형 2열과 범주형 2열로 FEATURES를 분리한다.
NUM, CAT = FEATURES[:2], FEATURES[2:]
# train과 valid에서 입력 열만 골라 X_train, X_valid에 저장한다.
# X_train (type: DataFrame, 60행×4열), X_valid (type: DataFrame, 20행×4열): 모델에 들어갈 입력 표이다.
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
# y_train (type: Series, 60개), y_valid (type: Series, 20개): 정답인 실행시간(ms)이다. 회귀 문제이므로 정답이 연속된 숫자이다.
y_train, y_valid = train["elapsed_ms"], valid["elapsed_ms"]

# -----------------------------------------------------------------------------
# 출력 경로 지정
# -----------------------------------------------------------------------------
# 현재의 Python 파일이 존재하는 경로 아래에 outputs 디렉토리를 만드는 경로를 OUT에 저장한다.
OUT = HERE / "outputs"
# 디렉토리가 이미 존재한다면 무시한다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 전처리기
# -----------------------------------------------------------------------------
# 각 열마다 다른 전처리를 적용하기 위해 외부 라이브러리 scikit-learn에서 ColumnTransformer를 가져온다.
from sklearn.compose import ColumnTransformer
# 전처리와 모델을 한 흐름으로 묶기 위해 외부 라이브러리 scikit-learn에서 Pipeline을 가져온다.
from sklearn.pipeline import Pipeline
# 결측치를 채우기 위해 외부 라이브러리 scikit-learn에서 SimpleImputer를 가져온다.
from sklearn.impute import SimpleImputer
# 숫자 크기를 맞추고 범주를 숫자로 바꾸기 위해 외부 라이브러리 scikit-learn에서 StandardScaler와 OneHotEncoder를 가져온다.
# 기능: StandardScaler는 열마다 평균 0, 표준편차 1로 맞추고, OneHotEncoder는 범주 하나를 0/1 열 여러 개로 바꾼다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# pre: 숫자 열 처리 흐름("num")과 범주 열 처리 흐름("cat")을 묶은 전처리기이다.
# 숫자 열(NUM): 결측치를 중앙값으로 채운 뒤 표준화한다.
#   strategy="median": 평균보다 극단값에 덜 흔들리는 중앙값을 쓴다.
#   keep_empty_features=True: 어떤 fold의 학습 부분에서 한 열이 전부 비어 있어도 그 열을 버리지 않고 남겨, 열 개수가 fold마다 달라지지 않게 한다.
# 범주 열(CAT): 결측치를 최빈값으로 채운 뒤 one-hot 인코딩한다.
#   handle_unknown="ignore": 학습 부분에 없던 범주가 평가 부분에 나오면 에러 대신 모든 one-hot 열을 0으로 둔다.
#   sparse_output=False: 결과를 0이 많은 희소 행렬이 아니라 일반 numpy 배열로 돌려준다.
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

# -----------------------------------------------------------------------------
# 모델과 교차검증 준비
# -----------------------------------------------------------------------------
# [외부 라이브러리] scikit-learn의 clone: 설정은 같고 학습 기록은 없는 새 복사본을 만들기 위해 불러온다.
#   두 모델이 pre 객체 하나를 함께 쓰면 서로의 학습 결과가 섞일 수 있으므로, 모델마다 깨끗한 전처리기를 따로 준다.
from sklearn.base import clone
# [외부 라이브러리] scikit-learn의 LinearRegression: 입력과 정답 사이를 직선(평면) 관계로 가정해 예측하는 선형회귀 모델이다.
from sklearn.linear_model import LinearRegression
# [외부 라이브러리] scikit-learn의 KNeighborsRegressor: 입력이 가장 비슷한 이웃 k개의 정답 평균을 예측값으로 쓰는 KNN 회귀 모델이다.
from sklearn.neighbors import KNeighborsRegressor
# [외부 라이브러리] scikit-learn의 model_selection 모듈:
#   - KFold          : 데이터를 k등분해 "한 조각은 검증, 나머지는 학습"을 k번 돌리는 분할 규칙을 만든다.
#   - cross_val_score: 주어진 분할마다 모델을 학습·채점해 fold별 점수 배열을 돌려준다.
from sklearn.model_selection import KFold, cross_val_score

# -----------------------------------------------------------------------------
# 공통 fold 고정
# -----------------------------------------------------------------------------
# folds: fold마다 (학습 행 번호 배열, 검증 행 번호 배열) 한 쌍이다. 각 fold는 학습 48행, 검증 12행이다.
#   - n_splits=5: train 60행을 12행씩 5조각으로 나눈다.
#   - shuffle=True: 나누기 전에 행을 섞는다. 파일 순서에 특정 경향이 있어도 한 fold에 몰리지 않게 한다.
#   - random_state=42: 몇 번을 실행해도 같은 fold가 나오도록 고정한다.
folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))

# -----------------------------------------------------------------------------
# 두 모델 교차검증
# -----------------------------------------------------------------------------
# rows: (모델 이름, fold 번호, RMSE) 기록을 한 줄씩 모을 빈 목록이다.
rows = []

# 후보 모델 2개를 하나씩 꺼내 같은 과정을 반복한다.
for name, estimator in [("Linear", LinearRegression()), ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    # model (type: Pipeline): [전처리 → 모델] 두 단계를 묶은 전체 모델이다.
    #   - clone(pre): 모델마다 새 전처리기를 준다. 전처리와 모델을 함께 묶어야 cross_val_score가 fold마다 학습 부분으로만 중앙값·평균·범주 목록을 배운다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # scores: fold 5개 각각의 "음수 RMSE"이다.
    #   - cv=folds: 위에서 고정한 fold 목록을 그대로 넘긴다. cv=5처럼 숫자를 주지 않는 이유는, 두 모델이 확실히 같은 조각으로 시험을 보게 하기 위해서이다.
    #   - scoring="neg_root_mean_squared_error": RMSE에 마이너스를 붙인 값으로 채점한다.
    #     scikit-learn은 "점수는 클수록 좋다"는 한 가지 규칙으로 모든 지표를 다룬다. RMSE는 작을수록 좋으므로 음수로 뒤집어 이 규칙에 맞춘다.
    #   - n_jobs=1: CPU 하나로 fold를 차례대로 처리한다. 데이터가 작아 병렬 처리의 이득이 없고, 실행 환경에 따른 오류나 출력 뒤섞임을 피한다.
    scores = cross_val_score(model, X_train, train["elapsed_ms"], cv=folds,
                             scoring="neg_root_mean_squared_error", n_jobs=1)
    # rmse: 부호를 되돌린 fold별 RMSE(ms)이다. 이제 "작을수록 좋은" 원래 의미의 값이 된다.
    #   - 부호를 되돌려야 사람이 "평균 약 600ms 틀린다"처럼 단위 그대로 읽고 보고할 수 있다.
    rmse = -scores
    # fold 번호와 RMSE를 짝지어 한 줄씩 rows에 쌓는다.
    #   - number: fold 번호이다. enumerate(..., start=1)로 0이 아니라 1부터 센다. 사람이 읽기 쉽게 하기 위해서이다.
    #   - value: 그 fold의 RMSE(ms)이다.
    for number, value in enumerate(rmse, start=1):
        rows.append({"model": name, "fold": number, "RMSE_ms": value})

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# result: 모델 2개 × fold 5개의 RMSE 기록 표이다. 열은 model, fold, RMSE_ms이다.
result = pd.DataFrame(rows)
# 표를 outputs/regression_cv.csv로 저장한다.
#   - index=False: 0, 1, 2… 행 번호는 의미 없는 값이므로 파일에 쓰지 않는다.
#   - encoding="utf-8-sig": 맨 앞에 BOM 표시를 붙인 UTF-8로 저장한다. 엑셀에서 열어도 한글이 깨지지 않는다.
result.to_csv(OUT / "regression_cv.csv", index=False, encoding="utf-8-sig")
# fold별 결과를 소수 둘째 자리까지 반올림해 행 번호 없이 출력한다.
print(result.round(2).to_string(index=False))
print("평균 RMSE(ms)")
# 모델별로 묶어 RMSE 평균을 구해 출력한다.
#   - groupby("model", sort=False): 이름을 알파벳순으로 다시 정렬하지 않고, 처음 나온 순서(Linear → KNN15)를 유지한다.
print(result.groupby("model", sort=False)["RMSE_ms"].mean().round(2).to_string())
print("회귀 교차검증의 정답은 elapsed_ms입니다. 입력에 elapsed_ms를 포함하지 않습니다.")
