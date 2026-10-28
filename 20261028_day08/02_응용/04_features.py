"""실습 04 응용: 행별 특성 두 개만 추가하고 같은 Logistic·5-fold로 비교합니다."""
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
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# -----------------------------------------------------------------------------
# 데이터 분할
# -----------------------------------------------------------------------------
# train에 해당하는 request_id를 읽어 position순으로 정렬해 train_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
# valid에 해당하는 request_id를 읽어 position순으로 정렬해 valid_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
# raw에서 train_ids와 valid_ids에 해당하는 행만 꺼내 각각 train, valid에 저장한다.
# train, valid (type: DataFrame): 각각 60행, 20행이다. copy()로 원본 raw와 분리된 복사본을 만들어, 이후 수정이 raw에 영향을 주지 않게 한다.
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
# 모델 입력에 쓸 열들을 FEATURES에 리스트로 저장한다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# 수치형 2열과 범주형 2열로 FEATURES를 분리한다.
NUM, CAT = FEATURES[:2], FEATURES[2:]
# train과 valid에서 입력 열만 골라 X_train, X_valid에 저장한다.
# X_train (type: DataFrame, 60행×4열), X_valid (type: DataFrame, 20행×4열): 모델에 들어갈 입력 표이다.
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
# train과 valid에서 정답 열 failed를 꺼내 y_train, y_valid에 저장한다.
# y_train, y_valid (type: Series[int]): 1은 실패, 0은 성공이며, 실패(1)가 찾고 싶은 양성 클래스이다.
y_train, y_valid = train["failed"], valid["failed"]

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
# 모델·평가 도구 불러오기
# -----------------------------------------------------------------------------
# 실패 여부를 예측하는 분류 모델을 만들기 위해 외부 라이브러리 scikit-learn에서 LogisticRegression을 가져온다.
from sklearn.linear_model import LogisticRegression
# fold를 나누고 교차검증 점수를 구하기 위해 외부 라이브러리 scikit-learn에서 StratifiedKFold와 cross_val_score를 가져온다.
# 기능: StratifiedKFold는 실패/성공 비율을 fold마다 비슷하게 유지하며 나누고, cross_val_score는 fold마다 학습·평가를 반복해 지표 하나의 점수 배열을 돌려준다.
from sklearn.model_selection import StratifiedKFold, cross_val_score
# valid 예측을 채점하기 위해 외부 라이브러리 scikit-learn에서 f1_score를 가져온다.
from sklearn.metrics import f1_score
# 전처리기를 후보마다 새로 복제하기 위해 외부 라이브러리 scikit-learn에서 clone을 가져온다.
# 기능: 설정(하이퍼파라미터)은 같고 학습된 값은 없는 새 객체를 만든다. 원본 pre를 직접 고치지 않아, 한 후보의 변경이 다른 후보에 번지지 않는다.
from sklearn.base import clone

# -----------------------------------------------------------------------------
# 행별 feature 추가
# -----------------------------------------------------------------------------
# X_train과 X_valid는 이미 위에서 train·valid로부터 복사되어 있으므로, Pipeline이 실제로 받는 X_train·X_valid에 열을 더해야 아래 Logistic_FE가 그 열을 찾을 수 있다.
# frame (type: DataFrame): 반복마다 X_train, X_valid를 차례로 가리킨다. 같은 객체를 가리키므로 frame에 열을 더하면 X_train, X_valid에 그대로 더해진다.
for frame in (X_train, X_valid):
    # log_input_chars (type: Series[float]): 글자 수에 1을 더하고 자연로그를 취한 값이다(교안 E08과 같은 정의).
    # 로그를 쓰는 이유: 글자 수는 큰 값 쪽으로 길게 늘어진 분포라서, 로그로 눌러 주면 아주 긴 요청 몇 건이 계수를 좌우하지 못한다.
    # log1p(=log(1+x))를 쓰는 이유: 글자 수가 0이어도 log(0)=-무한대가 아니라 0이 나온다. NaN은 NaN으로 남아 아래 SimpleImputer가 채운다.
    frame["log_input_chars"] = np.log1p(frame["input_chars"])
    # chars_per_slot (type: Series[float]): 허용 도구 한 칸(slot)당 글자 수로, 글자 수 ÷ (1 + 허용 도구 수)이다(교안 E08과 같은 정의).
    # 분모의 1.0: 허용 도구가 0개인 요청에서도 0으로 나누는 일이 생기지 않게 하는 설계 선택이며, 결과를 실수(float)로 만든다.
    # 두 값 모두 같은 행의 사전 입력만 쓰므로, 다른 행이나 정답(failed)의 정보가 섞이지 않는다. 그래서 fit 없이 미리 계산해도 누수가 아니다.
    frame["chars_per_slot"] = frame["input_chars"] / (1.0 + frame["allowed_tools"])

# -----------------------------------------------------------------------------
# 두 후보 비교: 기존 특성 vs 파생 특성 추가 / 동일 조건(같은 fold, 같은 모델)
# -----------------------------------------------------------------------------
# folds: train 60행을 5조각으로 나누는 규칙이다. 각 fold는 48행으로 학습하고 12행으로 평가한다.
# shuffle=True: 행 순서대로 자르지 않고 섞어서 나눈다. random_state=42: 섞는 방식을 고정해 실행할 때마다 같은 fold가 나오게 한다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# choices (type: dict[str, list[str]]): 후보 이름 → 수치 경로에 넣을 열 목록이다.
# "Logistic"은 원본 숫자 열 2개만, "Logistic_FE"는 여기에 파생 열 2개를 더한 4개를 쓴다. 범주 열은 두 후보가 같다.
choices = {"Logistic": NUM, "Logistic_FE": NUM + ["log_input_chars", "chars_per_slot"]}

# 후보마다 결과 한 줄을 쌓아 저장할 빈 리스트이다.
rows = []
# 후보 이름과 수치 열 목록을 하나씩 꺼내 같은 작업을 반복한다.
for name, numeric_columns in choices.items():
    # 기존 전처리 pre를 복제해 새 current_pre에 넣는다. pre를 직접 다루지 않는다.
    current_pre = clone(pre)
    # transformers[0]은 ("num", 숫자 처리 Pipeline, 열 목록) 튜플이다. 처리 방법은 그대로 두고 열 목록만 numeric_columns로 바꿔 끼운다.
    # 그래서 파생 열도 중앙값 대체(NaN 처리)와 표준화를 fold 학습 부분 기준으로 받는다.
    current_pre.transformers[0] = ("num", current_pre.transformers[0][1], numeric_columns)
    # model (type: Pipeline): 전처리기 current_pre와 로지스틱 회귀를 묶은 전체 모델이다.
    # C=1: 규제 세기의 역수로 기본값이다. max_iter=1000: 최적화가 수렴할 때까지 충분히 반복하도록 상한을 넉넉히 둔다. 두 후보가 같은 설정을 쓴다.
    model = Pipeline([("pre", current_pre), ("model", LogisticRegression(C=1, max_iter=1000))])
    # scores (type: numpy.ndarray, 길이 5): fold별 F1 점수이다.
    # model 전체를 넘기므로 fold마다 학습용 48행으로만 중앙값·평균·범주 목록을 정한다. cv=folds로 두 후보가 같은 fold를 쓴다.
    # scoring="f1": 실패(1)를 양성으로 본 F1로 채점한다. n_jobs=1: 병렬 없이 한 프로세스에서 차례로 실행한다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=1)
    # train 60행 전체로 모델을 다시 학습한다. valid 진단 점수를 내기 위해서이다.
    model.fit(X_train, y_train)
    # 후보 하나의 결과를 dict로 만들어 rows에 덧붙인다.
    # CV_F1: 5개 fold F1의 평균으로, 후보 선택의 기준이다.
    # CV_std: 5개 fold F1의 표준편차(ddof=0)로, fold에 따라 점수가 얼마나 흔들리는지 보여 준다.
    # valid_F1: train 전체로 학습한 모델을 valid 20행에 적용한 F1이다. 
    # zero_division=0은 실패를 하나도 예측하지 않아 Precision 분모가 0일 때 경고 대신 0점으로 처리한다.
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0)})

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame, 2행×4열): 두 후보의 model, CV_F1, CV_std, valid_F1을 한 표로 모은다.
result = pd.DataFrame(rows)
# 표를 outputs/feature_comparison.csv로 저장한다.
# index=False: 0, 1 행 번호는 저장하지 않는다. encoding="utf-8-sig": 엑셀에서 열어도 한글이 깨지지 않게 한다.
result.to_csv(OUT / "feature_comparison.csv", index=False, encoding="utf-8-sig")
# 표를 소수 넷째 자리로 반올림해 행 번호 없이 출력한다.
print(result.round(4).to_string(index=False))
print("CV는 선택용, valid는 진단용입니다. test는 아직 보지 않습니다.")
