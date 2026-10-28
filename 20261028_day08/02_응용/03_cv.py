"""실습 03 응용: Pipeline을 통째로 train 내부 5-fold에 넣어 다중 지표를 계산합니다."""
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
# 모델과 교차검증 준비
# -----------------------------------------------------------------------------
# 실패 확률을 예측하는 분류 모델을 만들기 위해 외부 라이브러리 scikit-learn에서 LogisticRegression을 가져온다.
# 기능: 입력의 가중합을 0~1 확률로 바꿔 클래스를 예측하고, predict_proba로 확률도 내준다(AUC 계산에 필요하다).
from sklearn.linear_model import LogisticRegression
# fold를 나누고 교차검증을 실행하기 위해 외부 라이브러리 scikit-learn에서 StratifiedKFold와 cross_validate를 가져온다.
# 기능: StratifiedKFold는 실패/성공 비율을 fold마다 비슷하게 유지하며 나누고, 
#      cross_validate는 fold마다 학습·평가를 반복해 여러 지표를 한 번에 계산한다.
from sklearn.model_selection import StratifiedKFold, cross_validate
# F1 채점 규칙을 직접 만들기 위해 외부 라이브러리 scikit-learn에서 make_scorer와 f1_score를 가져온다.
# 기능: f1_score는 Precision과 Recall의 조화평균을 계산하고, 
#      make_scorer는 이 함수를 cross_validate가 쓸 수 있는 채점기(scorer)로 감싼다.
from sklearn.metrics import make_scorer, f1_score

# model: 전처리기 pre와 로지스틱 회귀를 차례로 묶은 전체 모델이다.
# C=1: 규제 세기의 역수로 기본값이다. max_iter=1000: 최적화가 수렴할 때까지 충분히 반복하도록 상한을 넉넉히 둔다.
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1, max_iter=1000))])
# folds: train 60행을 5조각으로 나누는 규칙이다. 각 fold는 48행으로 학습하고 12행으로 평가한다.
# shuffle=True: 행 순서대로 자르지 않고 섞어서 나눈다. random_state=42: 섞는 방식을 고정해 실행할 때마다 같은 fold가 나오게 한다.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores: cross_validate가 돌려주는 결과이다. "test_f1", "test_auc"(각 길이 5)와 fit_time, score_time이 들어 있다.
# model 전체를 넘기므로 fold마다 학습용 48행으로만 전처리 대표값을 정하고, 평가용 12행에는 그 값을 적용만 한다. 그래서 평가 행의 정보가 새지 않는다.
# X_train, y_train만 넣는다: valid와 보관 test는 최종 비교·최종 평가용으로 아껴 두어야 하므로 교차검증에 섞지 않는다.
# cv=folds: 위에서 고정한 5-fold 규칙을 그대로 쓴다.
# scoring은 dict로 준다. 키 이름 "f1", "auc" 앞에 "test_"가 붙어 결과 키가 "test_f1", "test_auc"가 된다.
#   - "f1": make_scorer(f1_score, zero_division=0)로 만든다. 
#           zero_division=0은 어떤 fold에서 실패를 하나도 예측하지 않아 Precision 분모가 0이 될 때 경고 대신 0점으로 처리한다는 뜻이다.
#   - "auc": "roc_auc" 문자열을 쓰면 scikit-learn이 predict_proba의 확률로 AUC를 계산한다. AUC는 임계값 0.5로 자른 예측이 아니라 확률의 순서를 채점한다.
scores = cross_validate(model, X_train, y_train, cv=folds,
    scoring={"f1": make_scorer(f1_score, zero_division=0), "auc": "roc_auc"},
)

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
result = pd.DataFrame({"fold": range(1, 6), "F1": scores["test_f1"], "ROC_AUC": scores["test_auc"]})
result.to_csv(OUT / "classification_cv.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("평균F1/표준편차:", round(result["F1"].mean(), 3), round(result["F1"].std(ddof=0), 3))
print("각 fold: 학습48·평가12. test_f1의 test는 CV 내부 평가이며 보관 test20이 아닙니다.")
