"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
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
# test 20건의 입력·정답은 아래 모델에 전달하지 않는다.

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
# 트리는 이 실습에서 표준화하지 않는다.

# -----------------------------------------------------------------------------
# 비교 후보 모델 정의
# -----------------------------------------------------------------------------
# 단일 규칙 기반의 decision tree 분류기를 생성하기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 여러 개의 결정 트리를 앙상블하여 예측하는 랜덤 포레스트 분류기를 생성하기 위해 scikit-learn에서 RandomForestClassifier를 가져온다.
from sklearn.ensemble import RandomForestClassifier
# f1 score로 모델의 성능을 평가하고 혼동행렬을 체크하기 위해 scikit-learn에서 f1_score, confusion_matrix를 가져온다.
from sklearn.metrics import f1_score, confusion_matrix

# 트리는 깊이 3까지만, 잎마다 최소 3건을 두어 train을 외우지 못하게 제한한다. random_state=42로 고정해 결과를 재현한다.
# 숲은 트리 100그루를 만든다. 각 트리는 깊이 4, 잎 최소 2건으로 제한한다.
# max_features="sqrt"는 분할마다 전체 특성 수의 제곱근만큼만 무작위로 후보에 올려 트리끼리 서로 다르게 만든다. 
# random_state=42는 부트스트랩 표본과 특성 선택을 고정한다. n_jobs=1은 CPU 1개로 실행해 환경 차이 없이 같은 결과를 낸다.
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))]

# -----------------------------------------------------------------------------
# 학습 · 예측 · 평가
# -----------------------------------------------------------------------------
# 모델별 결과 저장을 위한 빈 리스트를 만든다.
rows = []
for name, estimator in candidates:
    # 각 모델 후보마다 독립적인 전처리 상태를 보장하기 위해 pre를 복제하여 Pipeline을 구성한다.
    # clone을 사용하여, 앞 후보가 배운 통계가 다음 후보로 넘어가지 않는다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 학습
    model.fit(X_train, y_train)
    # 예측
    prediction = model.predict(X_valid)
    # 평가
    # 혼동행렬을 통해 오차 종류(FP, FN)를 분리하여 집계한다.
    # labels=[0, 1]은 한쪽 라벨이 아예 예측되지 않아도 항상 2×2 모양을 보장한다.
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    # 모델별 학습 세트 점수, 검증 세트 점수 및 오류 건수를 기록한다.
    # train_F1은 학습한 데이터를 다시 맞힌 점수이다. valid_F1과의 차이로 과적합 정도를 본다.
    # valid_F1은 처음 보는 데이터 점수로 후보 선택의 기준이다. 
    # zero_division=0은 실패를 하나도 예측하지 않아 분모가 0일 때 경고 대신 0을 쓴다. FP는 성공을 실패로 잘못 경보한 수, FN은 실제 실패를 놓친 수이다.
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------    
result = pd.DataFrame(rows)
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")
# 소수 셋째 자리로 반올림해 행번호 없이 화면에 출력한다.
print(result.round(3).to_string(index=False))
