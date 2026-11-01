"""실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
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
# 단일 규칙 기반의 decision tree 분류기를 생성하기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 여러 개의 결정 트리를 앙상블하여 예측하는 랜덤 포레스트 분류기를 생성하기 위해 scikit-learn에서 RandomForestClassifier를 가져온다.
from sklearn.ensemble import RandomForestClassifier
# 부스팅 분류 후보를 만들기 위해 외부 라이브러리 xgboost에서 XGBClassifier를 가져온다. 
# scikit-learn처럼 fit·predict·predict_proba를 제공해 Pipeline에 그대로 넣을 수 있다.
from xgboost import XGBClassifier
# 교차검증을 수행하여 각 fold의 평가지표 점수 배열을 얻기 위해 cross_val_score 함수를 가져온다.
from sklearn.model_selection import StratifiedKFold, cross_val_score

# train 내부에서 클래스 비율을 보존하는 5-fold 분할 목록을 사전에 고정한다.
# (학습 행 번호, 평가 행 번호) 쌍 5개 (type: list, 원소는 (ndarray, ndarray) 튜플 5개)
# n_splits=5는 5등분해 fold마다 48건으로 학습하고 12건으로 평가한다. 
# shuffle=True는 나누기 전에 행을 섞어 파일 순서의 치우침을 없앤다. 
# random_state=42는 섞는 순서를 고정해 매번 같은 fold가 나오게 한다.
# split에 y_train을 넘기는 이유는 각 fold의 실패(1) 비율을 train 전체 비율과 같게 맞추기 위해서이다. list로 미리 고정하면 세 후보가 정확히 같은 fold로 비교된다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# 사전 고정된 하이퍼파라미터 조건을 가진 트리, 포레스트, XGBoost 분류기 후보 목록을 정의한다.
# 비교할 세 후보의 (이름, 학습 전 모델) 쌍을 candidates에 저장한다. (type: list, 원소는 (str, 분류기) 튜플 3개)
# Tree: max_depth=3은 질문을 최대 3번까지만 해서 과적합을 막는다. 
#   min_samples_leaf=3은 잎 하나에 최소 3건을 남겨 몇 건짜리 규칙을 만들지 않게 한다. 
#   random_state=42는 동점 분할 선택을 고정한다.
# Forest: n_estimators=100은 나무 100그루를 만든다. 
#   max_depth=4·min_samples_leaf=2는 나무 하나의 복잡도를 제한한다.
#   max_features="sqrt"는 분할마다 특성 수의 제곱근만큼만 후보로 골라 나무끼리 덜 닮게 한다. 
#   random_state=42는 부트스트랩·특성 선택을 고정한다. n_jobs=1은 CPU 1개만 써서 결과를 재현한다.
# XGB: n_estimators=80은 보정 트리 80회, max_depth=2는 보정 트리 하나의 질문을 최대 2번, learning_rate=0.05는 한 회차의 보정을 5%만 반영한다.
#   subsample=0.8은 회차마다 학습 행의 80%만 쓴다. colsample_bytree=1.0은 특성을 전부 쓴다. reg_lambda=1.0은 잎 가중치에 L2 규제를 건다.
#   objective="binary:logistic"은 실패 확률을 학습 목표로 삼는다. 
#   eval_metric="logloss"는 진단용 손실이며 비교 지표인 F1과 다르다.
#   tree_method="hist"는 히스토그램으로 분할을 빠르게 찾는다. 
#   device="cpu"는 CPU로 학습한다. random_state=42·n_jobs=1은 결과를 재현한다.
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]

# fold별 평가 결과를 수집할 빈 리스트를 생성한다
rows = []
# 후보를 하나씩 꺼내 동일한 5-fold 교차검증을 수행한다.
for name, estimator in candidates:
    # 전처리 상태를 독립적으로 초기화한 Pipeline을 구성하여 데이터 누수를 차단한다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # cross_val_score: 각 fold의 학습 48행에서 전처리·모델을 학습하고 평가 12행에서 F1 점수를 측정한다.
    # cross_val_score는 fold마다 model을 복제해 학습 48건으로 전처리기와 분류기를 함께 fit하고, 평가 12건은 변환·예측만 한다. 
    # 그래서 평가 12건의 값이 중앙값·최빈값 계산에 섞이지 않는다.
    # 전처리만 train 60건으로 미리 fit하고 모델만 교차검증하면 평가 fold의 정보가 전처리 통계에 새어 들어가므로, Pipeline 전체를 넘겨야 한다.
    # cv=folds는 위에서 고정한 같은 fold 5개를 세 후보에 똑같이 쓴다. scoring="f1"은 실패(1)를 양성으로 본 F1을 점수로 쓴다. 점수 순서는 folds 순서와 같다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    # fold별 점수, 평균 F1, 표준편차를 소수점 넷째 자리로 반올림하여 출력한다.
    print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))
    # 결과 CSV 저장을 위해 fold 번호, 학습 샘플 수(48), 평가 샘플 수(12), F1 점수를 기록한다.
    for number, value in enumerate(scores, start=1):
        rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
# 수집된 fold별 평가 기록을 06_cv_scores.csv 파일로 저장한다.
pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
# 검증 및 테스트 세트 누수 방지 원칙을 안내한다
print("사용 자료: train60 내부만 / valid20·test20 미사용")
