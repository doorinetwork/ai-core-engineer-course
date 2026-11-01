"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
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
# 후보 모델과 fold 준비
# -----------------------------------------------------------------------------
# 단일 규칙 기반의 decision tree 분류기를 생성하기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 여러 개의 결정 트리를 앙상블하여 예측하는 랜덤 포레스트 분류기를 생성하기 위해 scikit-learn에서 RandomForestClassifier를 가져온다.
from sklearn.ensemble import RandomForestClassifier
# 모델의 성능을 비교하기 위한 최빈값 기준선(baseline)으로 DummyClassifier 클래스를 가져온다.
from sklearn.dummy import DummyClassifier
# 부스팅 분류 후보를 만들기 위해 외부 라이브러리 xgboost에서 XGBClassifier를 가져온다. 
# scikit-learn처럼 fit·predict·predict_proba를 제공해 Pipeline에 그대로 넣을 수 있다.
from xgboost import XGBClassifier
# 교차검증을 수행하여 각 fold의 평가지표 점수 배열을 얻기 위해 cross_val_score 함수를 가져온다.
from sklearn.model_selection import StratifiedKFold, cross_val_score
# 외부 라이브러리 scikit-learn에서 f1_score와 confusion_matrix를 가져온다. 
# f1_score는 실패(1) 클래스의 F1을, confusion_matrix는 TN·FP·FN·TP 개수를 계산한다.
from sklearn.metrics import f1_score, confusion_matrix

# train 60건을 5개 fold로 나눈 (학습 행 번호, 평가 행 번호) 쌍 5개를 folds에 저장한다. (type: list, 원소는 (ndarray, ndarray) 튜플 5개)
# n_splits=5는 fold마다 48건으로 학습하고 12건으로 평가한다. 
# shuffle=True는 나누기 전에 행을 섞는다. random_state=42는 섞는 순서를 고정해 매번 같은 fold가 나오게 한다.
# split에 y_train을 넘겨 각 fold의 실패(1) 비율을 train 전체 비율과 같게 맞춘다. list로 미리 고정하므로 네 후보가 정확히 같은 fold로 비교된다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# 비교할 네 후보의 (이름, 학습 전 모델) 쌍을 candidates에 저장한다. (type: list, 원소는 (str, 분류기) 튜플 4개)
# Dummy: strategy="most_frequent"는 train에서 가장 많은 클래스(성공=0)만 예측한다. 실패를 하나도 맞히지 못하므로 F1이 0이고, 다른 후보가 이보다 나아야 의미가 있다.
# Tree: max_depth=3은 질문을 최대 3번까지만 해서 과적합을 막는다. min_samples_leaf=3은 잎 하나에 최소 3건을 남긴다. random_state=42는 동점 분할 선택을 고정한다.
# Forest: n_estimators=100은 나무 100그루를 만든다. 
#   max_depth=4·min_samples_leaf=2는 나무 하나의 복잡도를 제한한다.
#   max_features="sqrt"는 분할마다 특성 수의 제곱근만큼만 후보로 골라 나무끼리 덜 닮게 한다. 
#   random_state=42는 무작위성을 고정한다. n_jobs=1은 CPU 1개만 써서 결과를 재현한다.
# XGB: n_estimators=80은 보정 트리 80회, max_depth=2는 보정 트리 하나의 질문을 최대 2번, learning_rate=0.05는 한 회차의 보정을 5%만 반영한다.
#   subsample=0.8은 회차마다 학습 행의 80%만 쓴다. colsample_bytree=1.0은 특성을 전부 쓴다. reg_lambda=1.0은 잎 가중치에 L2 규제를 건다.
#   objective="binary:logistic"은 실패 확률을 학습 목표로 삼는다. 
#   eval_metric="logloss"는 진단용 손실이며 비교 지표인 F1과 다르다.
#   tree_method="hist"는 히스토그램으로 분할을 빠르게 찾는다. 
#   device="cpu"는 CPU로 학습한다. random_state=42·n_jobs=1은 결과를 재현한다.
candidates = [("Dummy", DummyClassifier(strategy="most_frequent")),
              ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]

# -----------------------------------------------------------------------------
# 교차검증 · 학습 · valid 진단
# -----------------------------------------------------------------------------
# 후보별 요약 행을 모을 rows와 fold별 점수 행을 모을 cv_rows를 빈 리스트로 만든다. 
rows, cv_rows = [], []
# valid 20건의 request_id와 실제 정답을 담은 예측 기록표를 predictions에 만든다.
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})
# 네 후보를 하나씩 꺼내 같은 절차로 평가한다.
for name, estimator in candidates:
    # 새로 복제한 전처리기와 후보 분류기를 하나의 Pipeline으로 묶어 model에 저장한다. clone(pre)를 쓰므로 다른 후보에서 학습된 통계가 섞이지 않는다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # model 전체를 고정한 5개 fold로 교차검증해 fold별 F1을 scores에 저장한다. (type: ndarray, 길이 5)
    # cross_val_score는 fold마다 model을 복제해 학습 48건으로 전처리기와 분류기를 함께 fit하고, 평가 12건은 변환·예측만 한다. 
    # 그래서 평가 12건의 값이 중앙값·최빈값 계산에 섞이지 않는다.
    # cv=folds는 네 후보에 똑같은 fold를 쓴다. scoring="f1"은 실패(1)를 양성으로 본 F1을 점수로 쓴다. 
    # Dummy는 실패를 예측하지 않으므로 F1이 0이 되고 경고가 나올 수 있다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    # 학습
    model.fit(X_train, y_train)
    # 예측
    prediction = model.predict(X_valid)
    # valid 실제값과 예측값으로 혼동행렬을 만들고 ravel()로 펼쳐 tn, fp, fn, tp에 저장한다.
    # labels=[0, 1]은 순서를 고정한다. Dummy처럼 1을 한 번도 예측하지 않아도 2×2 행렬이 만들어져 값 4개가 항상 나온다.
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    # CV_F1은 5개 fold F1의 평균, CV_std는 표준편차(ddof=0)이다. train_F1은 학습한 자료를 다시 맞힌 점수라서 과적합 진단에 쓴다.
    # valid_F1·FP(성공을 실패로 잘못 경고)·FN(실패를 놓침)은 valid 진단 값이다. zero_division=0은 실패 예측이 하나도 없을 때 오류 대신 0을 쓴다.
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
    # fold 번호를 1부터 붙여 점수를 하나씩 꺼낸다. number는 fold 번호, value는 그 fold의 F1이다.
    for number, value in enumerate(scores, start=1):
        # fold 한 행을 cv_rows에 추가한다.
        cv_rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
    # 이 후보의 valid 예측을 이름 열로 predictions에 추가한다. 
    predictions[name] = prediction
    # Forest일 때만 특성 중요도 표를 만든다.
    if name == "Forest":
        # 전처리 뒤 실제 열 이름과 MDI(불순도 감소 기반) 중요도를 같은 순서로 짝지어 importance에 저장한다. (type: DataFrame, 8행)
        # get_feature_names_out()은 One-hot으로 늘어난 열 이름을 모델 입력 순서대로 돌려주므로, 중요도 값과 이름이 어긋나지 않는다. 
        # 중요도는 이 모델의 의존도이지 실패의 원인이 아니다.
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
# 후보별 요약 행 4개를 비교표 comparison으로 만든다.
comparison = pd.DataFrame(rows)
# 비교표를 outputs/classification_comparison.csv로 저장한다. index=False는 행번호를 빼고, utf-8-sig는 엑셀에서 한글이 깨지지 않게 한다.
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# fold 점수 20행(후보 4개 × fold 5개)을 outputs/cv_scores.csv로 저장한다.
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
# valid 20건의 실제값과 네 후보의 예측값을 outputs/validation_predictions.csv로 저장한다.
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
# Forest 중요도 8행을 outputs/feature_importance.csv로 저장한다.
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
# 비교표를 소수 넷째 자리로 반올림해 행번호 없이 출력한다.
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
# 저장한 파일 4개의 이름을 출력한다.
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
