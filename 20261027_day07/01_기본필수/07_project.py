"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
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
# valid에서 input_chars 열을 X_valid으로 저장한다.
X_valid = valid[["input_chars"]]
# train에서 failed 열을 y_train으로 저장한다.
y_train = train["failed"]
# valid에서 failed 열을 y_valid으로 저장한다.
y_valid = valid["failed"]

# -----------------------------------------------------------------------------
# 전처리기
# -----------------------------------------------------------------------------
# 단일 규칙 기반의 decision tree 분류기를 생성하기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 여러 개의 결정 트리를 앙상블하여 예측하는 랜덤 포레스트 분류기를 생성하기 위해 scikit-learn에서 RandomForestClassifier를 가져온다.
from sklearn.ensemble import RandomForestClassifier
# 교차검증을 수행하여 각 fold의 평가지표 점수 배열을 얻기 위해 cross_val_score 함수를 가져온다.
from sklearn.model_selection import StratifiedKFold, cross_val_score
# 각 열마다 다른 전처리를 적용하기 위해 외부 라이브러리 scikit-learn에서 ColumnTransformer를 가져온다.
from sklearn.compose import ColumnTransformer
# 범주형 데이터의 One-hot 인코딩을 위해 외부 라이브러리 scikit-learn에서 OneHotEncoder를 가져온다.
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
# 파이프라인 내부의 전처리기 상태를 초기화된 새 객체로 복제하기 위해 외부 라이브러리 scikit-learn에서 clone을 가져온다.
from sklearn.base import clone
# f1 score로 모델의 성능을 평가하기 위해 scikit-learn에서 f1_score를 가져온다.
from sklearn.metrics import f1_score

# 모델 입력에 쓸 열들을 features에 리스트로 저장한다.
features = ["input_chars", "task_type"]
# 각 train, valid 데이터에서 입력에 쓸 열들만 추려서 X_train, X_valid로 저장한다.
X_train, X_valid = train[features], valid[features]
# 각 열마다 다른 전처리 기법을 사용해 pre에 저장한다. (type: ColumnTransformer) 
#   각 항목은 (이름, 변환 도구, 적용할 열) 순서이다.
#   "num": input_chars를 "passthrough"로 값을 그대로 통과시킨다. 트리 계열 모델은 "기준값보다 큰가/작은가"만 묻기 때문에 표준화가 결과를 바꾸지 않는다.
#   "cat": task_type을 OneHotEncoder로 종류마다 0/1 열을 하나씩 만든다.
#   handle_unknown="ignore": train에 없던 새 종류가 valid에 나오면 오류 대신 모든 0/1 열을 0으로 둔다.
#   sparse_output=False: 결과를 일반 2차원 배열로 돌려준다(희소행렬이 아님).
pre = ColumnTransformer([
    ("num", "passthrough", ["input_chars"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])

# -----------------------------------------------------------------------------
# 후보 모델과 fold 준비
# -----------------------------------------------------------------------------
# train을 3조각(fold)으로 나누는 규칙을 folds에 저장한다. (type: StratifiedKFold)
#   n_splits=3: 3번 반복하며 매번 2조각으로 학습하고 남은 1조각으로 평가한다.
#   Stratified: 각 fold의 실패/성공 비율을 train 전체와 같게 맞춘다.
#   shuffle=True, random_state=42: 나누기 전에 행을 섞되, 섞는 순서를 고정해 재실행해도 같은 fold가 나오게 한다.
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# 비교할 후보 모델 2개를 (이름, 모델 객체) 튜플로 묶어 candidates에 저장한다. (type: list)
# Tree: max_depth=3으로 질문을 최대 3단계까지만 하게 해 train을 외우는 과적합을 줄인다.
# Forest: n_estimators=20으로 트리 20개를 만들고, 각 트리도 max_depth=3으로 제한한다.
#   random_state=42: 트리의 무작위 선택을 고정해 재실행해도 같은 모델이 나오게 한다.
#   n_jobs=1: CPU 코어 1개만 써서 실행 환경과 상관없이 같은 방식으로 계산하게 한다.
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]
# 후보별 요약 행을 모을 rows와 fold별 점수 행을 모을 cv_rows를 빈 리스트로 만든다. 
rows, cv_rows = [], []
# valid의 요청 ID와 실제 정답만 복사해 predictions에 저장한다. 아래 반복에서 모델별 예측 열을 옆에 붙인다.
# .copy()는 원본 valid가 바뀌지 않도록 별도 사본을 만든다.
predictions = valid[["request_id", "failed"]].copy()
# 후보를 하나씩 꺼내 같은 절차로 평가한다.
for name, estimator in candidates:
    # 새로 복제한 전처리기와 후보 분류기를 하나의 Pipeline으로 묶어 model에 저장한다. clone(pre)를 쓰므로 다른 후보에서 학습된 통계가 섞이지 않는다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # train 안에서 3-fold 교차검증을 수행해 fold별 F1 점수 3개를 scores에 저장한다.
    # scoring="f1": 실패(1)를 양성으로 보고 F1을 계산한다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    # 학습
    model.fit(X_train, y_train)
    # 예측
    prediction = model.predict(X_valid)
    # 이 후보의 결과를 딕셔너리 1개로 만들어 rows에 추가한다.
    # CV_F1: fold 점수 3개의 평균(type: float), CV_std: 그 표준편차로 fold마다 점수가 얼마나 흔들리는지 보여준다.
    # valid_F1: valid 실제 정답과 예측으로 계산한 F1이다.
    # zero_division=0: 모델이 실패(1)를 한 번도 예측하지 않아 Precision 분모가 0이 되면 경고 대신 0으로 계산한다.
    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
    # fold 번호를 붙여(1부터 시작) fold별 점수를 cv_rows에 한 줄씩 append한다.
    # start=1: fold 번호가 0이 아닌 1부터 시작되도록 한다.
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "F1": value})
    # valid 예측값을 모델 이름의 열로 predictions에 붙인다. 
    predictions[name] = prediction
    # Forest 차례일 때만 특성 중요도 표를 만든다.
    if name == "Forest":
        # 전처리 후 열 이름과 Forest의 MDI 중요도를 나란히 놓은 표를 importance에 저장한다.
        # get_feature_names_out(): "num__input_chars", "cat__task_type_qa"처럼 변환 후 열 이름을 돌려준다.
        # feature_importances_: 각 열이 트리 분기에서 불순도를 줄인 정도의 합으로, 모두 더하면 1이다.
        # 이 값은 "이 모델이 이 데이터에서 어느 열에 기대었는가"이고, 그 열이 실패의 원인이라는 뜻은 아니다.
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# -----------------------------------------------------------------------------
# 결과 저장
# -----------------------------------------------------------------------------
# 후보별 요약을 비교표 comparison으로 만든다.
comparison = pd.DataFrame(rows)
# 비교표를 outputs/classification_comparison.csv로 저장한다. index=False는 행번호를 빼고, utf-8-sig는 엑셀에서 한글이 깨지지 않게 한다.
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# fold 점수를 outputs/cv_scores.csv로 저장한다.
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
# valid의 실제값과 각 후보의 예측값을 outputs/validation_predictions.csv로 저장한다.
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
# Forest 중요도를 outputs/feature_importance.csv로 저장한다.
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# train, valid, test의 행 개수를 차례로 출력해 60/20/20 분할이 맞는지 확인한다.
print("분할:", len(train), len(valid), len(test))
# 비교표를 소수 셋째 자리까지 반올림해 행 번호 없이 출력한다.
print(comparison.round(3).to_string(index=False))
print("중요도")
# 중요도 표를 소수 셋째 자리까지 반올림해 행 번호 없이 출력한다.
print(importance.round(3).to_string(index=False))
# 저장한 파일 4개의 이름을 출력한다.
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
