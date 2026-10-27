"""실습 02 응용: 기존 PPT의 5개 깊이 후보를 비교합니다."""
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
# 트리는 이 실습에서 표준화하지 않는다. 각 후보·fold마다 새 전처리기를 학습한다.

# -----------------------------------------------------------------------------
# 비교 후보 모델 정의
# -----------------------------------------------------------------------------
# 단일 규칙 기반의 decision tree 분류기를 생성하기 위해 외부 라이브러리 scikit-learn에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# f1 score로 모델의 성능을 평가하기 위해 scikit-learn에서 f1_score를 가져온다.
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# 깊이별 모델 생성 / 학습 / 평가
# -----------------------------------------------------------------------------
# 각 깊이별 실험 결과 딕셔너리를 저장할 빈 리스트를 생성한다.
rows = []
# 비교할 깊이 리스트를 정해두고 하나씩 꺼내 아래 학습 및 평가를 반복한다.
for depth in [1, 2, 3, 5, None]:
    # TODO A02: depth를 실제 트리 깊이로 적용하세요.
    estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=42)
    # 각 모델 후보마다 독립적인 전처리 상태를 보장하기 위해 pre를 복제하여 Pipeline을 구성한다.
    # clone을 사용하여, 앞 후보가 배운 통계가 다음 후보로 넘어가지 않는다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 학습
    model.fit(X_train, y_train)
    # 깊이 조건, train F1, valid F1 점수, 잎 개수를 딕셔너리로 묶어 결과 목록에 추가한다.
    # depth: str(depth)로 문자열로 바꿔 None도 "None"이라는 글자로 표에 남긴다. (type: str)
    # train_F1: 학습에 쓴 데이터에 대한 F1이다. valid_F1보다 크게 높으면 과적합 신호이다. (type: float)
    # valid_F1: 학습에 쓰지 않은 데이터에 대한 F1로, 깊이 후보를 고르는 기준이다. (type: float)
    # zero_division=0은 모델이 1을 하나도 예측하지 않아 정밀도를 계산할 수 없을 때 경고 대신 0으로 처리한다.
    # leaves: 파이프라인에서 "model" 단계(트리)를 꺼내 잎(최종 판정 칸)의 개수를 센다. 잎이 많을수록 복잡한 트리이다. (type: int)
    rows.append({"depth": str(depth), "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0),
                 "leaves": model["model"].get_n_leaves()})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------    
# rows의 딕셔너리 5개를 5행 4열(depth, train_F1, valid_F1, leaves) 표로 바꿔 result에 저장한다. (type: DataFrame)
result = pd.DataFrame(rows)
# 결과 표를 outputs/02_depth.csv로 저장한다.
# index=False는 0~4 행 번호를 파일에 쓰지 않게 하고, utf-8-sig는 Excel에서 열어도 글자가 깨지지 않게 한다.
result.to_csv(OUT / "02_depth.csv", index=False, encoding="utf-8-sig")
# 소수 셋째 자리까지 반올림한 표를 행 번호 없이 화면에 출력한다. 반올림은 출력용이며 CSV에는 원래 값이 저장된다.
print(result.round(3).to_string(index=False))
