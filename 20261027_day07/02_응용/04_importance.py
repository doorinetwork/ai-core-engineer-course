"""실습 04 응용: 전처리 출력 이름 8개와 중요도를 대응시킵니다."""
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
# 트리는 이 실습에서 표준화하지 않는다.

# -----------------------------------------------------------------------------
# 비교 후보 모델 정의
# -----------------------------------------------------------------------------
# 여러 개의 결정 트리를 앙상블하여 예측하는 랜덤 포레스트 분류기를 생성하기 위해 scikit-learn에서 RandomForestClassifier를 가져온다.
from sklearn.ensemble import RandomForestClassifier
# model: 전처리 단계 "pre"와 모델 단계 "model"을 묶은 파이프라인이다. (type: Pipeline)
#   clone(pre): 설계도 pre의 학습 전 복사본을 넣는다. 그래서 학습되는 것은 이 복사본이고, 원본 pre는 끝까지 학습되지 않는다.
#   n_estimators=100: 나무 100그루를 만들어 투표한다.
#   max_depth=4: 각 나무는 질문을 최대 4단계까지만 한다. train 60건에 지나치게 맞추는 것을 막는다.
#   min_samples_leaf=2: 마지막 잎에 최소 2건이 남아야 한다. 1건짜리 규칙을 만들지 않게 한다.
#   max_features="sqrt": 분기마다 8개 열 중 √8≈2개 열만 후보로 본다. 나무끼리 서로 다르게 자라게 한다.
#   random_state=42: 표본·열 추출 난수를 고정해 몇 번을 실행해도 같은 중요도가 나오게 한다.
#   n_jobs=1: CPU 코어 1개로 학습한다. 실행 환경과 상관없이 같은 방식으로 돌게 한다.
model = Pipeline([("pre", clone(pre)), ("model", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))])
# fit: train으로 전처리 복사본을 먼저 학습·변환하고, 그 결과로 숲을 학습한다.
#   이 한 줄이 끝나야 model["pre"]에 출력 열 이름이, model["model"]에 중요도가 생긴다.
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 출력 이름과 중요도 가져오기
# -----------------------------------------------------------------------------
# names: 파이프라인 안에서 학습된 전처리 단계의 출력 열 이름 8개이다. 
#   수정 전 pre.get_feature_names_out()은 학습되지 않은 원본 pre를 불러 NotFittedError가 난다.
#   학습된 것은 파이프라인 안의 복사본이므로 model["pre"]로 꺼내야 한다.
names = model["pre"].get_feature_names_out()
# values: 숲이 계산한 열별 중요도 8개이다. 합은 1이다.
#   수정 전 model.feature_importances_는 Pipeline 자체에 그 속성이 없어 AttributeError가 난다.
#   중요도는 파이프라인의 마지막 단계인 숲이 가지고 있으므로 model["model"]로 꺼내야 한다.
values = model["model"].feature_importances_

# -----------------------------------------------------------------------------
# 결과 표 만들기, 저장, 출력
# -----------------------------------------------------------------------------
# result: 이름과 중요도를 같은 행에 묶고 중요도가 큰 순서로 정렬한 표이다. (type: DataFrame)
#   ascending=False: 내림차순이다. 가장 많이 쓰인 열이 맨 위에 온다.
result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
# to_csv: outputs/04_importance.csv로 저장한다.
#   index=False: 행 번호는 의미가 없으므로 저장하지 않는다.
#   encoding="utf-8-sig": 엑셀에서 열어도 한글이 깨지지 않게 한다.
result.to_csv(OUT / "04_importance.csv", index=False, encoding="utf-8-sig")
# len(names): 변환된 열 개수이다. 8이 나와야 전처리 설계(숫자 2 + 원-핫 6)와 일치한다.
print("변환된 특성 수:", len(names))
# round(4): 소수점 넷째 자리까지 보여 준다. to_string(index=False): 행 번호 없이 표 전체를 출력한다.
print(result.round(4).to_string(index=False))
print("중요도는 원인이나 신규 데이터 성능을 보장하지 않습니다.")
