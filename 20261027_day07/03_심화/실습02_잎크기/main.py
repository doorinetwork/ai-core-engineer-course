"""실습 02 심화: 깊이는 고정하고 최소 잎 표본 수만 비교합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (데이터 준비용)
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [외부 라이브러리] pandas: 표 형태 데이터를 다루는 라이브러리이다. pd라는 짧은 이름으로 쓴다.
import pandas as pd

# -----------------------------------------------------------------------------
# 데이터 읽기와 train/valid 분할 (제공 코드)
# -----------------------------------------------------------------------------
# HERE (type: Path): 이 스크립트 파일이 들어 있는 폴더의 절대 경로이다.
HERE = Path(__file__).resolve().parent
# raw (type: DataFrame): 요청 100건 전체 원본 표이다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest (type: DataFrame): 각 요청 ID가 train/valid/test 중 어디에 속하는지와 그 안의 순서(position)를 적은 표이다.
# 분할을 매번 새로 무작위로 하지 않고 파일로 고정해 두었기 때문에, 전날과 똑같은 train/valid를 다시 쓸 수 있다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids (type: Series): split이 "train"인 행만 골라 position 순서로 정렬한 요청 ID 목록이다(60개).
# position 순으로 정렬하는 이유: 행 순서까지 전날과 같아야 트리가 동점 분기를 고를 때도 같은 결과가 나온다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids (type: Series): split이 "valid"인 행만 골라 position 순서로 정렬한 요청 ID 목록이다(20개).
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train (type: DataFrame): raw에서 train_ids 순서대로 꺼낸 학습용 행 60개이다.
train = raw.loc[train_ids]
# valid (type: DataFrame): raw에서 valid_ids 순서대로 꺼낸 검증용 행 20개이다.
valid = raw.loc[valid_ids]
# FEATURES (type: list): 모델 입력으로 쓸 열 이름 4개이다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train (type: DataFrame): 학습 입력 60행 × 4열이다.
# X_valid (type: DataFrame): 검증 입력 20행 × 4열이다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train (type: Series): 학습 정답(failed: 실패 1, 성공 0) 60개이다.
# y_valid (type: Series): 검증 정답 20개이다.
y_train, y_valid = train["failed"], valid["failed"]
# OUT (type: Path): 결과 CSV를 저장할 outputs 폴더 경로이다.
OUT = HERE / "outputs"
# outputs 폴더를 만든다. exist_ok=True는 폴더가 이미 있어도 오류 없이 넘어가게 한다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (모델·평가용)
# -----------------------------------------------------------------------------
# [외부 라이브러리] sklearn.tree.DecisionTreeClassifier: scikit-learn의 분류용 결정 트리 클래스이다.
from sklearn.tree import DecisionTreeClassifier
# [외부 라이브러리] sklearn.metrics.f1_score: 분류 성능 지표 F1을 계산하는 함수이다.
from sklearn.metrics import f1_score

# [사용자 모듈] preprocessing.build_model: 같은 폴더의 preprocessing.py에 있는 함수이다.
from preprocessing import build_model


# -----------------------------------------------------------------------------
# 최소 잎 표본 수별 모델 만들기·학습·평가
# -----------------------------------------------------------------------------
# rows (type: list): 조건 하나당 결과 한 줄(dict)을 모아 둘 빈 리스트이다.
rows = []
# leaf (type: int): 이번 반복에서 시험할 최소 잎 표본 수이다.
for leaf in [1, 3, 5]:
    # model (type: Pipeline): 전처리("pre") + 결정 트리("model") 두 단계로 된 파이프라인이다.
    # max_depth=3: 질문을 최대 3번까지만 이어서 하게 막는다. 모든 반복에서 3으로 고정해야 차이가 leaf 때문인지 알 수 있다.
    # min_samples_leaf=leaf: 나눈 뒤 양쪽 잎에 학습 행이 leaf개 이상 남는 분기만 허용한다.
    #   값이 클수록 소수 몇 개만 따로 떼어 내는 분기를 금지하므로, 학습 데이터 암기(과대적합)를 줄인다.
    # random_state=42: 점수가 같은 분기 후보가 여럿일 때 고르는 순서를 고정해 매번 같은 트리가 나오게 한다.
    # build_model은 호출할 때마다 새 Pipeline을 만들기 때문에, 반복마다 이전 학습 결과가 섞이지 않는다.
    model = build_model(DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf, random_state=42))
    # 학습 데이터로 파이프라인 전체를 학습한다.
    # 전처리 단계는 train에서만 중앙값·최빈값·평균·표준편차·범주 목록을 배우고, 트리는 변환된 값으로 분기를 배운다.
    # valid를 fit에 넣지 않는 이유: valid 정보가 학습에 섞이면 valid 점수가 더 이상 "처음 보는 데이터" 점수가 아니게 된다.
    model.fit(X_train, y_train)
    # 이번 조건의 결과 한 줄(dict)을 rows에 추가한다.
    # "min_samples_leaf" (type: int): 이번에 쓴 최소 잎 표본 수이다.
    # "leaves" (type: int): 학습된 트리의 잎 개수이다. model["model"]은 파이프라인에서 이름이 "model"인 단계,
    #   즉 DecisionTreeClassifier를 꺼낸다. build_model이 이 이름을 붙였기 때문에 이 이름으로 꺼낼 수 있다.
    # "train_F1" (type: float): 학습 데이터에 대한 F1이다. 높을수록 학습 데이터를 잘 맞힌 것이다.
    # "valid_F1" (type: float): 검증 데이터에 대한 F1이다. 처음 보는 데이터에서 얼마나 잘하는지를 나타낸다.
    # zero_division=0: 모델이 실패(1)를 하나도 예측하지 않아 정밀도 분모가 0이 되면, 경고 대신 0점으로 처리한다.
    rows.append({"min_samples_leaf": leaf, "leaves": model["model"].get_n_leaves(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0)})

# -----------------------------------------------------------------------------
# 결과 표 저장과 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame): rows의 dict 3개를 3행 × 4열 표로 바꾼 것이다.
result = pd.DataFrame(rows)
# 결과 표를 outputs/leaf_comparison.csv로 저장한다.
result.to_csv(OUT / "leaf_comparison.csv", index=False, encoding="utf-8-sig")
# 소수 넷째 자리까지 반올림한 표를 행 번호 없이 출력한다.
print(result.round(4).to_string(index=False))
