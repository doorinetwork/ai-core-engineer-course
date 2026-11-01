"""실습 06: test를 보지 않고 train 내부 3-fold로 두 표현을 비교하여 계획을 기록합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 데이터를 train / valid / test로 나누기 위해 외부 라이브러리 scikit-learn(sklearn)의 model_selection에서 train_test_split을 가져온다.
# 목적: 원본 표를 학습용·검증용·최종 평가용으로 나누기 위해 쓴다.
# 기능: 행을 무작위로 섞은 뒤 지정한 비율로 두 묶음으로 나눈다. stratify를 주면 정답 비율을 양쪽에 똑같이 유지한다.
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
# raw를 develop(80%)과 test(20%)로 나눈다.
# develop (type: DataFrame): 개발용 80%이다. 이 안에서 다시 train과 valid로 나눈다.
# test (type: DataFrame): 최종 평가용 20%이다. 지금은 id만 기록하고 열어 보지 않는다.
# test_size=0.2: 전체의 20%를 test로 떼어 둔다.
# stratify=raw["failed"]: 실패(1)와 성공(0)의 비율이 develop과 test에서 원본과 같게 유지되도록 나눈다. 행이 적을수록 한쪽에 실패가 몰리는 일을 막는 역할이 크다.
# random_state=42: 섞는 순서를 고정한다. 다시 실행해도 똑같은 행이 똑같은 묶음에 들어가므로 결과를 재현할 수 있다.
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
# develop을 train(75%)과 valid(25%)로 나눈다.
# develop은 전체의 80%이므로 0.8 × 0.25 = 0.2, 즉 valid는 전체의 20%이고 train은 전체의 60%이다. 20건 자료이면 train 12건, valid 4건이다.
# train (type: DataFrame): 모델을 학습하고 교차검증으로 표현을 비교하는 데 쓰는 유일한 자료이다.
# valid (type: DataFrame): 이 파일에서는 학습·선택에 쓰지 않고 id만 기록한다.
# stratify와 random_state의 의미는 위와 같다. 기준 열은 develop의 failed이다.
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

# -----------------------------------------------------------------------------
# 추가 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np
# 결과 기록을 위해 Python 표준 라이브러리 json을 가져온다.
import json
# 분류 모델을 만들기 위해 외부 라이브러리 scikit-learn의 tree에서 DecisionTreeClassifier를 가져온다.
# 목적: 두 특성 표현을 같은 조건의 결정 트리로 학습시켜 비교하기 위해 쓴다.
# 기능: "글자 수가 몇 이하인가" 같은 질문을 나무처럼 이어 붙여 실패(1)/성공(0)을 분류한다.
from sklearn.tree import DecisionTreeClassifier
# 교차검증을 위해 외부 라이브러리 scikit-learn의 model_selection에서 StratifiedKFold와 cross_val_score를 가져온다.
# 목적: test나 valid를 쓰지 않고 train 안에서만 두 표현의 성능을 공정하게 비교하기 위해 쓴다.
# 기능: StratifiedKFold는 train을 정답 비율이 같은 k개 묶음(fold)으로 나누는 규칙을 만들고,
#       cross_val_score는 그 규칙대로 "k-1개 묶음으로 학습 → 남은 1개 묶음으로 채점"을 k번 반복해 점수 k개를 돌려준다.
from sklearn.model_selection import StratifiedKFold, cross_val_score

# -----------------------------------------------------------------------------
# 파생 열 만들기
# -----------------------------------------------------------------------------
# train의 복사본을 만들어 train에 다시 저장한다.
train = train.copy()
# train의 글자 수에 log(1 + x)를 적용한 새 열 log_input_chars를 추가한다.
# train["log_input_chars"] (type: Series[float]): 각 행의 input_chars 하나만 써서 계산한 값이다.
# log가 아니라 log1p를 쓰는 이유: 글자 수가 0이면 log(0)은 -무한대가 되지만 log1p(0) = 0이라 안전하다.
train["log_input_chars"] = np.log1p(train["input_chars"])

# -----------------------------------------------------------------------------
# 교차검증 규칙과 비교 후보 정하기
# -----------------------------------------------------------------------------
# train을 3개의 fold로 나누는 규칙을 만들어 folds에 저장한다.
# folds: 아직 데이터를 나눈 것이 아니라 "어떻게 나눌지"를 담은 규칙 객체이다.
# n_splits=3: 3개 묶음으로 나눈다. 20건 자료의 train은 12건뿐이라 fold 하나가 4건이고, 이보다 더 잘게 나누면 fold 안에 실패(1)가 거의 없어 F1을 제대로 잴 수 없다.
# shuffle=True: 나누기 전에 행을 섞는다. 행 순서에 규칙이 있어도 특정 fold에 비슷한 행이 몰리지 않게 한다.
# random_state=42: 섞는 순서를 고정한다. 두 후보가 정확히 같은 fold로 채점되어야 공정한 비교가 되고, 다시 실행해도 같은 점수가 나온다.
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# 비교할 특성 표현 두 개를 이름과 열 목록으로 묶어 choices에 저장한다.
# choices:
# - "original": 원래 글자 수 1열만 쓴다.
# - "with_log": 원래 글자 수와 로그 변환 글자 수 2열을 함께 쓴다. 
choices = {"original": ["input_chars"], "with_log": ["input_chars", "log_input_chars"]}

# 후보 이름과 지표를 저장할 rows를 빈 리스트로 생성한다.
rows = []

# -----------------------------------------------------------------------------
# 두 표현을 3-fold 교차검증으로 비교하기
# -----------------------------------------------------------------------------
# choices의 두 후보를 차례로 꺼낸다.
for name, columns in choices.items():
    # 후보마다 새 결정 트리 모델을 만든다.
    # model: 아직 학습하지 않은 빈 모델이다. 두 후보에 똑같은 설정을 써야 차이가 "특성 표현" 하나에서만 생긴다.
    # max_depth=3: 질문을 최대 3단계까지만 이어 붙인다. train이 적을 때 나무가 너무 깊어지면 train 행을 외워 버리는 과적합이 생기므로 깊이를 제한한다.
    # random_state=42: 동점인 분기 후보 중 하나를 고를 때의 무작위성을 고정해 다시 실행해도 같은 나무가 나오게 한다.
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    # cross_val_score로 train 안에서 3-fold 교차검증을 하고, fold별 F1 점수 3개를 scores에 저장한다.
    # scores: fold 1, 2, 3을 각각 채점 fold로 썼을 때의 F1 점수이다.
    # - model: 위에서 만든 빈 모델이다. cross_val_score가 fold마다 복제본을 새로 학습하므로 원본 model은 학습되지 않은 채로 남는다.
    # - train[columns]: 입력(X)이다. train에서 이 후보의 열만 고른다. valid와 test는 넣지 않아 선택 과정에 전혀 쓰이지 않는다.
    # - train["failed"]: 정답(y)이다. 입력과 같은 train 행의 실패 여부이다.
    # - cv=folds: 위에서 만든 같은 fold 규칙을 넘긴다. 정수 3을 넘기면 섞지 않고 나누므로, 두 후보가 정확히 같은 fold로 채점되도록 반드시 folds를 넘긴다.
    # - scoring="f1": 양성(1=실패)에 대한 F1으로 채점한다. 실패가 적은 자료에서 정확도는 "모두 성공"이라고만 답해도 높게 나오므로, 실패를 얼마나 잘 찾는지 보는 F1을 쓴다.
    scores = cross_val_score(model, train[columns], train["failed"], cv=folds, scoring="f1")
    # 이 후보의 이름, F1 평균, F1 표준편차를 dict로 묶어 rows 끝에 추가한다.
    # "CV_F1": 3개 fold F1의 평균이며, 후보를 고르는 유일한 기준이다.
    # "CV_std": 3개 fold F1의 표준편차이며, fold에 따라 점수가 얼마나 흔들리는지 보여 주는 참고값이다.
    rows.append({"features": name, "CV_F1": scores.mean(), "CV_std": scores.std()})
# rows를 표로 바꿔 comparison에 저장한다.
# comparison: features, CV_F1, CV_std 열을 가진다. 행 순서는 choices에 적은 순서(original, with_log)이다.
comparison = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 후보 선택
# -----------------------------------------------------------------------------
# comparison["CV_F1"].idxmax()로 CV_F1이 가장 큰 행의 인덱스를 구하고, loc[그 행, "features"]로 그 행의 후보 이름을 꺼내 selected에 저장한다.
# idxmax는 최댓값이 여러 개(정확한 동점)이면 가장 앞에 있는 행의 인덱스를 돌려주므로, "동점이면 열이 더 적은 original"이라는 규칙을 별도 코드 없이 지킨다.
# 동점일 때 열이 적은 쪽을 고르는 이유: 성능이 같으면 만들고 검사할 것이 적은 단순한 표현이 재현과 유지에 유리하다.
# selected (type: str): 선택된 후보 이름("original" 또는 "with_log")이다.
selected = comparison.loc[comparison["CV_F1"].idxmax(), "features"]

# -----------------------------------------------------------------------------
# 평가 계획
# -----------------------------------------------------------------------------
# 최종 평가에 쓸 설정을 모두 dict로 모아 plan에 저장한다.
# plan: 최종 평가 때는 이 파일에 적힌 설정만 그대로 쓰고 다시 고르지 않는다.
# - "model": 사용할 모델 이름 "DecisionTreeClassifier"이다.
# - "max_depth": 비교 때와 같은 나무 깊이 3이다.
# - "random_state": 비교 때와 같은 seed 42이다.
# - "features": 선택된 후보의 열 목록이다. choices[selected]로 이름을 실제 열 목록으로 바꿔 적는다.
# - "threshold": 0.5이며, 예측 확률을 실패(1)/성공(0)으로 바꾸는 기준값이다.
# - "rule": "score > threshold"로, 확률이 정확히 0.5이면 양성으로 보지 않는다는 비교 방향까지 고정한다.
# - "positive_label": 1(실패)을 찾고 싶은 양성 클래스로 정한다.
# - "selection": 선택 근거(train12의 3-fold 평균 F1, 동점이면 original 우선)를 글로 남긴다.
# - "train_ids", "valid_ids", "test_ids": 각 분할에 속한 request_id를 분할 순서 그대로 적은 목록이다.
#   누가 어느 분할에 속했는지와 그 순서를 기록해 두어야 나중에 똑같은 분할로 재학습·재평가할 수 있다.
plan = {"model": "DecisionTreeClassifier", "max_depth": 3, "random_state": 42,
        "features": choices[selected], "threshold": 0.5, "rule": "score > threshold",
        "positive_label": 1, "selection": "train12 3-fold mean F1; tie: original first",
        "train_ids": train["request_id"].tolist(), "valid_ids": valid["request_id"].tolist(),
        "test_ids": test["request_id"].tolist()}

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# 비교 표를 이 파일과 같은 폴더의 feature_comparison.csv로 저장한다.
# index=False: 0, 1 같은 행 번호는 의미가 없으므로 저장하지 않는다.
# encoding="utf-8-sig": 맨 앞에 BOM 표시를 붙여 Excel에서 열어도 글자가 깨지지 않게 한다.
comparison.to_csv(HERE / "feature_comparison.csv", index=False, encoding="utf-8-sig")
# 평가 계획을 JSON 문자열로 바꿔 이 파일과 같은 폴더의 selection.json에 저장한다.
# ensure_ascii=False: 한글 같은 비영어권 문자를 \uXXXX 기호로 바꾸지 않고 그대로 쓴다.
# indent=2: 키마다 줄을 바꾸고 2칸 들여 써서 사람이 읽기 쉽게 만든다.
# encoding="utf-8": 파일을 UTF-8로 저장해 어느 운영체제에서 열어도 같은 글자로 읽히게 한다.
(HERE / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
# 비교 표를 소수점 셋째 자리로 반올림해 행 번호 없이 출력한다.
print(comparison.round(3).to_string(index=False))
# 선택된 후보 이름과 판정 규칙(확률이 0.5를 넘으면 실패로 예측)을 출력한다.
print("선택:", selected, "/ 임계값 0.5 초과일 때 실패")
print("설정 기록: selection.json / test는 아직 예측하지 않았습니다.")
