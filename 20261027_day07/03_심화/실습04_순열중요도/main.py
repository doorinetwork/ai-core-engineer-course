"""실습 04 심화: 변환 8열 MDI와 원본 4열 순열 중요도를 구분합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (자료 준비용)
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib의 Path: 파일 경로를 운영체제와 상관없이 다루는 도구이다.
from pathlib import Path
# [외부 라이브러리] pandas: 표 형태 자료를 다루는 라이브러리이다.
import pandas as pd

# -----------------------------------------------------------------------------
# 데이터 불러오기와 분할
# -----------------------------------------------------------------------------
# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽는다.
HERE = Path(__file__).resolve().parent
# raw (type: DataFrame): 전체 요청 자료 100건이다. request_id를 행 이름(index)으로 삼는다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest (type: DataFrame): 각 request_id가 train/valid/test 중 어디에 속하는지와 순서(position)를 적은 표이다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids (type: Series): train에 속한 아이디 60개를 position 순서대로 뽑은 것이다.
#   position 순으로 정렬해야 행 순서가 전날과 같아지고, 랜덤포레스트의 표본 추출 결과도 매번 같아진다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids (type: Series): valid에 속한 아이디 20개를 position 순서대로 뽑은 것이다.
#   순열 중요도는 행을 섞는 계산이라, 행 순서가 같아야 random_state=42로 섞은 결과도 똑같이 재현된다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train (type: DataFrame): 학습용 60행이다. 모델 학습과 MDI 계산에 쓰인다.
train = raw.loc[train_ids]
# valid (type: DataFrame): 검증용 20행이다. 이 실습에서는 순열 중요도를 계산하는 평가 자료로 쓴다.
valid = raw.loc[valid_ids]
# FEATURES (type: list of str): 모델에 넣을 원본 입력 열 이름 4개이다. 앞 2개는 숫자 열, 뒤 2개는 문자 열이다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train, X_valid (type: DataFrame): 입력 4개 열만 남긴 표이다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train, y_valid (type: Series): 정답 열 failed(0=성공, 1=실패)이다.
y_train, y_valid = train["failed"], valid["failed"]
# OUT (type: Path): 결과 CSV를 저장할 outputs 폴더 경로이다.
OUT = HERE / "outputs"
# exist_ok=True: 폴더가 이미 있어도 오류를 내지 않고 넘어간다. 여러 번 실행해도 멈추지 않게 하려는 설정이다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (모델·중요도용)
# -----------------------------------------------------------------------------
# [외부 라이브러리] scikit-learn의 RandomForestClassifier: 결정트리 여러 그루의 투표로 분류하는 모델이다.
from sklearn.ensemble import RandomForestClassifier
# [외부 라이브러리] scikit-learn의 permutation_importance: 순열 중요도를 계산하는 함수이다.
from sklearn.inspection import permutation_importance
# [사용자 모듈] preprocessing.build_model: 이 과정에서 제공한 전처리+모델 묶음 함수이다.
from preprocessing import build_model

# -----------------------------------------------------------------------------
# 모델 만들기와 학습
# -----------------------------------------------------------------------------
# model (type: Pipeline): 전처리 단계 "pre"와 랜덤포레스트 단계 "model"을 묶은 전체 모델이다.
#   n_estimators=100: 트리를 100그루 만든다. 그루 수가 많을수록 투표 결과가 안정된다.
#   max_depth=4: 트리 깊이를 4단계로 제한한다. 60행밖에 없는 train을 통째로 외우지 못하게 막는 설정이다.
#   min_samples_leaf=2: 잎 하나에 최소 2행이 들어가게 한다. 한 행만을 위한 규칙이 생기는 것을 막는다.
#   max_features="sqrt": 분기마다 전체 8개 열 중 √8≈2개만 후보로 본다. 트리끼리 서로 달라져야 투표 효과가 커진다.
#   random_state=42: 행 뽑기·열 후보 뽑기의 무작위를 고정해 매번 같은 숲이 만들어지게 한다.
#   n_jobs=1: CPU 하나로만 계산한다. 병렬 계산 순서 차이 없이 결과를 그대로 재현하기 위한 설정이다.
model = build_model(RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))
# 학습: train 60행으로 전처리 기준값(중앙값·평균·범주 목록)과 트리 100그루를 함께 학습한다.
#   valid는 여기서 쓰지 않는다. valid까지 학습에 넣으면 아래 순열 중요도가 "본 적 있는 자료"로 계산되어 의미가 없어진다.
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# MDI 중요도: 변환 후 8개 열 기준
# -----------------------------------------------------------------------------
# mdi (type: DataFrame): 전처리 출력 열 이름 8개와 랜덤포레스트의 MDI 중요도를 한 행씩 짝지은 표이다.
#   model["pre"].get_feature_names_out(): 전처리 뒤 실제로 모델에 들어간 열 이름 8개를 순서대로 돌려준다. (type: ndarray of str)
#     예: num__input_chars, num__allowed_tools, cat__task_type_qa, ..., cat__priority_normal 처럼 원-핫 더미 열이 따로 나온다.
#   model["model"].feature_importances_: 위 8개 열 순서와 똑같은 순서의 MDI 값이다. 합이 1이 되도록 정규화되어 있다. (type: ndarray of float)
#   두 배열의 순서가 같으므로 같은 위치끼리 짝지으면 이름표가 정확히 맞는다.
mdi = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                    "importance": model["model"].feature_importances_})

# -----------------------------------------------------------------------------
# 순열 중요도: 원본 4개 열 기준
# -----------------------------------------------------------------------------
# result (type: Bunch): 순열 중요도 계산 결과 묶음이다.
#   importances_mean(열별 평균 감소량), importances_std(반복 간 표준편차), importances(열×반복 전체 값)를 담는다.
#   model: 전처리를 품은 Pipeline 전체를 넘긴다. 그래서 원본 4개 열 단위로 섞이고,
#     task_type 같은 문자 열을 섞으면 그 열에서 나온 원-핫 더미 묶음 전체가 함께 흔들린다.
#   X_valid, y_valid: train이 아닌 valid로 잰다. 모델이 본 적 없는 자료에서 점수가 떨어져야 "일반화에 쓰이는 열"이라고 볼 수 있다.
#   scoring="f1": 실패(1) 클래스의 F1 점수가 얼마나 떨어지는지 잰다. 이 과제는 실패 요청을 잘 잡아내는 것이 목표이므로 정확도가 아니라 F1을 쓴다.
#   n_repeats=20: 열 하나를 20번 서로 다르게 섞어 감소량을 20개 얻는다. 한 번만 섞으면 우연히 잘/못 섞인 결과에 휘둘리므로 평균과 흔들림(표준편차)을 함께 본다.
#   random_state=42: 섞는 순서를 고정해 다시 실행해도 같은 중요도가 나오게 한다.
#   n_jobs=1: CPU 하나로 계산해 결과 재현성을 지킨다.
#   모델을 다시 fit하지 않는다. 이미 학습한 모델은 그대로 두고, 입력 열만 섞어 예측 점수를 다시 매긴다.
# 쉽게 말하면, 학생(모델)은 그대로 두고 시험지의 한 항목만 뒤죽박죽 섞었을 때 점수가 얼마나 깎이는지를 20번 재 보는 것이다.
result = permutation_importance(model, X_valid, y_valid, scoring="f1",
                                n_repeats=10, random_state=42, n_jobs=1)

# 원본 열을 하나씩 섞은 뒤 F1 감소량을 본다. 모델을 다시 학습하지 않는다.
# permutation (type: DataFrame): 원본 4개 열마다 F1 감소량 평균과 반복 표준편차를 정리한 표이다.
#   "feature": FEATURES 순서이다. permutation_importance는 X_valid의 열 순서대로 결과를 내므로 FEATURES와 순서가 정확히 맞는다.
#   "mean_F1_drop": 20번 섞었을 때 (기준 F1 − 섞은 F1)의 평균이다. 클수록 그 열이 없으면 예측이 많이 망가진다는 뜻이다. (type: ndarray of float)
#   "repeat_std": 20번 반복 사이의 표준편차이다. 섞는 방식에 따른 흔들림이며, 신뢰구간이나 표본 전체의 불확실성과는 다르다. (type: ndarray of float)
permutation = pd.DataFrame({"feature": FEATURES,
                            "mean_F1_drop": result.importances_mean,
                            "repeat_std": result.importances_std})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
# MDI 8행 표를 CSV로 저장한다.
#   두 중요도는 대상 열·자료·단위가 다르므로 한 파일에 합치지 않고 파일을 나눠 저장한다.
mdi.to_csv(OUT / "mdi_importance.csv", index=False, encoding="utf-8-sig")
# 순열 중요도 4행 표를 CSV로 저장한다. 설정 이유는 바로 위 MDI 저장과 같다.
permutation.to_csv(OUT / "permutation_importance.csv", index=False, encoding="utf-8-sig")
# 두 표의 행 수(8과 4)를 출력해 대상 열이 다르다는 점을 눈으로 확인한다.
print("MDI: 변환 후", len(mdi), "열 / 순열: 원본", len(permutation), "열")
# 순열 중요도 표를 소수 넷째 자리로 반올림해 출력한다. to_string(index=False)는 행 번호 없이 표만 보여 준다.
print(permutation.round(4).to_string(index=False))
print("음수는 이 valid 표본에서 섞었을 때 점수가 높아졌다는 뜻입니다. 인과관계는 아닙니다.")
