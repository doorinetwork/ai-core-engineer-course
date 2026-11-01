"""실습 03 심화: 회귀 숲의 트리 수 20·100을 비교합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (데이터 준비용)
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib.Path: 파일·폴더 경로를 객체로 다루는 클래스이다.
from pathlib import Path
# [외부 라이브러리] pandas: 표 형태 데이터를 다루는 라이브러리이다. pd라는 짧은 이름으로 쓴다.
import pandas as pd

# -----------------------------------------------------------------------------
# 데이터 읽기와 train/valid 분할
# -----------------------------------------------------------------------------
# HERE (type: Path): 이 스크립트 파일이 들어 있는 폴더의 절대 경로이다.
HERE = Path(__file__).resolve().parent
# raw (type: DataFrame): 요청 100건 전체 원본 표이다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest (type: DataFrame): 각 요청 ID가 train/valid/test 중 어디에 속하는지와 그 안의 순서(position)를 적은 표이다.
# 분할을 매번 새로 무작위로 하지 않고 파일로 고정해 두었기 때문에, 전날과 똑같은 train/valid를 다시 쓸 수 있다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids (type: Series): split이 "train"인 행만 골라 position 순서로 정렬한 요청 ID 목록이다(60개).
# position 순으로 정렬하는 이유: 행 순서까지 전날과 같아야 숲이 부트스트랩 표본을 뽑을 때도 같은 행이 뽑혀 결과가 재현된다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids (type: Series): split이 "valid"인 행만 골라 position 순서로 정렬한 요청 ID 목록이다(20개).
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train (type: DataFrame): raw에서 train_ids 순서대로 꺼낸 학습용 행 60개이다.
train = raw.loc[train_ids]
# valid (type: DataFrame): raw에서 valid_ids 순서대로 꺼낸 검증용 행 20개이다.
valid = raw.loc[valid_ids]
# FEATURES (type: list): 모델 입력으로 쓸 열 이름 4개이다.
# 숫자 2개(input_chars, allowed_tools)와 문자 2개(task_type, priority)이며, preprocessing.py의 FEATURES와 같은 순서이다.
# elapsed_ms는 이 실습의 정답이므로 입력에 넣으면 정답을 그대로 보여 주는 셈이 되어 넣지 않는다.
# retry_count·status·failed도 요청을 실행한 뒤에야 알 수 있는 값이라 예측 시점에는 없으므로 넣지 않는다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train (type: DataFrame): 학습 입력 60행 × 4열이다.
# X_valid (type: DataFrame): 검증 입력 20행 × 4열이다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train (type: Series): 학습용 failed 열(실패 1, 성공 0) 60개이다.
# y_valid (type: Series): 검증용 failed 열 20개이다.
y_train, y_valid = train["failed"], valid["failed"]
# OUT (type: Path): 결과 CSV를 저장할 outputs 폴더 경로이다.
OUT = HERE / "outputs"
# outputs 폴더를 만든다. exist_ok=True는 폴더가 이미 있어도 오류 없이 넘어가게 한다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (모델·평가용)
# -----------------------------------------------------------------------------
# [외부 라이브러리] sklearn.ensemble.RandomForestRegressor: 회귀용 랜덤포레스트 모델이다.
from sklearn.ensemble import RandomForestRegressor
# [외부 라이브러리] sklearn.metrics.mean_absolute_error, root_mean_squared_error: 회귀 오차 지표 함수이다.
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# [사용자 모듈] preprocessing.build_model: 이 과정에서 제공한 전처리+모델 묶음 함수이다.
from preprocessing import build_model

# -----------------------------------------------------------------------------
# 트리 수 20·100 비교 실험
# -----------------------------------------------------------------------------
# rows (type: list): 트리 수별 결과(dict)를 모을 빈 목록이다. 
rows = []
for count in [20, 100]:
    # model (type: Pipeline): 전처리기와 회귀 숲을 묶은 아직 학습 전의 모델이다. 반복마다 새로 만들어 앞 설정의 학습 결과가 섞이지 않는다.
    # n_estimators=count: 숲을 이루는 트리 수이다. 이 실험에서 유일하게 바꾸는 값이다.
    # max_depth=4: 각 트리가 질문을 최대 4단계까지만 하게 해 train 60건을 통째로 외우지 못하게 막는다.
    # min_samples_leaf=2: 잎 하나에 최소 2건이 있어야 해서, 요청 1건만 보고 예측값을 정하는 일을 막는다.
    # max_features=1.0: 분할마다 모든 특성(100%)을 후보로 본다. 트리끼리의 차이는 부트스트랩 표본 차이에서만 생긴다.
    # random_state=42: 부트스트랩 표본 뽑기를 고정해 다시 실행해도 같은 숫자가 나오게 한다.
    #   트리 수만 다르고 시드는 같으므로, 100그루 숲의 처음 20그루는 20그루 숲과 같은 트리이다.
    # n_jobs=1: CPU 1개로 순서대로 학습해 실행 환경이 달라도 결과가 똑같이 나오게 한다.
    model = build_model(RandomForestRegressor(n_estimators=count, max_depth=4,
                        min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=1))
    # train["elapsed_ms"] (type: Series, train 60건의 실행시간 ms)를 정답으로 넘긴다.
    #   fit은 train으로만 전처리 통계(중앙값·평균·표준편차·범주 목록)와 각 트리의 분할 기준·잎 평균값을 배운다.
    model.fit(X_train, train["elapsed_ms"])
    # prediction (type: ndarray): valid 20건의 예측 실행시간(ms, 실수)이다.
    # predict는 train에서 배운 통계로 valid를 변환만 하고 다시 학습하지 않으므로 valid의 정보가 모델에 새지 않는다.
    # 각 값은 트리 count그루가 낸 예측값(잎에 모인 train 실행시간의 평균)을 다시 평균한 것이다.
    prediction = model.predict(X_valid)
    # 이번 트리 수의 결과를 dict 하나로 rows에 추가한다.
    # "trees" (type: int): 트리 수이다.
    # "MAE_ms" (type: float): valid 실제 실행시간과 예측의 평균 절대 오차이다. "평균적으로 몇 ms 빗나가는가"를 뜻한다.
    # "RMSE_ms" (type: float): 평균 제곱근 오차이다. 크게 빗나간 요청에 벌점을 더 주므로 MAE보다 항상 크거나 같다.
    # 정답 쪽에 valid["elapsed_ms"]를 쓰므로, 학습 정답도 반드시 같은 elapsed_ms여야 비교가 성립한다.
    rows.append({"trees": count, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                 "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame): 트리 수 2가지 × (trees, MAE_ms, RMSE_ms) 비교표이다.
result = pd.DataFrame(rows)
# 비교표를 outputs/forest_regression.csv로 저장한다.
result.to_csv(OUT / "forest_regression.csv", index=False, encoding="utf-8-sig")
# 소수 셋째 자리로 반올림해 행 번호 없이 화면에 출력한다.
print(result.round(3).to_string(index=False))
print("트리 수만 바꾸었습니다. MAE·RMSE는 작을수록 좋고 단위는 ms입니다.")
