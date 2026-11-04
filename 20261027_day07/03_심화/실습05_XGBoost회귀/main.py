"""실습 05 심화: XGBoost의 회귀 목적과 오차를 연결합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# [표준 라이브러리] pathlib의 Path: 파일 경로를 운영체제와 상관없이 다루는 도구이다.
from pathlib import Path
# [외부 라이브러리] pandas: 표 형태 자료를 다루는 라이브러리이다.
import pandas as pd

# -----------------------------------------------------------------------------
# 데이터 불러오기와 분할
# -----------------------------------------------------------------------------
# HERE (type: Path): 이 .py 파일이 들어 있는 폴더의 절대 경로이다.
HERE = Path(__file__).resolve().parent
# raw (type: DataFrame): 전체 요청 자료 100건이다. request_id를 행 이름(index)으로 삼는다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest (type: DataFrame): 각 request_id가 train/valid/test 중 어디에 속하는지와 순서(position)를 적은 표이다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids (type: Series): train에 속한 아이디 60개를 position 순서대로 뽑은 것이다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids (type: Series): valid에 속한 아이디 20개를 position 순서대로 뽑은 것이다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train (type: DataFrame): 학습용 60행이다. 정답 열 elapsed_ms, failed도 함께 들어 있다.
train = raw.loc[train_ids]
# valid (type: DataFrame): 검증용 20행이다. 이 실습에서는 학습이 끝난 모델의 오차를 재는 평가 자료로만 쓴다.
valid = raw.loc[valid_ids]
# FEATURES (type: list of str): 모델에 넣을 원본 입력 열 이름 4개이다. 앞 2개는 숫자 열, 뒤 2개는 문자 열이다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train, X_valid (type: DataFrame, DataFrame): 입력 4개 열만 남긴 표이다.
#   전처리 전 원본 상태로 두어야 Pipeline이 스스로 결측 대체·스케일링·원-핫 인코딩을 train 기준으로 한다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train, y_valid (type: Series, Series): 분류 정답 failed(0=성공, 1=실패)이다.
y_train, y_valid = train["failed"], valid["failed"]
# OUT (type: Path): 결과 CSV를 저장할 outputs 폴더 경로이다. 
OUT = HERE / "outputs"
# outputs 폴더를 만든다. exist_ok=True이므로 폴더가 이미 있어도 오류를 내지 않고 넘어간다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (모델·평가용)
# -----------------------------------------------------------------------------
# [외부 라이브러리] xgboost의 XGBRegressor: 그레이디언트 부스팅 방식의 회귀 모델이다.
from xgboost import XGBRegressor
# [외부 라이브러리] scikit-learn의 mean_absolute_error, root_mean_squared_error: 회귀 오차를 재는 함수이다.
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
# [사용자 모듈] preprocessing.build_model: 이 과정에서 제공한 전처리+모델 묶음 함수이다.
from preprocessing import build_model

# -----------------------------------------------------------------------------
# 모델 만들기
# -----------------------------------------------------------------------------
# 목적함수는 회귀용 squared error, 진단 지표는 rmse이다.
# estimator (type: XGBRegressor): XGBoost 회귀 모델 설정이다. 아직 학습 전이다.
#   n_estimators=80: 보정 트리를 80회차 차례로 쌓는다. 회차 수를 고정해 두었으므로 조기 종료를 쓰지 않는다.
#   max_depth=2: 트리 하나가 질문을 최대 2번까지만 하게 해, 60행짜리 train을 외우지 못하게 복잡도를 낮춘다.
#   learning_rate=0.05: 한 회차가 계산한 보정량의 5%만 반영한다. 천천히 고쳐 나가야 한 트리의 실수에 크게 흔들리지 않는다.
#   subsample=0.8: 회차마다 train 행의 80%만 무작위로 골라 트리를 만든다. 매번 조금씩 다른 행을 보게 해 과적합을 줄인다.
#   colsample_bytree=1.0: 트리마다 특성 열을 전부 쓴다. 열은 무작위로 빼지 않는다.
#   reg_lambda=1.0: 잎 값(보정량)에 L2 규제를 걸어, 행이 적은 잎에서 보정량이 지나치게 커지지 않게 한다.
#   objective="reg:squarederror": (정답−예측)²을 줄이는 것을 학습 목표로 삼는다. 분류용 binary:logistic과 달리 확률이 아니라 ms 숫자를 그대로 출력한다.
#   eval_metric="rmse": 학습 진단용 지표를 RMSE로 정한다. 제곱 오차 목적과 같은 기준이며, 이 코드에서는 eval_set을 주지 않으므로 계산만 설정되고 회차 결정에는 쓰이지 않는다.
#   tree_method="hist": 숫자 값을 구간(히스토그램)으로 묶어 분할 기준 후보를 빠르게 찾는다.
#   device="cpu": GPU 없이 CPU로 학습한다. 실행 환경이 달라도 같은 계산 경로를 쓰게 한다.
#   random_state=42: subsample의 무작위 행 선택을 고정해 다시 실행해도 같은 결과가 나오게 한다.
#   n_jobs=1: CPU 1개만 써서 병렬 계산 순서 차이 없이 결과를 그대로 재현한다.
estimator = XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1)
# model (type: Pipeline): 전처리 단계 "pre"와 XGBoost 회귀 단계 "model"을 묶은 전체 모델이다.
model = build_model(estimator)

# -----------------------------------------------------------------------------
# 학습과 예측
# -----------------------------------------------------------------------------
# 회귀 정답 elapsed_ms로 학습하고 valid를 예측한다.
# train 60행의 입력 4열과 회귀 정답 elapsed_ms로 Pipeline 전체를 학습한다.
#   train["elapsed_ms"] (type: Series of int): 각 요청의 실제 처리 시간(ms)이다. 위에서 만든 y_train(failed)을 넣으면 0/1을 맞히는 엉뚱한 문제가 되므로 반드시 이 열을 쓴다.
#   전처리기의 중앙값·최빈값·평균·표준편차·범주 목록과 80개 보정 트리가 모두 이때 train에서만 정해진다.
#   valid는 fit에 넣지 않고 eval_set이나 early_stopping_rounds도 주지 않는다. 그래서 valid가 회차 수 결정에도 쓰이지 않아 아래 valid 오차가 부풀려지지 않는다.
model.fit(X_train, train["elapsed_ms"])
# prediction (type: ndarray of float32, 20개): valid 20건의 예측 처리 시간(ms)이다.
#   predict는 train에서 배운 기준으로 valid를 변환만 하고 다시 학습하지 않으므로 valid 정보가 모델에 새지 않는다.
#   분류와 달리 0.5 같은 기준선으로 자르지 않고, 80개 트리의 보정량을 모두 더한 숫자를 그대로 낸다.
prediction = model.predict(X_valid)

# -----------------------------------------------------------------------------
# 평가 및 결과
# -----------------------------------------------------------------------------
# result (type: DataFrame, 1행 3열): 모델 이름과 valid 오차 두 개를 한 행으로 묶은 결과표이다.
#   "model": 결과를 다른 후보와 합칠 때 구분하기 위한 이름표 "XGB"이다. (type: str)
#   "MAE_ms": valid에서 예측이 실제 시간과 평균적으로 몇 ms 차이 나는지이다. (type: float)
result = pd.DataFrame([{"model": "XGB", "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                       "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)}])
# 결과표를 outputs/xgb_regression.csv로 저장한다.
result.to_csv(OUT / "xgb_regression.csv", index=False, encoding="utf-8-sig")
# 소수 셋째 자리로 반올림하고 행 번호 없이 화면에 출력한다.
print(result.round(3).to_string(index=False))
print("고정 80회차·깊이2입니다. valid를 조기 종료에 사용하지 않았습니다.")
