"""실습 06 심화: 같은 회귀 KFold에서 음수 RMSE 부호를 되돌립니다."""
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
#   KFold(shuffle=True)는 "행 번호"를 섞어 fold를 만든다. 행 순서가 전날과 같아야 같은 행이 같은 fold에 들어가 결과가 재현된다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids (type: Series): valid에 속한 아이디 20개를 position 순서대로 뽑은 것이다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train (type: DataFrame): 학습용 60행이다. 
train = raw.loc[train_ids]
# valid (type: DataFrame): 검증용 20행이다. 
valid = raw.loc[valid_ids]
# FEATURES (type: list of str): 모델에 넣을 원본 입력 열 이름 4개이다. 앞 2개는 숫자 열, 뒤 2개는 문자 열이다.
#   정답인 elapsed_ms와, 처리 결과라서 미리 알 수 없는 retry_count·status·failed는 넣지 않는다. 넣으면 예측 시점에 없는 정보로 답을 베끼는 누수가 된다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train, X_valid (type: DataFrame, DataFrame): 입력 4개 열만 남긴 표이다.
#   전처리 전 원본 상태로 두어야 Pipeline이 fold마다 결측 대체·스케일링·원-핫 인코딩을 학습 fold 기준으로 새로 한다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train, y_valid (type: Series, Series): 분류 정답 failed(0=성공, 1=실패)이다.
#   앞 실습들과 똑같이 만들어 두는 제공 코드이며, 이 회귀 실습에서는 쓰지 않는다. 회귀 정답은 아래 TODO에서 elapsed_ms를 직접 꺼내 쓴다.
y_train, y_valid = train["failed"], valid["failed"]
# OUT (type: Path): 결과 CSV를 저장할 outputs 폴더 경로이다. 
OUT = HERE / "outputs"
# outputs 폴더를 만든다. exist_ok=True이므로 폴더가 이미 있어도 오류를 내지 않고 넘어간다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 라이브러리 불러오기 (모델·교차검증용)
# -----------------------------------------------------------------------------
# [외부 라이브러리] scikit-learn의 DecisionTreeRegressor: 결정 트리 회귀 모델이다.
from sklearn.tree import DecisionTreeRegressor
# [외부 라이브러리] scikit-learn의 RandomForestRegressor: 랜덤 포레스트 회귀 모델이다.
from sklearn.ensemble import RandomForestRegressor
# [외부 라이브러리] xgboost의 XGBRegressor: 그레이디언트 부스팅 방식의 회귀 모델이다.
from xgboost import XGBRegressor
# [외부 라이브러리] scikit-learn의 KFold, cross_val_score: 교차검증 도구이다.
from sklearn.model_selection import KFold, cross_val_score
# [사용자 모듈] preprocessing.build_model: 이 과정에서 제공한 전처리+모델 묶음 함수이다.
from preprocessing import build_model

# -----------------------------------------------------------------------------
# fold 고정과 후보 모델 만들기 (제공 코드)
# -----------------------------------------------------------------------------
# folds (type: list, 원소는 (ndarray, ndarray) 튜플 5개): train 60행을 5개 fold로 나눈 (학습 행 번호 48개, 평가 행 번호 12개) 쌍이다.
#   n_splits=5: 60행을 12행씩 5조각으로 나눠, 매번 1조각(12행)으로 평가하고 나머지 4조각(48행)으로 학습한다.
#   shuffle=True: 행 순서대로 자르지 않고 섞은 뒤 나눈다. 원본 순서에 치우침이 있어도 한 fold에 몰리지 않게 한다.
#   random_state=42: 섞는 방법을 고정해 다시 실행해도 같은 fold가 나오게 한다.
#   list(...)로 미리 펼쳐 두는 이유는 세 후보가 정확히 같은 fold 5개로 평가받게 하기 위해서이다. fold가 같아야 점수 차이를 모델 차이로 읽을 수 있다.
#   split(X_train)에는 정답을 주지 않는다. KFold는 정답과 상관없이 행 번호만으로 나누기 때문이다.
folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
# candidates (type: list of tuple(str, 회귀 모델)): (이름, 아직 학습 전인 회귀 모델) 쌍 3개이다.
#   "Tree": max_depth=3으로 질문을 최대 3번까지만 하고, min_samples_leaf=3으로 잎 하나에 최소 3행이 모이게 해 48행을 외우지 못하게 한다.
#     random_state=42는 같은 점수의 분할 후보가 여럿일 때 고르는 순서를 고정한다.
#   "Forest": n_estimators=100은 트리 100개의 평균을 쓴다는 뜻이다. max_depth=4, min_samples_leaf=2로 트리 하나의 복잡도를 제한한다.
#       max_features=1.0은 분기마다 특성을 모두 후보로 본다는 뜻이다. 회귀 기본값과 같고, 다양성은 행 복원 추출만으로 만든다.
#       random_state=42는 복원 추출을 고정하고, n_jobs=1은 CPU 1개만 써서 결과를 그대로 재현한다.
#   "XGB": 보정 트리 80개, 깊이 2, 한 회차 보정량의 5%만 반영한다.
#       subsample=0.8은 회차마다 행의 80%만 쓰고, colsample_bytree=1.0은 열을 모두 쓰고, reg_lambda=1.0은 잎 값에 L2 규제를 건다.
#       objective="reg:squarederror"는 제곱 오차를 줄이도록 학습하고, eval_metric="rmse"는 진단 지표를 RMSE로 정한다. eval_set이 없으므로 회차 결정에는 쓰이지 않는다.
#       tree_method="hist"는 숫자를 구간으로 묶어 빠르게 분할하고, device="cpu"는 GPU 없이 계산한다. random_state=42·n_jobs=1로 결과를 재현한다.
candidates = [("Tree", DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2,
                  max_features=1.0, random_state=42, n_jobs=1)), 
              ("XGB", XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
# rows (type: list of dict): 후보마다 CV 평균·표준편차를 담을 요약 행 목록이다. 처음에는 비어 있다.
# detail (type: list of dict): 후보×fold마다 RMSE 하나씩을 담을 상세 행 목록이다. 처음에는 비어 있다.
rows, detail = [], []

# -----------------------------------------------------------------------------
# 후보별 교차검증
# -----------------------------------------------------------------------------
for name, estimator in candidates:
    # model (type: Pipeline): 전처리 "pre"와 회귀 모델 "model"을 묶은 전체 모델이다.
    #   전처리까지 Pipeline 안에 넣어야 fold마다 중앙값·평균·범주 목록을 학습 48행에서만 다시 계산해, 평가 12행 정보가 새지 않는다.
    #   build_model의 기본값 scale="standard"라서 숫자 2열이 표준화된다. 트리는 값의 크고 작은 순서로만 나누므로 Tree·XGB 점수는 scale="none"일 때와 같다.
    model = build_model(estimator)
    # 회귀 정답으로 neg_root_mean_squared_error 점수를 구한다.
    # negative (type: ndarray of float, 5개): fold 5개의 "음수 RMSE"이다. 예를 들어 RMSE가 700ms면 -700이 들어 있다.
    #   train["elapsed_ms"] (type: Series of int): 회귀 정답인 실제 처리 시간(ms)이다. y_train(failed 0/1)을 넣으면 실패 여부를 숫자로 맞히는 엉뚱한 문제가 되므로 반드시 이 열을 쓴다.
    #   cv=folds: 위에서 고정한 같은 fold 5개를 세 후보에 똑같이 쓴다. 점수 순서는 folds 순서와 같다.
    #   scoring="neg_root_mean_squared_error": fold마다 평가 12행의 RMSE를 계산해 마이너스를 붙여 돌려준다.
    #     scikit-learn의 scorer는 모두 "클수록 좋다"는 규칙을 따른다. RMSE는 작을수록 좋으므로 부호를 뒤집어야 이 규칙에 맞는다.
    #   cross_val_score는 fold마다 model을 복제해 48행으로 fit하고 12행은 예측만 한다. 원본 model 객체는 학습되지 않은 채 남는다.
    negative = cross_val_score(model, X_train, train["elapsed_ms"], cv=folds,
                               scoring="neg_root_mean_squared_error")
    # 음수 scorer 값을 양수 RMSE로 되돌린다.
    # rmse (type: ndarray of float, 5개): fold 5개의 RMSE(ms)이다. negative에 마이너스를 붙여 부호를 원래대로 되돌린다.
    #   abs()가 아니라 -를 쓰는 이유는 scorer가 붙인 부호를 정확히 한 번 되돌리는 것이 목적이기 때문이다. RMSE는 항상 0 이상이므로 결과는 모두 양수가 된다.
    rmse = -negative
    # 요약 한 행을 rows에 추가한다.
    #   "model" (type: str): 후보 이름이다.
    #   "CV_RMSE_ms" (type: float): fold 5개 RMSE의 단순 평균이다. 작을수록 처리 시간을 잘 맞힌 후보이다.
    #   "CV_std_ms" (type: float): fold 5개 RMSE의 표준편차이다. 클수록 어떤 12행으로 평가하느냐에 따라 성적이 크게 흔들린다는 뜻이다.
    #     numpy의 std()는 기본값 ddof=0(모집단 표준편차)으로 계산한다. pandas Series.std()의 기본값 ddof=1과 다르다.
    rows.append({"model": name, "CV_RMSE_ms": rmse.mean(), "CV_std_ms": rmse.std()})
    # fold 번호를 1부터 붙여 fold별 RMSE를 하나씩 꺼낸다. number (type: int)는 fold 번호, value (type: float)는 그 fold의 RMSE이다.
    for number, value in enumerate(rmse, start=1):
        # 상세 한 행을 detail에 추가한다. fit_n=48은 학습 행 수, evaluate_n=12는 평가 행 수이다.
        #   60행을 5-fold로 나누면 모든 fold가 48/12로 똑같이 나뉘므로 고정값으로 적는다.
        detail.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "RMSE_ms": value})

# -----------------------------------------------------------------------------
# 결과 저장과 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame, 3행 3열): 후보 3개의 CV 평균 RMSE와 표준편차 요약표이다.
result = pd.DataFrame(rows)
# 요약표를 outputs/regression_cv.csv로 저장한다.
result.to_csv(OUT / "regression_cv.csv", index=False, encoding="utf-8-sig")
# 상세표(15행 = 후보 3개 × fold 5개)를 DataFrame으로 만들어 outputs/cv_scores.csv로 저장한다.
pd.DataFrame(detail).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
# 요약표를 소수 셋째 자리로 반올림하고 행 번호 없이 화면에 출력한다.
print(result.round(3).to_string(index=False))
# CV_RMSE를 해석할 때의 주의점을 화면에 남긴다.
#   fold마다 제곱근을 먼저 씌운 뒤 평균했으므로, 60행 예측을 한데 모아 한 번에 계산한 RMSE와 값이 다르다.
#   제곱근은 큰 값을 덜 키우는 함수라서, fold별 RMSE의 평균은 전체를 합친 RMSE보다 작거나 같다.
print("CV_RMSE는 fold별 RMSE의 단순 평균입니다. 전체 예측을 합친 RMSE와 다를 수 있습니다.")
