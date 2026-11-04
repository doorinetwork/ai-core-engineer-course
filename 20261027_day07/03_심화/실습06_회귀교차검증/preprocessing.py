"""제공 모듈: 전날의 전처리를 재사용합니다. 이 파일에는 TODO가 없습니다."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]

def build_model(estimator, scale=False):
    """입력 4열의 전처리와 전달받은 모델을 연결한 새 Pipeline을 반환합니다."""
    # Tree·Forest·XGB에는 스케일링을 생략하고, 전날의 선형·KNN 모델에는 유지합니다.
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler() if scale else "passthrough")])
    category = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", numeric, FEATURES[:2]), ("cat", category, FEATURES[2:])])
    # 아직 학습하지 않았습니다. main.py에서 fit할 때 중앙값·범주·모델이 함께 학습됩니다.
    return Pipeline([("pre", pre), ("model", estimator)])
