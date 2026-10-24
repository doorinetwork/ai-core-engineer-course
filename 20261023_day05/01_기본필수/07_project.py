# -----------------------------------------------------------------------------
# 07~08 기본필수: 이 파일 하나를 완성하고 같은 파일로 재실행합니다.
# -----------------------------------------------------------------------------
# 경로 지정을 위해 파이썬 표준 라이브러리인 Path를 import한다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리로 Pandas를 import한다.
import pandas as pd
# 데이터를 무작위로 나누기 위해 외부 라이브러리 scikit-learn의 train_test_split 함수를 import한다.
from sklearn.model_selection import train_test_split
# 수치형 표준화와 범주형 one-hot 인코딩을 위해
# 외부 라이브러리 scikit-learn의 StandardScaler, OneHotEncoder 클래스를 import한다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 열마다 서로 다른 변환을 한 번에 적용하기 위해
# 외부 라이브러리 scikit-learn의 ColumnTransformer 클래스를 import한다.
from sklearn.compose import ColumnTransformer
# 전처리와 모델을 하나의 순서로 묶기 위해
# 외부 라이브러리 scikit-learn의 Pipeline 클래스를 import한다.
from sklearn.pipeline import Pipeline
# 성능 비교를 위한 기준점(Baseline) 모델로 DummyClassifier를 import한다.
from sklearn.dummy import DummyClassifier

# -----------------------------------------------------------------------------
# 파일 경로 지시
# -----------------------------------------------------------------------------
# 현재 Python 파일이 들어 있는 폴더의 절대경로를 HERE에 저장한다.
HERE = Path(__file__).resolve().parent
# 같은 폴더에 있는 requests.csv를 Pandas로 읽어 data에 저장한다. (type: DataFrame)
data = pd.read_csv(HERE / "requests.csv")

# -----------------------------------------------------------------------------
# 데이터 분할 (Train / Valid / Test)
# -----------------------------------------------------------------------------
# data를 train_valid(80%)와 test(20%)로 나누어 각각 저장한다. (type: DataFrame)
# random_state=42: 실행할 때마다 같은 결과가 나오도록 분할 기준을 고정한다.
# stratify=data["failed"]: 실패(1)와 성공(0)의 비율이 두 데이터에서 비슷하게 유지되도록 나눈다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
# train_valid를 train(75%)과 valid(25%)로 나누어 각각 저장한다. (type: DataFrame)
# 남은 80%의 25%이므로 전체 기준으로는 train 60%, valid 20%, test 20%이다.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test 데이터는 모델 평가의 공정성을 위해 개수 확인 외에 오늘 학습과 예측에 사용하지 않는다.

# -----------------------------------------------------------------------------
# 입력과 정답 준비
# -----------------------------------------------------------------------------
# 입력으로 사용할 열 이름 2개를 리스트로 저장한다. (type: list)
features = ["input_chars", "task_type"]
# train에서 입력 열 2개를 골라 X_train에 저장한다. (type: DataFrame)
X_train = train[features]
# train에서 정답(failed) 열 1개를 골라 y_train에 저장한다. (type: Series)
y_train = train["failed"]
# valid에서 입력 열 2개를 골라 X_valid에 저장한다. (type: DataFrame)
X_valid = valid[features]

# -----------------------------------------------------------------------------
# 전처리기 설정
# -----------------------------------------------------------------------------
# ColumnTransformer를 이용해 각 열에 맞는 전처리 기법을 동시에 적용한다: (이름, 변환도구, 적용 열) 순서로 묶음
# 수치형(num)->표준화: input_chars 열에 StandardScaler를 적용
# 범주형(cat)->one-hot 인코딩: task_type 열에 OneHotEncoder를 적용
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# -----------------------------------------------------------------------------
# pipeline 구축
# -----------------------------------------------------------------------------
# 전처리(pre)와 모델(model)을 순서대로 묶은 Pipeline 객체를 만들어 pipeline에 저장한다.(type: Pipeline)
# DummyClassifier는 train에서 가장 많은 정답만 반복하는 비교 출발점(baseline)이다.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
# pipeline으로 전처리 기준과 모델을 한 번에 학습(fit)한다. 학습에 valid와 test를 사용하지 않는다.
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
# 학습된 파이프라인을 이용해 검증 데이터(X_valid)의 결과를 예측하여 prediction에 저장한다.
# pipeline 내부에서 전처리기가 X_valid를 transform만 하고,
# 이 변환된 결과를 가지고 모델이 predict한다.
prediction = pipeline.predict(X_valid)

# -----------------------------------------------------------------------------
# 결과 저장
# -----------------------------------------------------------------------------
# 예측 결과를 원래의 ID·정답과 나란히 놓고, 이 파일 옆에 CSV 파일로 저장한다.
# valid에서 ID(request_id)와 정답(failed) 열을 복사해 result에 저장한다. (type: DataFrame)
# .copy()는 원본 valid가 바뀌지 않도록 별도 사본을 만든다.
result = valid[["request_id", "failed"]].copy()
# 예측값을 predicted 열로 추가하여 실제 정답과 나란히 비교할 수 있게 한다.
result["predicted"] = prediction
# result를 이 파일과 같은 폴더에 valid_predictions.csv로 저장한다.
# index=False: 행 번호를 나타내는 열은 저장하지 않는다.
# encoding="utf-8-sig": utf-8-sig 인코딩을 사용해 저장한다. 
result.to_csv(HERE / "valid_predictions.csv", index=False, encoding="utf-8-sig")
# 실제 정답과 예측이 같은 행의 비율을 계산해 accuracy에 저장한다. (type: float)
accuracy = (result["failed"] == result["predicted"]).mean()

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# train, valid, test의 행 개수를 차례로 출력한다.
print("분할:", len(train), len(valid), len(test))
# 요청 ID, 실제 정답, 예측값을 행 번호 없이 표 형태로 출력한다.
print(result.to_string(index=False))
# valid 정확도를 소수 둘째 자리까지 반올림하여 출력한다.
print("valid 정확도:", round(accuracy, 2))
# test는 마지막 최종 평가를 위해 사용하지 않는다.
print("저장: valid_predictions.csv / test는 사용하지 않았습니다.")