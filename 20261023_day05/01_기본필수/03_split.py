# -----------------------------------------------------------------------------
# 03 기본필수: 20행을 train 12행, valid 4행, test 4행으로 나눕니다.
# -----------------------------------------------------------------------------
# 경로 지정을 위해 파이썬 표준 라이브러리인 Path를 import한다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리로 Pandas를 import한다.
import pandas as pd
# 외부 라이브러리인 scikit-learn의 train_test_split 함수를 import한다.
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 파일 경로 지시
# -----------------------------------------------------------------------------
# 현재 Python 파일의 경로가 어디인지(폴더)를 HERE에 저장한다.
# __file__: 파일 경로
# .resolve(): 절대경로 변환
# .parent: 상위 폴더
HERE = Path(__file__).resolve().parent
# 현재 Python 파일과 같은 경로에 있는 requests.csv를 Pandas로 읽어 data에 저장한다. 
# (type:DataFrame)
data = pd.read_csv(HERE / "requests.csv")

# -----------------------------------------------------------------------------
# 데이터 분할 (Train / Valid / Test)
# -----------------------------------------------------------------------------
# 테스트 데이터 비율을 20%(0.2)로 설정한다. (type: float)
test_ratio = 0.2
# data를 train_valid(80%, 16행)와 test(20%, 4행)로 나누어 각각 저장한다. (type: DataFrame)
# test_size: test에 배정할 비율
# random_state=42: 무작위 분할의 기준값을 고정하여, 실행할 때마다 같은 결과가 나오게 한다.
# stratify=data["failed"]: 실패(1)와 성공(0)의 비율이 
# train_valid와 test 두 데이터에서 비슷하게 유지되도록 나눈다.
train_valid, test = train_test_split(
    data, test_size=test_ratio, random_state=42, stratify=data["failed"])
# 남은 16행의 25%인 4행이 valid이다. 
# train(75%, 12행)과 valid(25%, 4행)로 나누어 각각 저장한다. (type: DataFrame)
# 마찬가지로 stratify를 적용해 failed 비율이 train과 valid에서 비슷하게 유지한다.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# train의 행 개수를 출력한다.
print("train:", len(train))
# valid의 행 개수를 출력한다.
print("valid:", len(valid))
# test의 행 개수를 출력한다.
print("test:", len(test))
# valid의 request_id 열을 리스트로 바꿔 출력한다.
print("valid ID:", valid["request_id"].tolist())
# test는 최종 평가에만 활용하고 학습과 예측에는 사용하지 않는다.
print("test는 오늘 학습과 예측에 사용하지 않습니다.")
