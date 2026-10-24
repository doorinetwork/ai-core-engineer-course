# -----------------------------------------------------------------------------
# 06 기본필수: 숫자 1열과 문자 1열을 함께 변환합니다.
# -----------------------------------------------------------------------------
# 데이터 처리를 위해 외부 라이브러리로 Pandas를 import한다.
import pandas as pd
# 수치형 표준화와 범주형 one-hot 인코딩을 위해
# 외부 라이브러리 scikit-learn의 StandardScaler, OneHotEncoder 클래스를 import한다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 열마다 서로 다른 변환을 한 번에 적용하기 위해
# 외부 라이브러리 scikit-learn의 ColumnTransformer 클래스를 import한다.
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------------------
# 데이터 준비
# -----------------------------------------------------------------------------
# 입력글자수(input_chars, 수치형)와 작업유형(task_type, 범주형) 2개의 열을 가진
# train 데이터를 만든다. (type: DataFrame)
train = pd.DataFrame({"input_chars": [100, 200, 300],
                      "task_type": ["qa", "summary", "qa"]})
# 같은 열을 가진 valid 데이터를 만든다. (type: DataFrame)
valid = pd.DataFrame({"input_chars": [400, 500],
                      "task_type": ["summary", "qa"]})
# ColumnTransformer를 이용해 각 열에 맞는 전처리 기법을 동시에 적용한다: (이름, 변환도구, 적용 열) 순서로 묶음
# 수치형(num)->표준화: input_chars 열에 StandardScaler를 적용
# 범주형(cat)->one-hot 인코딩: task_type 열에 OneHotEncoder를 적용
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
# train으로 input_chars, task_type 각 열의 변환 기준을 학습하여 preprocessor에 저장한다.
preprocessor.fit(train)

# -----------------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------------
# train으로 학습한 기준으로 valid의 두 열(input_chars, task_type)을 동시에 변환하여 
# valid_ready에 저장한다.
valid_ready = preprocessor.transform(valid)

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# 변환 전 valid의 모양을 (행의 개수, 열의 개수) 튜플로 출력한다.
print("변환 전:", valid.shape)
# 변환 후 valid_ready의 모양을 (행의 개수, 열의 개수) 튜플로 출력한다.
# Q: 왜 2열이 3열이 되나요?
# A: 수치형인 input_chars는 그대로 1열이지만, 범주형인 task_type은 범주의 개수대로 
#    one-hot 인코딩을 거쳐 2열이 되기 때문이다. (1 + 2 = 3열)
print("변환 후:", valid_ready.shape)
# valid 변환 결과를 소수 셋째 자리까지 반올림하여 출력한다. 
print(valid_ready.round(3))