"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 결과 기록을 위해 Python 표준 라이브러리 json을 가져온다.
import json
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np
# 열마다 다른 전처리를 붙이기 위해 외부 라이브러리 scikit-learn의 ColumnTransformer를 가져온다.
from sklearn.compose import ColumnTransformer
# 결측치 처리를 위해 외부 라이브러리 scikit-learn의 SimpleImputer를 가져온다.
from sklearn.impute import SimpleImputer
# 숫자 표준화와 범주 인코딩을 위해 외부 라이브러리 scikit-learn의 StandardScaler, OneHotEncoder를 가져온다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 전처리와 모델을 하나로 묶기 위해 외부 라이브러리 scikit-learn의 Pipeline을 가져온다.
from sklearn.pipeline import Pipeline
# 분류 모델로 외부 라이브러리 scikit-learn의 LogisticRegression을 가져온다.
from sklearn.linear_model import LogisticRegression
# 평가 지표 계산을 위해 외부 라이브러리 scikit-learn의 지표 함수들을 가져온다.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# 경로 지정
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# -----------------------------------------------------------------------------
# 평가 계획 읽기
# -----------------------------------------------------------------------------
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")

# -----------------------------------------------------------------------------
# 개발80과 test20 나누기
# -----------------------------------------------------------------------------
# 계획에 기록된 분할별 id 목록을 ids에 저장한다.
# ids (type: dict[str, list[str]]): 키는 "train", "valid", "test"이고 값은 각 분할의 request_id 목록(60개, 20개, 20개)이다.
ids = plan["split_ids"]
# train60 id 목록과 valid20 id 목록을 리스트 덧셈으로 이어 붙여 그 행들을 골라 develop에 저장한다.
# develop (type: DataFrame): 개발80 데이터(80행)이다. 선택이 끝났으므로 valid도 더 이상 비교용으로 남겨 둘 필요가 없어 학습 데이터에 합친다.
develop = raw.loc[ids["train"] + ids["valid"]]
# test20 id 목록의 행을 골라 test에 저장한다.
# test (type: DataFrame): 최종 평가용 20행이다. 지금까지 어떤 학습·선택에도 쓰지 않은 데이터이다.
test = raw.loc[ids["test"]]
# 계획에 적힌 입력 열 4개만 골라 복사본을 만들어 X_develop와 X_test에 저장한다.
# X_develop (type: DataFrame): 80행 × 4열의 학습 입력이다. X_test (type: DataFrame): 20행 × 4열의 평가 입력이다.
# copy()를 하는 이유: 아래에서 파생 열을 추가할 때 원본 raw를 건드리지 않고 경고 없이 독립된 표에 쓰기 위해서이다.
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
# 입력 열 목록의 앞 2개를 숫자 열 목록 NUM에, 나머지를 범주 열 목록 CAT에 저장한다.
# NUM (type: list[str]): ["input_chars", "allowed_tools"]이다. CAT (type: list[str]): ["task_type", "priority"]이다.
NUM, CAT = plan["features"][:2], plan["features"][2:]

# -----------------------------------------------------------------------------
# 파생 특성 만들기 (선택 모델이 Logistic_FE일 때)
# -----------------------------------------------------------------------------
# 계획의 derived가 True(선택 모델이 Logistic_FE)일 때만 파생 특성 2개를 만든다.
if plan["derived"]:
    # 개발 입력과 평가 입력에 똑같은 공식을 적용하기 위해 두 표를 차례로 part에 담아 반복한다.
    for part in (X_develop, X_test):
        # 입력 글자 수에 log(1+x)를 적용해 log_input_chars 열을 만든다.
        part["log_input_chars"] = np.log1p(part["input_chars"])
        # 입력 글자 수를 (1 + 허용 도구 수)로 나눠 chars_per_slot 열을 만든다.
        # 도구 하나당 처리해야 할 글자 수라는 뜻의 비율 특성이며, 분모에 1을 더해 도구가 0개일 때 0으로 나누는 일을 막는다.
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    # 숫자 열 목록에 새로 만든 파생 열 2개를 더해 NUM을 새 리스트로 바꾼다.
    # NUM (type: list[str]): ["input_chars", "allowed_tools", "log_input_chars", "chars_per_slot"]이 된다.
    NUM = NUM + ["log_input_chars", "chars_per_slot"]

# -----------------------------------------------------------------------------
# 전처리와 모델 Pipeline 만들기
# -----------------------------------------------------------------------------
# 숫자 열과 범주 열에 각각 다른 전처리 파이프라인을 붙인 ColumnTransformer를 만들어 pre에 저장한다.
# 숫자 열(NUM): 빈칸을 중앙값으로 채운 뒤 표준화한다.
#   strategy="median"은 극단값의 영향을 덜 받는 중앙값을 쓰게 하고, 
#   keep_empty_features=True는 값이 전부 비어 있는 열이 생겨도 열을 버리지 않아 열 개수가 바뀌지 않게 한다.
# 범주 열(CAT): 빈칸을 가장 많이 나온 값으로 채운 뒤 one-hot 인코딩한다.
#   handle_unknown="ignore"는 학습 때 없던 범주가 test에 나오면 오류 대신 모든 0/1 열을 0으로 두게 하고, 
#   sparse_output=False는 결과를 일반 배열로 돌려준다.
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])
# 전처리기 pre와 로지스틱 회귀를 순서대로 묶은 Pipeline을 만들어 model에 저장한다.
# C는 규제 강도의 역수(작을수록 가중치를 강하게 눌러 단순한 모델이 됨)이고, max_iter는 최적화 반복의 최대 횟수이다.
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])

# -----------------------------------------------------------------------------
# 개발80으로 최종 재학습
# -----------------------------------------------------------------------------
# model.fit은 Pipeline의 단계를 차례로 학습한다: 
#   먼저 개발80의 입력으로 중앙값·최빈값·평균·표준편차·범주 목록을 계산하고, 변환된 입력과 정답 develop["failed"]로 로지스틱 회귀의 가중치를 학습한다.
#   새로 fit하는 이유: train60으로 학습한 예전 모델을 재사용하면 valid20의 정보가 빠지므로, 
#                    선택이 끝난 지금은 쓸 수 있는 데이터 80행을 모두 써서 더 많이 배운 최종 모델을 만든다.
#   X를 X_develop, y를 develop["failed"]로 주며, test는 여기에 절대 넣지 않는다.
model.fit(X_develop, develop["failed"])

# -----------------------------------------------------------------------------
# test20 확률 예측
# -----------------------------------------------------------------------------
# 학습된 모델이 기억하는 클래스 목록에서 1(실패)이 몇 번째 열인지 찾아 positive_column에 저장한다.
# model.classes_: [0, 1]이며, positive_column은 1이 된다. 
# 확률 열의 순서를 추측하지 않고 실재하는 실패 확률 열을 찾는다.
positive_column = model.classes_.tolist().index(1)
# model.predict_proba(X_test)는 개발80에서 배운 통계로 X_test를 변환(transform만 수행)한 뒤 각 행의 [성공 확률, 실패 확률] 2열 배열을 돌려준다.
#   [:, positive_column]은 모든 행(:)에서 실패 확률 열만 꺼내 1차원 배열로 만든다.
#   probability: test 요청 20건 각각의 실패 확률(0~1)이다. 임계값 비교와 ROC-AUC 계산에 함께 쓴다.
#   predict 대신 predict_proba를 쓰는 이유: 계획에 고정한 임계값 규칙(score > 0.5)을 아래에서 직접 적용하고, 
#                                        순위 지표인 ROC-AUC를 계산하려면 확률이 필요하기 때문이다.
probability = model.predict_proba(X_test)[:, positive_column]

# -----------------------------------------------------------------------------
# 임계값 적용과 최종 지표 계산
# -----------------------------------------------------------------------------
# 실패 확률이 계획의 임계값(0.5)보다 큰 행만 True로 만든 뒤 정수 1/0으로 바꿔 prediction에 저장한다.
# prediction: 최종 예측(1=실패, 0=성공)이다. 
#             ">"를 쓰는 이유는 계획의 rule "score > threshold"와 같게 해, 정확히 0.5이면 실패로 보지 않기 위해서이다.
prediction = (probability > plan["threshold"]).astype(int)
# 정답과 예측으로 혼동행렬을 만들고 ravel()로 펼쳐 네 개수를 tn, fp, fn, tp에 저장한다.
# tn, fp, fn, tp: 각각 성공을 성공으로, 성공을 실패로, 실패를 성공으로, 실패를 실패로 예측한 개수이다.
# labels=[0, 1]은 한쪽 클래스가 예측에 하나도 없어도 항상 2×2 표가 되게 해서 네 값의 순서를 고정한다.
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
# 최종 지표를 dict로 모아 metrics에 저장한다.
# metrics (type: dict): 
#                       n(평가 행 수 20), 
#                       Accuracy(전체 정답 비율), 
#                       Precision(실패라고 한 것 중 진짜 실패 비율), 
#                       Recall(진짜 실패 중 찾아낸 비율), 
#                       F1(정밀도와 재현율의 조화평균), 
#                       ROC_AUC(확률 순위가 실패를 성공보다 위에 두는 정도), 
#                       TN·FP·FN·TP(혼동행렬 칸), 
#                       actual_failed(test의 실제 실패 수)를 가진다.
#   zero_division=0은 실패 예측이 하나도 없어 분모가 0이 될 때 경고 대신 0을 쓰게 한다.
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(test["failed"].sum())}

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# 요청 id, 실제 정답, 실패 확률, 최종 예측을 한 표로 묶어 result에 저장한다.
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
# result를 outputs/test_predictions.csv로 저장한다.
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
# metrics를 JSON 문자열로 바꿔 outputs/test_metrics.json에 UTF-8로 저장한다.
# ensure_ascii=False는 비ASCII 글자를 그대로 쓰게 하고, indent=2는 2칸 들여쓰기로 사람이 읽기 쉽게 한다.
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# 확정 모델 이름과 학습·평가 데이터 구성을 화면에 출력한다.
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
# metrics를 한 행짜리 표로 바꾸고 소수 넷째 자리로 반올림해 행 번호 없이 출력한다.
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
# 실제 정답과 예측이 다른 행만 골라 앞의 3개를 소수 셋째 자리로 반올림해 출력한다.
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")
