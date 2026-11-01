"""실습 07~08: 06번의 설정을 그대로 읽어 개발 16건으로 학습하고 test 4건을 최종 평가합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np
# 결과 기록을 위해 Python 표준 라이브러리 json을 가져온다.
import json
# 분류 모델을 만들기 위해 외부 라이브러리 scikit-learn의 tree에서 DecisionTreeClassifier를 가져온다.
from sklearn.tree import DecisionTreeClassifier
# 평가 지표 계산을 위해 외부 라이브러리 scikit-learn의 지표 함수들을 가져온다.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 06_plan.py에서 생성한 selection.json파일을 읽어 plan에 저장한다.
plan = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다. 
raw = pd.read_csv(HERE / "requests.csv").set_index("request_id")
# train id 목록과 valid id 목록을 리스트 덧셈으로 이어 붙여 그 행들을 골라 develop에 저장한다.
develop = raw.loc[plan["train_ids"] + plan["valid_ids"]].copy()
# test id 목록의 행을 골라 test에 저장한다.
test = raw.loc[plan["test_ids"]].copy()

# -----------------------------------------------------------------------------
# 파생 열 만들기
# -----------------------------------------------------------------------------
# develop과 test를 차례로 꺼내 같은 공식으로 파생 열을 추가한다.
for table in (develop, test):
    # 글자 수에 log(1 + x)를 적용한 새 열 log_input_chars를 추가한다.
    table["log_input_chars"] = np.log1p(table["input_chars"])

# -----------------------------------------------------------------------------
# 모델 만들기와 개발 자료로 학습하기
# -----------------------------------------------------------------------------
# model:
#   max_depth=plan["max_depth"]: 06에서 정한 깊이(3)를 그대로 쓴다. 
#   random_state=plan["random_state"]: 같은 값(42)으로 고정해 다시 실행해도 같은 나무가 만들어진다.
model = DecisionTreeClassifier(max_depth=plan["max_depth"], random_state=plan["random_state"])
# 개발 자료의 선택된 입력 열과 정답 열로 모델을 학습한다.
# - develop[plan["features"]]: 입력(X)이다.
# - develop["failed"]: 정답(y)이다. 0=성공, 1=실패이다.
# test는 절대 넣지 않는다. test로 학습하면 모델이 답을 미리 본 상태가 되어 최종 점수가 실제보다 좋게 나온다.
model.fit(develop[plan["features"]], develop["failed"])

# -----------------------------------------------------------------------------
# test의 실패 확률과 0/1 예측 구하기
# -----------------------------------------------------------------------------
# 학습된 모델이 기억하는 클래스 목록에서 양성 1이 몇 번째 자리인지 찾아 positive_column에 저장한다.
# model.classes_: 학습 정답에 나온 클래스를 정렬한 배열이다. 0과 1이 모두 있으면 [0, 1]이다.
# tolist()로 리스트로 바꾼 뒤 index(1)로 1의 위치를 찾는다.
# positive_column: "두 번째 열이 실패"라고 짐작하지 않고 직접 확인해야, 클래스 순서가 달라져도 실패 확률 열을 정확히 고른다.
positive_column = model.classes_.tolist().index(1)

# test의 선택된 입력 열로 클래스별 확률을 계산하고, 실패(1) 확률 열만 골라 probability에 저장한다.
# model.predict_proba: 각 행마다 [성공 확률, 실패 확률]이 들어 있다.
# test[plan["features"]]: 학습 때와 똑같은 열을 똑같은 순서로 넣어야 모델이 올바르게 계산한다.
# [:, positive_column]: 앞의 : 는 "모든 행", 뒤의 positive_column은 "실패 확률 열 하나"를 뜻한다.
# probability: 0~1 사이 실패 확률이다. 결정 트리에서는 그 행이 도착한 잎(leaf)에 있던 개발 자료 중 실패 비율이다.
probability = model.predict_proba(test[plan["features"]])[:, positive_column]

# 실패 확률이 계획의 기준값(threshold)보다 크면 1, 아니면 0으로 바꿔 prediction에 저장한다.
# prediction: 최종 0/1 판정이다.
# probability > plan["threshold"]는 True/False 배열이고, astype(int)로 True=1, False=0으로 바꾼다.
prediction = (probability > plan["threshold"]).astype(int)

# -----------------------------------------------------------------------------
# 최종 지표 계산
# -----------------------------------------------------------------------------
# 정답과 0/1 예측으로 F1을 계산해 f1에 저장한다.
# f1: 정밀도와 재현율의 조화평균이며, 실패(1)를 얼마나 잘 찾았는지 하나의 숫자로 보여 준다.
# F1은 "맞혔다/틀렸다"로 세는 지표라 확률이 아니라 기준값으로 자른 0/1 예측을 넣는다.
# zero_division=0: 모델이 실패를 하나도 예측하지 않아 정밀도의 분모가 0이 되면 경고 대신 0점으로 처리한다.
f1 = f1_score(test["failed"], prediction, zero_division=0)
# 정답과 실패 확률로 ROC AUC를 계산해 auc에 저장한다.
# auc: 진짜 실패 행의 확률이 진짜 성공 행의 확률보다 높게 매겨진 정도이다. 0.5는 무작위 수준, 1.0은 완벽한 순서이다.
# AUC는 모든 기준값을 한꺼번에 따져 "순서"를 보는 지표라, 0/1로 자르기 전의 확률을 넣어야 한다. 0/1 예측을 넣으면 순서 정보가 사라져 점수가 달라진다.
# 상냥한 한 문장: F1은 "정답을 몇 개 맞혔나"를 세는 점수이고, AUC는 "위험한 순서대로 잘 줄 세웠나"를 보는 점수라서 넣는 재료가 다르다.
auc = roc_auc_score(test["failed"], probability)

# metrics (type: dict):
# - "n" (type: int): test 행 수이다.
# - "Accuracy" (type: float): 전체 test 중 맞힌 비율이다.
# - "Precision" (type: float): 실패라고 예측한 것 중 진짜 실패 비율이다. zero_division=0은 실패 예측이 0건일 때 0으로 처리한다.
# - "Recall" (type: float): 진짜 실패 중 실패라고 찾아낸 비율이다. zero_division=0은 진짜 실패가 0건일 때 0으로 처리한다.
# - "F1", "ROC_AUC" (type: float): 위에서 계산한 값이다.
# - "TP" (type: int): 진짜 실패(1)이면서 실패(1)로 예측한 건수이다. 두 조건을 & 로 묶은 True/False를 sum()으로 세고 int로 바꾼다.
# - "actual_failed" (type: int): test에 있는 진짜 실패 건수이다. Recall의 분모이다.
# - "threshold" (type: float): 이번 판정에 쓴 기준값이다. 어떤 기준으로 채점했는지 결과와 함께 남긴다.
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1, "ROC_AUC": auc, "TP": int(((test["failed"] == 1) & (prediction == 1)).sum()),
           "actual_failed": int(test["failed"].sum()), "threshold": plan["threshold"]}

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# test 행별 정답, 실패 확률, 0/1 예측을 한 표로 묶어 result에 저장한다.
# result: request_id, actual, probability_failed, predicted 열을 가진다.
# test.index는 request_id 값이고, to_numpy()로 정답 열을 순수 배열로 바꿔 행 순서대로 넣는다. probability와 prediction도 test와 같은 행 순서이다.
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
# result를 test_predictions.csv로 저장한다.
result.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8-sig")
# metrics를 보기 좋은 JSON 글자로 바꿔 test_metrics.json으로 저장한다.
(HERE / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# 개발 자료 건수와 최종 평가 건수를 출력한다.
print("개발/최종평가:", len(develop), len(test))
# 행별 예측 표를 소수점 셋째 자리로 반올림해 행 번호 없이 출력한다.
print(result.round(3).to_string(index=False))
# 혼동행렬을 리스트로 출력한다.
print("혼동행렬:", confusion_matrix(test["failed"], prediction, labels=[0, 1]).tolist())
# 다섯 지표를 소수점 셋째 자리까지 한 줄로 출력한다. :.3f는 소수점 아래 3자리로 맞추는 형식이다.
print(f"Accuracy={metrics['Accuracy']:.3f}, Precision={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
print("저장: test_predictions.csv, test_metrics.json")
print("08번은 이 설정 그대로 재실행합니다. test를 보고 06번 설정을 다시 고르지 않습니다.")
