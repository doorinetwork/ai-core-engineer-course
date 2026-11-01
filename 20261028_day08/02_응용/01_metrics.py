"""실습 01 응용: 모델 예시와 전부 성공으로 예측하는 기준선을 비교합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# [외부 라이브러리] scikit-learn의 metrics 모듈: 모델 예측을 채점하는 함수 모음이다.
#   - confusion_matrix: 정답/예측 조합별 개수(TN, FP, FN, TP)를 2x2 표로 센다.
#   - accuracy_score  : 전체 중 맞힌 비율을 계산한다.
#   - precision_score : "실패라고 예측한 것" 중 진짜 실패의 비율을 계산한다.
#   - recall_score    : "진짜 실패" 중 실패라고 찾아낸 비율을 계산한다.
#   - f1_score        : Precision과 Recall의 조화평균을 계산한다. 이 코드에서는 직접 계산한 F1과 대조하는 확인용으로만 쓴다.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# 정답과 예측 후보 준비하기
# -----------------------------------------------------------------------------
# 실제 정답 레이블 정의 (총 20건: 성공 0이 14건, 관심 클래스인 실패 1이 6건)
y = [0]*14 + [1]*6
# 비교 대상 예측 결과 정의
# predictions (type: dict[str, list[int]]): 후보 이름 → 20개 예측값을 담는다.
#   - "example"    : TP=2, FP=4, FN=4, TN=10을 나타내는 모델 예시
#                    앞 14개(정답 0) 중 10개는 0, 4개는 1로 예측한다 → TN 10, FP 4.
#                    뒤 6개(정답 1) 중 4개는 0, 2개는 1로 예측한다   → FN 4, TP 2.
#   - "all_success": 20개 모두 0(성공)으로 예측하는 기준선(baseline)이다. 아무것도 배우지 않은 모델의 최소 기준 역할을 한다.
predictions = {"example": [0]*10 + [1]*4 + [0]*4 + [1]*2, "all_success": [0]*20}

# -----------------------------------------------------------------------------
# 후보별 지표 계산하기
# -----------------------------------------------------------------------------
# rows: 후보마다 지표 한 줄(dict)을 쌓아 두는 빈 리스트이다.
rows = []

# name: 후보 이름이다. pred: 그 후보의 예측 20개이다.
for name, pred in predictions.items():
    # confusion_matrix: [[TN, FP], [FN, TP]] 순서의 혼동행렬이다.
    # labels=[0, 1]로 고정하여 혼동행렬을 풀기 (TN, FP, FN, TP 순서)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # f1_score를 측정해 f1에 저장한다. 예측된 양성(1)이 없어 분모가 0이 될 경우 0.0으로 기록하도록 zero_division=0을 설정합니다.
    f1 = f1_score(y, pred, zero_division=0)
    # 현재 후보의 결과 한 줄을 rows에 추가한다.
    # zero_division=0: 분모가 0일 때 경고 대신 0을 돌려주게 한다.
    #                  all_success는 1을 한 번도 예측하지 않아 Precision 분모(TP+FP)가 0이므로 이 인자가 필요하다.
    rows.append({"model": name, "Accuracy": accuracy_score(y, pred),
                 "Precision": precision_score(y, pred, zero_division=0),
                 "Recall": recall_score(y, pred, zero_division=0), "F1": f1,
                 "FP": fp, "FN": fn, "TP": tp})

# -----------------------------------------------------------------------------
# 결과 출력
# -----------------------------------------------------------------------------
# pd.DataFrame(rows) (type: DataFrame): dict 리스트를 행=후보, 열=지표인 표로 바꾼다.
# .round(3): 소수점 셋째 자리까지 반올림한다.
# .to_string(index=False): 왼쪽 행 번호(0, 1)를 빼고 표 전체를 문자열로 만든다.    
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("정확도가 높은 all_success도 실패 6건을 하나도 찾지 못합니다.")
