"""실습 02 응용: 가상 확률로 임계값만 바꿉니다. 최종 모델의 임계값을 탐색하는 코드가 아닙니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# [외부 라이브러리] scikit-learn의 metrics 모듈: 모델 예측을 채점하는 함수 모음이다.
#   - confusion_matrix: 정답/예측 조합별 개수(TN, FP, FN, TP)를 2x2 표로 센다.
#   - f1_score        : Precision과 Recall의 조화평균을 계산한다.
#   - roc_auc_score   : 확률 점수가 실패(1)를 성공(0)보다 높게 매기는 정도(순위 능력)를 0~1로 계산한다. 임계값을 쓰지 않는다.
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# -----------------------------------------------------------------------------
# 경로 지정
# -----------------------------------------------------------------------------
# 현재의 Python 파일이 존재하는 경로 아래에 outputs 디렉토리를 만드는 경로를 OUT에 저장한다.
OUT = Path(__file__).resolve().parent / "outputs"
# 디렉토리가 이미 존재한다면 무시한다.
OUT.mkdir(exist_ok=True)
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 정답과 확률 준비하기
# -----------------------------------------------------------------------------
# y: 실제 정답이다. 0은 성공, 1은 실패이며 실패(1)가 찾고 싶은 양성 클래스이다.
y = np.array([0, 0, 1, 1])
# score: 모델이 냈다고 가정한 "실패(1)일 확률"이다. 실제 모델이 아니라 설명용 값이다.
score = np.array([0.10, 0.40, 0.35, 0.80])

# -----------------------------------------------------------------------------
# 임계값별 예측과 지표 계산하기
# -----------------------------------------------------------------------------
# rows: 임계값마다 결과 한 줄을 저장하는 빈 리스트이다.
rows = []
# 두 개의 임계값을 하나씩 threshold에 넣고 for문을 수행한다.
for threshold in [0.5, 0.3]:
    # score > threshold : 원소마다 "점수가 기준보다 큰가"를 True/False로 계산한다.
    # 부등호를 >=가 아니라 >로 쓰는 이유: 과제 규칙이 "같은 값은 성공0"이므로, 기준과 같은 점수는 False(0)가 되어야 한다.
    # .astype(int): True/False를 1/0으로 바꾼다. confusion_matrix와 f1_score에 정답 y와 같은 정수 형태로 넘기기 위해서이다.
    # prediction: 현재 임계값에서의 예측이다. 0.5이면 [0,0,0,1], 0.3이면 [0,1,1,1]이 된다.
    prediction = (score > threshold).astype(int)
    # confusion_matrix: [[TN, FP], [FN, TP]] 순서의 혼동행렬이다.
    # labels=[0, 1]: 행·열 순서를 0, 1로 고정한다. 예측에 한쪽 값이 없어도 항상 2x2 모양을 보장한다.
    # .ravel(): 2x2 배열을 [TN, FP, FN, TP] 1차원으로 펼쳐서 네 변수에 순서대로 담는다.
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    # 현재 임계값의 결과를 dict 한 줄로 만들어 rows에 추가한다.
    #   - "threshold"   : 이번에 쓴 임계값이다.
    #   - "FP"          : 성공(0)을 실패(1)로 잘못 경고한 개수이다.
    #   - "FN"          : 진짜 실패(1)를 놓친 개수이다.
    #   - "F1"          : f1_score가 계산한 F1이다. 기본적으로 1을 양성으로 본다.
    #   - "assumed_cost": FP 1개당 비용 1, FN 1개당 비용 5라고 가정한 총비용이다. 실패를 놓치는 쪽이 더 비싸다는 가정이다.
    #   - "AUC"         : 예측(prediction)이 아니라 확률(score)로 계산하므로 임계값과 상관없이 두 줄 모두 같은 값이 나온다.
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, prediction),
                 "assumed_cost": fp + 5*fn, "AUC": roc_auc_score(y, score)})

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# result (type: DataFrame): 임계값 2개 × 열 6개의 비교표이다.
result = pd.DataFrame(rows)
# round(3): 소수점 셋째 자리까지 반올림해 보기 좋게 만든다. index=False: 행 번호(0, 1)를 출력하지 않는다.
print(result.round(3).to_string(index=False))
# 비교표를 outputs/threshold_comparison.csv로 저장한다.
# encoding="utf-8-sig": 엑셀에서 열어도 한글이 깨지지 않도록 BOM을 붙여 저장한다. index=False: 행 번호 열을 저장하지 않는다.
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
