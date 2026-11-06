"""실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
# ============================================================
# 1. 라이브러리 불러오기와 기록 읽기
# ============================================================
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
# metrics_example.csv : 읽기 연습용 가상 학습 기록 (epoch 1~5)
#   epoch     : 학습 반복 회차
#   map50     : IoU 0.5 기준 mean AP
#   map50_95  : IoU 0.50~0.95(0.05 간격) 10개 기준 AP의 평균
#   precision : 예측 중 맞은 비율 / recall : 정답 중 찾은 비율
# 주의: 실제 YOLO 학습 기록이 아니다.
history = pd.read_csv(BASE / "data/metrics_example.csv")

# ============================================================
# 2. map50_95 최대 행 선택 (TODO B06)
# ============================================================
# >>> B06
# 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
# 메서드 체인을 한 단계씩 나누어 읽는다.
#   history["map50_95"] : map50_95 열(Series) -> [0.08, 0.15, 0.24, 0.27, 0.25]
#   idxmax()            : 최대값 0.27이 처음 나오는 행 index -> 3 (epoch 4)
#   history.loc[3]      : 그 행 전체(epoch·map50·map50_95·precision·recall)
# 주의: 최대값이 여러 행이면 먼저 나온 행을 고른다. 마지막 epoch(5, 0.25)가 최고라는 뜻이 아님
best = history.loc[
    history["map50_95"]
    .idxmax()
]
# <<< B06

# ============================================================
# 3. 선택 결과 출력
# ============================================================
print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")
print("선택 epoch:", int(best["epoch"]))
print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])
print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")
