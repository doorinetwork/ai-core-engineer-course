"""실습 02. 손으로 정한 예측 4개를 정답 3개와 비교합니다. YOLO 결과가 아닙니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 2. 입력 읽기 (가상 예측 4개·정답 3개)
# ============================================================
BASE = Path(__file__).resolve().parent
# boxes_fixture.csv : 손으로 정한 예측 4개 (confidence 내림차순 0.95 -> 0.85 -> 0.80 -> 0.40)
# 주의: 원리 연습값이며 YOLO 예측이 아니다.
# confidence 내림차순 예시
pred = pd.read_csv(BASE / "data/boxes_fixture.csv")
# boxes_gt.csv : 정답 3개 (square·circle·triangle 각 1개, confidence 없음)
truth = pd.read_csv(BASE / "data/boxes_gt.csv")
# used : 이미 TP로 연결된 정답 행 번호 모음 -> 같은 정답을 두 번 TP로 세지 않기 위함
# 하나의 정답은 한 번만 TP로 연결합니다.
used = set()
# rows : 예측별 [class, confidence, max_iou, TP/FP] 기록
rows = []

# ============================================================
# 3. 예측마다 IoU 계산과 일대일 매칭
# ============================================================
# pred는 confidence 내림차순이므로 점수가 높은 예측이 먼저 정답을 차지한다.
for p in pred.itertuples():
    # --------------------------------------------------------
    # 예측 1개 vs 정답 3개의 IoU 계산
    # --------------------------------------------------------
    # overlaps : 이 예측과 정답 3개 각각의 IoU 목록 (정답 순서와 같음)
    overlaps = []
    for g in truth.itertuples():
        # 겹친 영역의 폭·높이 = (작은 오른쪽 끝 - 큰 왼쪽 끝), 안 겹치면 음수 -> max(0, ...)로 0
        width = max(0, min(p.x2, g.x2) - max(p.x1, g.x1))
        height = max(0, min(p.y2, g.y2) - max(p.y1, g.y1))
        # 교집합 넓이
        intersection = width * height
        # 합집합 넓이 = 예측 넓이 + 정답 넓이 - 교집합 (교집합을 두 번 세지 않음)
        union = (p.x2-p.x1)*(p.y2-p.y1) + (g.x2-g.x1)*(g.y2-g.y1) - intersection
        # >>> B02
        # 요구: intersection을 union으로 나누어 iou를 계산하세요.
        # 이 실습의 데이터는 모두 양의 면적이므로 union은 0이 아닙니다.
        # IoU = 교집합 / 합집합, 범위 0(안 겹침) ~ 1(완전히 같음)
        # 예: 예측 (14,20,52,58) vs 정답 (12,18,52,58)
        #     -> 교집합 38*38=1444, 합집합 1444+1600-1444=1600 -> IoU 0.9025
        iou = intersection / union
        # <<< B02
        overlaps.append(iou)

    # --------------------------------------------------------
    # IoU가 큰 정답부터 확인해 하나만 연결
    # --------------------------------------------------------
    # matched : 이 예측이 TP가 되었는지 표시
    matched = False
    # np.argsort(-배열) : 값이 큰 순서의 index 목록. 예: [0.90, 0.0, 0.0] -> [0, 1, 2]
    for j in np.argsort(-np.asarray(overlaps)):
        # 다른 클래스 정답이거나 이미 다른 예측이 차지한 정답이면 건너뜀
        if truth.iloc[j].class_id != p.class_id or j in used:
            continue
        # >>> B03
        # 요구: 같은 클래스의 미사용 정답에 IoU>=0.5이면 연결하고 used에 기록하세요.
        # overlaps[j] >= 0.5 : 위치 기준 통과 -> 정답 j를 사용 처리하고 TP로 표시
        # break : 예측 1개는 정답 1개에만 연결 (일대일)
        # 주의: IoU 최대인 같은 클래스 정답이 0.5 미만이면 더 작은 IoU도 미달이므로 FP로 남는다.
        if overlaps[j] >= 0.5:
            used.add(j)
            matched = True
            break
        # <<< B03
    # max_iou : 클래스와 무관하게 가장 많이 겹친 정답과의 IoU (설명용)
    rows.append([p.class_id, p.confidence, round(max(overlaps), 4), "TP" if matched else "FP"])

# ============================================================
# 4. TP/FP/FN·Precision/Recall 집계와 저장
# ============================================================
table = pd.DataFrame(rows, columns=["class", "confidence", "max_iou", "result"])
print(table.to_string(index=False))
# tp : 연결된 정답 수 / fp : TP가 아닌 예측 수 / fn : 아무 예측과도 연결되지 않은 정답 수
tp = len(used); fp = len(pred)-tp; fn = len(truth)-tp
print("TP/FP/FN:", tp, fp, fn)
# Precision = TP / 전체 예측 수, Recall = TP / 전체 정답 수 -> 2/4 = 0.5, 2/3 = 0.6667
print("Precision/Recall:", round(tp/len(pred), 4), round(tp/len(truth), 4))
# encoding="utf-8-sig" : 엑셀에서 열어도 한글·BOM이 깨지지 않도록 저장
table.to_csv(BASE / "p02_matching.csv", index=False, encoding="utf-8-sig")
