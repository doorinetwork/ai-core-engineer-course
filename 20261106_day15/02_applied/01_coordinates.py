"""실습 01. 픽셀 xyxy와 YOLO 정규화 중심·크기를 왕복합니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 2. 경로·입력 준비
# ============================================================
# BASE : 이 파일 폴더 / out : 결과 폴더 outputs/ (없으면 생성)
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
# boxes_gt.csv : 정답 박스 3개 (픽셀 xyxy)
table=pd.read_csv(BASE/"data/boxes_gt.csv")
# b : (3, 4) float 배열 [[12,18,52,58], [70,60,108,98], [116,18,150,62]]
# width, height : 이미지 크기 W=160, H=128
b=table[["x1","y1","x2","y2"]].to_numpy(float);width,height=160,128

# ============================================================
# 3. xyxy -> YOLO 정규화 cxcywh (TODO A01)
# ============================================================
# >>> A01
# 요구: 중심 x/y와 폭/높이를 계산하고 각 축 크기로 정규화하세요.
# np.column_stack((열1, 열2, 열3, 열4)) : 길이 3짜리 1차원 배열 4개를 (3, 4) 배열의 열로 묶음
#   cx = (x1 + x2) / 2 / W   : 중심 x를 너비 160으로 나눔
#   cy = (y1 + y2) / 2 / H   : 중심 y를 높이 128로 나눔
#   w  = (x2 - x1) / W       : 폭을 너비로 나눔
#   h  = (y2 - y1) / H       : 높이를 높이로 나눔
# 예: [12,18,52,58] -> 중심 (32, 38), 크기 40×40 -> [0.2, 0.296875, 0.25, 0.3125]
# 주의: x는 W, y는 H로 나눈다. 160×128처럼 정사각형이 아니면 바꾸어 나눌 때 값이 틀어진다.
# 주의: 연속 경계 [x1, x2) 정의라 폭에 +1을 하지 않는다.
yolo=np.column_stack((
    (b[:,0]+b[:,2])/2/width,
    (b[:,1]+b[:,3])/2/height,
    (b[:,2]-b[:,0])/width,
    (b[:,3]-b[:,1])/height,
))
# <<< A01

# ============================================================
# 4. 다시 픽셀 xyxy로 복원해 왕복 오차 확인
# ============================================================
# pixel : 정규화 값에 [W, H, W, H]를 곱해 픽셀 cx, cy, w, h로 되돌림
pixel=yolo*np.array([width,height,width,height])
# restored : 중심 ± 크기/2 -> x1, y1, x2, y2
restored=np.column_stack((pixel[:,0]-pixel[:,2]/2,pixel[:,1]-pixel[:,3]/2,pixel[:,0]+pixel[:,2]/2,pixel[:,1]+pixel[:,3]/2))
# np.max(np.abs(b - restored)) : 원본과 복원 좌표 차이의 최대값 (px). 기대 0.0
print("최대 왕복 오차(px):", float(np.max(np.abs(b-restored))))
# yolo_coordinates.csv : 박스별 cx, cy, w, h (0~1)
pd.DataFrame(yolo,columns=["cx","cy","w","h"]).to_csv(out/"yolo_coordinates.csv",index=False)
