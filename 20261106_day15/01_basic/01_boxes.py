"""실습 01. 정답 박스 3개를 그립니다. 원본 scene.png는 바꾸지 않습니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import cv2
import pandas as pd

# ============================================================
# 2. 입력 읽기 (이미지·정답 박스 표)
# ============================================================
# BASE : 이 .py 파일이 있는 폴더 (어느 폴더에서 실행해도 입력 위치가 같음)
BASE = Path(__file__).resolve().parent
# cv2.imread(경로) : PNG를 BGR 순서의 NumPy 배열로 읽음 -> shape (H=128, W=160, C=3)
image = cv2.imread(str(BASE / "data/scene.png"))
# boxes_gt.csv : 정답(GT) 박스 표. 열 = image_id, class_id, x1, y1, x2, y2 (confidence 열 없음)
# 예: scene, 0, 12, 18, 52, 58 -> square 박스 왼쪽 위 (12,18) ~ 오른쪽 아래 (52,58)
boxes = pd.read_csv(BASE / "data/boxes_gt.csv")
# image.copy() : 그림용 배열을 따로 만듦 -> 박스를 그려도 원본 image 배열은 그대로
# 그릴 배열을 복사하여 원본을 보존합니다.
result = image.copy()

# ============================================================
# 3. 정답 박스 그리기 (TODO B01)
# ============================================================
# itertuples() : 표의 한 행을 row.x1, row.class_id처럼 속성으로 읽는 묶음으로 돌려줌
for row in boxes.itertuples():
    # >>> B01
    # 요구: result에 (x1,y1)~(x2,y2) 박스를 두께 2로 그리세요.
    # cv2.rectangle의 이미지·두 점·색은 필수, thickness는 선택입니다.
    # 해설: 파일의 좌표는 연속 경계이고 화면 표시 때만 정수로 바꿉니다.
    # cv2.rectangle(): 이미지 위에 Bounding Box(사각형)를 그리는 함수
    #
    # result : 사각형을 그릴 대상 이미지 (원본 배열이 직접 수정됨)
    #
    # (int(row.x1), int(row.y1))
    #   → Bounding Box의 왼쪽 위 좌표 (x1, y1)
    # (int(row.x2), int(row.y2))
    #   → Bounding Box의 오른쪽 아래 좌표 (x2, y2)
    # int(): 픽셀 좌표는 정수로 전달해야 하므로 실수 좌표를 정수로 변환
    #
    # (255, 0, 0): 사각형 테두리 색상
    #   → OpenCV는 BGR 순서를 사용하므로 파란색(Blue)
    #
    # 2: 사각형 테두리 두께 (2픽셀)
    #
    # 목적: 객체 탐지 결과의 위치를 이미지에 표시하여
    #       모델이 어떤 영역을 객체로 인식했는지 시각적으로 확인

    cv2.rectangle(
        result,
        (int(row.x1), int(row.y1)),
        (int(row.x2), int(row.y2)),
        (255, 0, 0),
        2
    )
    # <<< B01
    # GT에는 confidence가 없으므로 N/A로 출력 (모델 예측만 confidence를 가짐)
    print("class:", row.class_id, "xyxy:", row.x1, row.y1, row.x2, row.y2, "GT confidence: N/A")

# ============================================================
# 4. 결과 저장
# ============================================================
# cv2.imwrite(경로, 배열) : 박스를 그린 복사본을 p01_boxes.png로 저장 (원본 scene.png는 덮어쓰지 않음)
cv2.imwrite(str(BASE / "p01_boxes.png"), result)
# image.shape -> (128, 160, 3) = (높이, 너비, 채널)
print("shape:", image.shape, "저장: p01_boxes.png")
