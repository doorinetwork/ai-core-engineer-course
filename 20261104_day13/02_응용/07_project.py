from pathlib import Path
import os
import cv2
import numpy as np

# ============================================================
# 1. 작업 경로 설정 및 출력 폴더 준비
# ============================================================
# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
# 현재 .py 파일이 있는 폴더로 작업 경로를 이동
os.chdir(Path(__file__).resolve().parent)

import csv
import json
from time import perf_counter
# 결과 저장 폴더 생성 (이미 있으면 그대로 사용)
Path("outputs").mkdir(exist_ok=True)

# ============================================================
# 2. 이미지 처리 함수
# ============================================================
def process_image(image):
    """원본 배열 → Gray·Resize·Edge. 파일 읽기는 호출한 쪽에서 합니다.

    - image: BGR uint8 배열 (이미지 파일 또는 영상 frame)
    - 반환 gray: 원본 크기 1채널 Gray
    - 반환 small: 160×120으로 줄인 Gray
    - 반환 edge: small을 흐린 뒤 Canny로 만든 0/255 Edge
    """
    # >>> TODO A07
    # Gray→160×120 Resize→3×3 Gaussian→Canny 80/160을 구현하세요. 반환 변수 gray·small·edge를 만듭니다.

    # --------------------------------------------------------
    # 1단계: BGR -> Gray
    # --------------------------------------------------------
    # B·G·R 가중합으로 밝기 1채널 생성
    # 예: (240, 320, 3) -> (240, 320)
    # 주의: image[:, :, 0]은 B 채널만 꺼낸 것이라 Gray가 아님
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # 2단계: 160×120 Resize
    # --------------------------------------------------------
    # dsize=(160, 120) : (너비, 높이). 결과 shape는 (120, 160)
    small = cv2.resize(gray, (160, 120))

    # --------------------------------------------------------
    # 3단계: 3×3 Gaussian -> 4단계: Canny 80/160
    # --------------------------------------------------------
    # ksize=(3, 3) : 작은 커널로 잡음만 살짝 줄임
    # sigmaX=0     : 표준편차 자동 계산
    blurred = cv2.GaussianBlur(small, (3, 3), 0)

    # threshold1=80  : 약한 Edge 기준
    # threshold2=160 : 강한 Edge 기준
    # 결과: small과 같은 (120, 160) 0/255 이미지
    edge = cv2.Canny(blurred, 80, 160)
    # <<< TODO A07
    return gray, small, edge

# ============================================================
# 3. 영상 frame 처리 함수
# ============================================================
def process_frame(frame):
    """Writer가 받을 160×120 BGR 프레임을 반환합니다.

    - frame: VideoCapture가 읽은 BGR frame
    - 반환: process_image의 edge를 3채널로 바꾼 (120, 160, 3) 배열
    """
    # >>> TODO A08
    # process_image(frame)의 세 번째 반환값 edge를 받아 3채널 BGR로 변환해 반환하세요.

    # 이미지와 같은 처리 함수를 재사용 (파일을 새로 열지 않음)
    # gray·small은 쓰지 않으므로 _로 받음
    _, _, edge = process_image(frame)

    # VideoWriter는 3채널 frame을 기대
    # 1채널 (120, 160) -> 3채널 (120, 160, 3), 세 채널 값이 같음
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # <<< TODO A08

# ============================================================
# 4. 이미지 3장 처리 및 단계별 PNG 저장·재읽기 검증
# ============================================================
rows = []
for filename in ["scene.png", "noisy.png", "objects.png"]:
    # --------------------------------------------------------
    # 이미지 읽기 및 처리 시간 측정
    # --------------------------------------------------------
    # 파일 읽기는 함수 밖(호출한 쪽)에서 수행
    image = cv2.imread("data/" + filename)
    # process_image 한 번의 소요 시간 (환경 의존 참고값)
    start = perf_counter(); gray, small, edge = process_image(image); seconds = perf_counter()-start
    # 예: "scene.png" -> "scene"
    stem = Path(filename).stem

    # --------------------------------------------------------
    # 단계별 결과 저장 후 다시 읽어 픽셀 일치 확인
    # --------------------------------------------------------
    for stage, result in [("gray", gray), ("small", small), ("edge", edge)]:
        # saved: imwrite 성공 여부 (True/False)
        path = f"outputs/p07_{stem}_{stage}.png"; saved = cv2.imwrite(path, result)
        # IMREAD_UNCHANGED: 저장된 그대로(1채널) 읽음
        # 주의: 기본 플래그로 읽으면 3채널로 바뀌어 비교가 항상 False
        reread = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # 한 행 기록 (CSV 열 순서와 같음)
        # result.shape[1] : 너비, result.shape[0] : 높이
        # np.array_equal  : PNG는 무손실이므로 True여야 정상
        rows.append([filename, stage, result.shape[1], result.shape[0], str(result.dtype), saved, np.array_equal(result,reread), seconds])

    # --------------------------------------------------------
    # 비교 PNG: 원본 축소 | small | edge 가로 연결
    # --------------------------------------------------------
    # hconcat은 높이·채널 수가 같아야 하므로
    # 원본은 160×120으로 줄이고, 1채널 small·edge는 3채널로 변환
    # 결과: (120, 480, 3)
    preview = cv2.hconcat([cv2.resize(image,(160,120)), cv2.cvtColor(small,cv2.COLOR_GRAY2BGR), cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(f"outputs/p07_{stem}_compare.png", preview)

# ============================================================
# 5. 영상 전체 frame 처리 후 저장
# ============================================================
# 원본 영상 열기 및 원본 FPS 확인
cap=cv2.VideoCapture("data/sample.avi"); fps=cap.get(cv2.CAP_PROP_FPS)
# fourcc=MJPG          : JPEG 기반 코덱
# fps=fps              : 모든 frame을 저장하므로 원본 FPS 그대로 사용
# frameSize=(160, 120) : process_frame 결과 크기와 같아야 함
writer=cv2.VideoWriter("outputs/p07_video.avi", cv2.VideoWriter_fourcc(*"MJPG"),fps,(160,120))
count=0
while cap.isOpened() and writer.isOpened():
    ok, frame=cap.read()
    # 더 읽을 frame이 없으면 종료
    if not ok: break
    # frame -> 3채널 Edge frame -> 저장
    writer.write(process_frame(frame)); count+=1
cap.release();writer.release()

# ============================================================
# 6. 요약 CSV 저장
# ============================================================
# source    : 원본 파일 이름
# stage     : 처리 단계 (gray / small / edge)
# width     : 결과 너비
# height    : 결과 높이
# dtype     : 결과 자료형 (uint8)
# saved     : PNG 저장 성공 여부
# png_equal : 저장 후 재읽기 픽셀 일치 여부
# process_s : 해당 이미지 process_image 소요 시간 (환경 의존)
with open("outputs/result_summary.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);w.writerow(["source","stage","width","height","dtype","saved","png_equal","process_s"]);w.writerows(rows)

# ============================================================
# 7. 결과 확인 출력
# ============================================================
# 저장한 영상을 다시 열어 frame 수·크기·FPS 검증
check=cv2.VideoCapture("outputs/p07_video.avi")
# 이미지 3장 × 단계 3개 = 9행
print("이미지 처리표:",len(rows),"행 / 원본3장")
# 처리한 frame 수와 재읽기 frame 수 (예: 48 48)
print("Frame 처리/재읽기:",count,int(check.get(cv2.CAP_PROP_FRAME_COUNT)))
# get(3) : CAP_PROP_FRAME_WIDTH  -> 160
# get(4) : CAP_PROP_FRAME_HEIGHT -> 120
# get(5) : CAP_PROP_FPS          -> 12.0
print("재읽기 크기/FPS:",int(check.get(3)),int(check.get(4)),check.get(5));check.release()
# 9행 모두 저장 성공 + 재읽기 일치면 True
print("PNG 저장·재읽기 모두 일치:",all(r[5] and r[6] for r in rows))
