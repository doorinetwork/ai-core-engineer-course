"""실습 03. 정상 이미지 6장과 YOLO TXT 라벨을 읽습니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import cv2

# ============================================================
# 2. 경로와 클래스 이름
# ============================================================
BASE = Path(__file__).resolve().parent
# names[i] : 클래스 번호 i의 이름 (dataset.yaml의 names와 같은 순서) -> 0=square, 1=circle, 2=triangle
names = ["square", "circle", "triangle"]

# ============================================================
# 3. 이미지와 같은 이름의 라벨 TXT 읽기
# ============================================================
# sorted(glob("*.png"))[:6] : train 이미지 12장 중 이름순 앞 6장
for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:
    # image.shape -> (128, 160, 3) = (H, W, C)
    image = cv2.imread(str(image_path))
    # 라벨 경로 = labels/train/ + 이미지와 같은 stem + .txt
    # 예: images/train/basic_train_000.png -> labels/train/basic_train_000.txt
    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    # >>> B04
    # 요구: UTF-8 라벨 파일의 문자열을 읽고 줄 목록으로 나누세요.
    # 메서드 체인을 한 단계씩 나누어 읽는다.
    #   read_text(encoding="utf-8") : 파일 전체를 문자열 1개로 읽음
    #   splitlines()                : 줄바꿈 기준으로 나눈 문자열 목록 (줄 1개 = 객체 1개)
    # 예: "0 0.796875 0.33984375 0.19375 0.2421875\n1 ...\n2 ..." -> 문자열 3개
    # 주의: 빈 파일이면 [] -> 객체 0개인 정상 negative 이미지 (파일이 없는 것과 다름)
    lines = (
        label_path
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # <<< B04
    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))

    # --------------------------------------------------------
    # 한 줄 = class cx cy w h (좌표 4개는 0~1 정규화)
    # --------------------------------------------------------
    for line in lines:
        # split() : 공백 기준 5칸 -> map(float, ...)로 숫자 변환
        #   cls    : 클래스 번호 (0.0 -> int로 바꿔 names 조회)
        #   cx, cy : 박스 중심 x/W, y/H
        #   width  : 박스 폭 / 이미지 W(160), height : 박스 높이 / 이미지 H(128)
        # 예: square 0.796875 0.33984375 0.19375 0.2421875
        #     -> 중심 (127.5, 43.5)px, 크기 31×31px
        cls, cx, cy, width, height = map(float, line.split())
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)
