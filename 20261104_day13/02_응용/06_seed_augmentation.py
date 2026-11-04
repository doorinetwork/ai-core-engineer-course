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
# 2. Seed 기반 증강 함수
# ============================================================
# - image: BGR uint8 배열. 복사본에만 변환을 적용하고 원본은 그대로 둔다
# - seed: 같은 값이면 flip·beta가 항상 같게 결정된다
# - 반환: (증강 이미지, {"seed", "flip", "beta"} 설정 dict)
def augment(image, seed):
    """로컬 난수로 밝기·좌우반전을 결정합니다. 입력 배열은 변경하지 않습니다."""
    # --------------------------------------------------------
    # 로컬 난수 생성기로 증강 설정 결정
    # --------------------------------------------------------
    # default_rng(seed): 이 함수 안에서만 쓰는 Generator 생성
    # 주의: np.random.seed()처럼 전역 난수 상태를 바꾸지 않음
    rng = np.random.default_rng(seed)
    # [0, 1) 난수가 0.5 미만이면 좌우반전 (확률 50%)
    flip = bool(rng.random() < 0.5)
    # 밝기 변화량: -25 이상 25 이하 정수 (high=26은 포함되지 않음)
    # 예: seed=42 -> flip=False, beta=8
    beta = int(rng.integers(-25, 26))
    # 원본 보호를 위해 복사본에서 작업
    result = image.copy()

    # --------------------------------------------------------
    # 좌우반전 적용
    # --------------------------------------------------------
    # cv2.flip(src, flipCode)
    # flipCode=1 : 좌우(수평) 반전
    if flip:
        result = cv2.flip(result, 1)

    # --------------------------------------------------------
    # 밝기 이동 (float32에서 계산 후 uint8 복귀)
    # --------------------------------------------------------
    # >>> TODO A06
    # float32에서 beta를 더하고 [0,255]로 clip한 뒤 uint8로 돌리세요. uint8 상태로 직접 더하지 않습니다.
    # astype(np.float32): 음수·255 초과를 표현할 수 있도록 변환
    # + beta            : 모든 픽셀 밝기를 beta만큼 이동
    # np.clip(..., 0, 255): 범위를 벗어난 값을 0 또는 255로 고정
    # astype(np.uint8)  : 이미지 저장 형식으로 복귀
    #
    # 예: beta=8, 픽셀 250 -> 258 -> clip 255
    #     uint8에서 직접 더하면 250 + 8 = 2 (순환 오류)
    result = np.clip(result.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    # <<< TODO A06
    # 결과 이미지와 재현에 필요한 설정 반환
    return result, {"seed": seed, "flip": flip, "beta": beta}

# ============================================================
# 3. 같은 seed로 두 번 증강해 재현성 확인
# ============================================================
image = cv2.imread("data/scene.png")
# 같은 seed=42로 두 번 호출 -> 두 결과가 픽셀까지 같아야 함
first, settings = augment(image, 42)
second, _ = augment(image, 42)
# 원본 | 1회차 | 2회차를 가로로 붙여 비교 이미지 저장
# 주의: hconcat은 세 이미지의 높이·dtype이 같아야 함
cv2.imwrite("outputs/p06_comparison.png", cv2.hconcat([image, first, second]))
# 증강 설정을 JSON으로 저장
# seed : 사용한 난수 seed
# flip : 좌우반전 여부
# beta : 더한 밝기 값
Path("outputs/p06_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

# ============================================================
# 4. 결과 출력
# ============================================================
# np.array_equal(): shape와 모든 픽셀 값이 같으면 True
print("같은 seed 결과 일치:", np.array_equal(first, second))
# 예: {'seed': 42, 'flip': False, 'beta': 8}
print("설정:", settings)
