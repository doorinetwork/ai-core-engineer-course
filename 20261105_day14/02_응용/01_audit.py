"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
# 파일·폴더 경로를 객체로 다루는 Path
from pathlib import Path
# 결과 목록을 JSON 파일로 저장
import json
# 수치 배열 계산과 source 가중치(npz) 읽기
import numpy as np
# 결과 표(DataFrame) 생성·CSV 저장
import pandas as pd
# PyTorch: Tensor 연산, 자동 미분, 난수 seed
import torch
# 신경망 층(Conv2d, Linear 등)과 손실 함수
from torch import nn
# data/분할/클래스 폴더 구조를 데이터셋으로 읽는 ImageFolder
from torchvision.datasets import ImageFolder
# 이미지 변환(Resize, ToTensor) 묶음
from torchvision import transforms as tr
# 분류 평가 지표: 정확도, F1, 클래스별 보고서, 혼동행렬
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ============================================================
# 2. 재현 조건 고정 및 경로 설정
# ============================================================
# CPU 연산 스레드 1개로 고정 (스레드 수에 따른 결과 차이 축소)
torch.set_num_threads(1)
# 난수 seed 고정 -> 새 Head 초기값이 매 실행 같음
torch.manual_seed(14)

# BASE : 현재 .py 파일이 있는 폴더 (어느 작업 폴더에서 실행해도 같은 위치 기준)
# DATA : 원본 이미지·source 가중치 폴더 (읽기만 함)
# OUT  : 결과 저장 폴더 (없으면 생성, 있으면 그대로 사용)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# tr.Resize((32, 32)) : 이미지를 높이 32 × 너비 32로 크기 변경
# tr.ToTensor()       : PIL 이미지(H,W,C, 0~255) -> Tensor(C,H,W, 0.0~1.0)
#
# 예: 40×40 RGB PNG -> Tensor shape (3, 32, 32), 값 범위 [0, 1]
# 주의: ImageNet용 Normalize(mean/std)는 넣지 않음 (source 학습 조건과 동일하게 유지)
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 라벨별 파일 목록을 자동으로 만드는 dict (없는 키는 빈 list로 시작)
from collections import defaultdict
# 파일 바이트의 SHA-256 지문 계산 (바이트 동일 중복 감사)
import hashlib

# ============================================================
# 3. 분할·클래스별 개수 집계 및 중복 지문 수집
# ============================================================
# rows : 표의 한 행 = (split, class, n, ratio)
# seen : SHA-256 지문 -> 그 지문을 가진 파일 경로 목록
## 예: seen["3e15..."] = ["train/circle/circle_000.png"]
#     같은 지문에 경로가 2개 이상이면 바이트가 완전히 같은 중복 파일
rows = []
seen = defaultdict(list)
for split in ["train", "valid", "test"]:

    # --------------------------------------------------------
    # ImageFolder로 분할 폴더 읽기
    # --------------------------------------------------------
    # ImageFolder(root) : root 아래 하위 폴더 이름을 클래스로 사용
    # root=DATA / split : 예) data/train -> circle/, square/, triangle/
    # .classes      : 알파벳 순 클래스 이름 목록 -> ["circle", "square", "triangle"]
    # .class_to_idx : 클래스 이름 -> 번호 -> {"circle": 0, "square": 1, "triangle": 2}
    # .targets      : 파일 순서와 같은 라벨 번호 목록 -> [0, 0, ..., 1, ..., 2]
    # .samples      : (파일 경로, 라벨 번호) 목록
    # 주의: transform 없이 읽으므로 이미지를 Tensor로 바꾸지 않음 (개수·경로 감사 전용)
    dataset = ImageFolder(DATA / split)

    for label, name in enumerate(dataset.classes):
        # --------------------------------------------------------
        # 클래스별 개수와 비율 계산
        # --------------------------------------------------------
        # TODO A01: 해당 라벨의 개수를 세세요.
        # BEGIN A01
        # list.count(label) : targets 안에서 label과 같은 값의 개수
        # 예: train targets = [0]*120 + [1]*40 + [2]*20
        #     label=0 -> 120, label=1 -> 40, label=2 -> 20
        count = dataset.targets.count(label)
        # END A01

        # split : 분할 이름 (train / valid / test)
        # class : 클래스 이름 (circle / square / triangle)
        # n     : 해당 분할 안의 클래스 이미지 수
        # ratio : 해당 분할 전체 대비 비율 -> 예) train circle 120 / 180 = 0.6667
        rows.append({"split": split, "class": name, "n": count, "ratio": count/len(dataset)})

    # --------------------------------------------------------
    # 파일 바이트 SHA-256 지문 수집 (중복 감사)
    # --------------------------------------------------------
    # Path(path).read_bytes()     : 파일 전체 바이트 읽기
    # hashlib.sha256(...)         : 바이트의 SHA-256 계산
    # .hexdigest()                : 64자리 16진수 문자열로 변환
    # .relative_to(DATA)          : data/ 기준 상대 경로 -> 예) "valid/circle/circle_000.png"
    # 주의: test 파일도 지문을 계산하지만 픽셀로 모델 성능을 계산하지 않음
    for path,_ in dataset.samples:
        seen[hashlib.sha256(Path(path).read_bytes()).hexdigest()].append(str(Path(path).relative_to(DATA)))


# ============================================================
# 4. 개수표 저장·출력 및 중복 그룹 저장
# ============================================================
# 9행 표 (3분할 × 3클래스) -> outputs/class_counts.csv
table = pd.DataFrame(rows)
table.to_csv(OUT/"class_counts.csv", index=False)

# round(4) : 비율을 소수 넷째 자리까지 표시 -> 0.6666... -> 0.6667
print(table.round(4).to_string(index=False))

# 경로가 2개 이상인 지문만 중복 그룹으로 간주
# 예: [["train/circle/circle_000.png", "test/circle/circle_003.png"]] -> 중복 그룹 1개
duplicates = [paths for paths in seen.values() if len(paths)>1]
print("바이트 동일 중복 그룹:",len(duplicates))

# ensure_ascii=False : 한글 경로도 그대로 기록
# indent=2           : 사람이 읽기 쉽게 들여쓰기
# 중복이 없으면 빈 목록 [] 저장
(OUT/"duplicates.json").write_text(json.dumps(duplicates,ensure_ascii=False,indent=2),encoding="utf-8")
# test 픽셀은 중복 검사에만 쓰며 모델 성능을 계산하지 않습니다.
