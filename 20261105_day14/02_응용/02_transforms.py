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
# 예: 40×40 RGB PNG -> Tensor shape (3, 32, 32), 값 범위 [0, 1]
# 주의: ImageNet용 Normalize(mean/std)는 넣지 않음 (source 학습 조건과 동일하게 유지)
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 공식 ResNet18의 ImageNet 가중치 정보(Enum)와 전용 transforms
from torchvision.models import ResNet18_Weights
# PNG 이미지를 PIL 형식으로 열기
from PIL import Image

# ============================================================
# 3. 공식 ResNet18 Weight의 입력 변환 가져오기
# ============================================================
# ResNet18_Weights.IMAGENET1K_V1 : ImageNet-1K로 학습한 공식 가중치의 이름표(Enum)
# Enum·변환 생성은 다운로드하지 않음
# 주의: get_state_dict()나 resnet18(weights=...)를 호출해야 실제 가중치를 내려받음
weights = ResNet18_Weights.IMAGENET1K_V1

# TODO A02: Weight가 제공하는 transforms()를 가져오세요.
# BEGIN A02
# weights.transforms() : 이 가중치를 학습할 때 쓴 입력 전처리를 그대로 만든 객체
# 처리 순서: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(mean, std)
# .resize_size : 짧은 변 기준 크기 변경 값 -> [256]
# .crop_size   : 가운데 자르기 크기 -> [224]
# .mean / .std : ImageNet 채널별 평균·표준편차 -> [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
official = weights.transforms()
# END A02


# ============================================================
# 4. 같은 이미지에 두 변환을 적용해 입력 조건 비교
# ============================================================
# valid/circle의 첫 PNG를 RGB 3채널로 열기
# sorted(...) : 파일 이름 순서 고정 -> 매 실행 같은 이미지 선택
# .convert("RGB") : 알파·회색 채널이 있어도 RGB 3채널로 통일
image = Image.open(sorted((DATA/"valid"/"circle").glob("*.png"))[0]).convert("RGB")

# 소형 source 경로: 40×40 -> (3, 32, 32), 값 범위 [0, 1]
print("소형 source 입력:",tuple(transform(image).shape),"범위 [0,1]")

# 공식 ResNet 경로: 40×40 -> 256으로 확대 -> 224 자르기 -> (3, 224, 224)
print("공식 ResNet 입력:",tuple(official(image).shape))
print("공식 Resize/Crop:", official.resize_size,official.crop_size)
print("공식 mean/std:", official.mean,official.std)

# 주의: 소형 source Backbone은 Normalize 없이 [0,1]로 학습됨
#       ImageNet mean/std를 소형 모델에 넣거나, [0,1] 입력을 ResNet18에 넣으면 학습 조건이 어긋남
print("작은 source와 ImageNet의 Normalize를 서로 섞지 않습니다.")
