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

# OpenCV: 열지도 색 입히기, 합성, PNG 저장
import cv2

# ============================================================
# 3. train / valid 이미지를 Tensor 묶음으로 준비
# ============================================================
# ImageFolder(root, transform) : 폴더 구조로 이미지와 라벨을 읽고 transform 적용
# root=DATA / "train"   : 학습에만 사용 (180장: circle 120, square 40, triangle 20)
# root=DATA / "valid"   : epoch·후보 비교에만 사용 (90장: 클래스별 30)
# transform=transform   : Resize(32,32) + ToTensor -> (3, 32, 32), [0,1]
# 주의: test 폴더는 읽지 않음
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# train[i] -> (이미지 Tensor (3,32,32), 라벨 번호)
# torch.stack : 이미지 N장을 하나의 배치 Tensor로 쌓음
# X_train : (180, 3, 32, 32)  /  y_train : (180,) 라벨 번호 0·1·2
# X_valid : (90, 3, 32, 32)   /  y_valid : (90,)
# dtype=torch.long : CrossEntropyLoss가 요구하는 정수 라벨 자료형
X_train = torch.stack([train[i][0] for i in range(len(train))])
y_train = torch.tensor(train.targets, dtype=torch.long)
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)

# ============================================================
# 4. 모델 생성 및 source Backbone 불러오기
# ============================================================
# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
# [0] Conv2d(3, 8, 3, padding=1)  : (B,3,32,32) -> (B,8,32,32)
# [1] BatchNorm2d(8), [2] ReLU, [3] MaxPool2d(2) -> (B,8,16,16)
# [4] Conv2d(8, 16, 3, padding=1) : (B,8,16,16) -> (B,16,16,16)
# [5] BatchNorm2d(16), [6] ReLU   -> Grad-CAM에 쓰는 features 위치 (model[:7] 출력)
# [7] AdaptiveAvgPool2d((4, 4))   : (B,16,16,16) -> (B,16,4,4)
# [8] Flatten                     : (B,256)
# [9] Linear(256, 3)              : (B,3) 클래스별 logit  <- 새 Head
model = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
)

# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
# allow_pickle=False : 파이썬 객체 복원 금지 -> 수치 배열만 읽음
# .copy()            : npz 파일을 닫은 뒤에도 쓸 수 있도록 배열 복사
# state.update(...)  : Backbone 키만 source 값으로 덮어씀, Head(9.*)는 seed 14 초기값 유지
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    state = model.state_dict()
    state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
model.load_state_dict(state)

# ============================================================
# 5. 초기 Head-only 모델 학습 (제공 부분)
# ============================================================
# requires_grad는 가중치 갱신, eval()은 BatchNorm 통계 사용 방식을 제어합니다.
# 전체 고정 후 Head(model[9])만 학습 대상으로 지정
for p in model.parameters():
    p.requires_grad = False
for p in model[9].parameters():
    p.requires_grad = True

# 고정 Backbone의 running 통계도 보존
# 주의: eval()이어도 Head의 gradient 계산·갱신은 정상 동작
model.eval()

# torch.optim.AdamW(params, lr, weight_decay)
# params=model[9].parameters() : Head 771개만 전달
# lr=0.01                      : 03·04·07(0.003)보다 큰 학습률
# weight_decay=0.0             : 규제 없음
optimizer = torch.optim.AdamW(model[9].parameters(), lr=0.01, weight_decay=0.0)
# 클래스 가중치 없는 기본 CrossEntropy
criterion = nn.CrossEntropyLoss()

# 제공된 8회 학습. 평가용 valid를 backward에 넣지 않습니다.
# zero_grad() -> 순전파·손실 -> backward() -> step() 순서
# 주의: best epoch 선택 없이 8회 학습한 마지막 상태를 그대로 사용 (07의 최종 모델과 다름)
for epoch in range(8):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

# --------------------------------------------------------
# valid 예측
# --------------------------------------------------------
# logits        : (90, 3) 클래스별 점수
# probabilities : softmax(dim=1) -> 행마다 합 1인 확률
# predictions   : argmax(dim=1) -> 가장 큰 점수의 클래스 번호
#
# 예: logits [1.6, -1.4, -0.3] -> 예측 0(circle)
with torch.no_grad():
    logits = model(X_valid)
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)

# ============================================================
# 6. 같은 이미지에서 target을 바꿔 Grad-CAM 계산
# ============================================================
# valid 0번 이미지(circle_000.png) 하나를 고정
index=0

# target=0 : circle logit 기준 지도, target=1 : square logit 기준 지도
for target in [0,1]:

    # --------------------------------------------------------
    # Sequential을 features 앞뒤로 나누어 순전파
    # --------------------------------------------------------
    # image    : (1, 3, 32, 32), requires_grad_(True)로 미분 그래프 연결
    # features : model[:7] 출력 -> (1, 16, 16, 16)
    # out      : model[7:] 출력 -> (1, 3) logit
    # 주의: target마다 순전파를 새로 해 그래프를 새로 만듦
    image=X_valid[index:index+1].clone().requires_grad_(True)
    features=model[:7](image)
    out=model[7:](features)

    # TODO A06: target logit의 features에 대한 gradient를 계산하세요.
    # BEGIN A06
    # torch.autograd.grad(outputs, inputs)
    # outputs=out[0, target] : 미분할 scalar (0번 이미지의 target 클래스 logit)
    # inputs=features        : 기울기를 구할 대상 Tensor
    # [0]                    : 반환 tuple의 첫 원소 -> (1, 16, 16, 16)
    # 주의: 파라미터 .grad에 누적하지 않고 gradient만 반환
    gradient = torch.autograd.grad(out[0, target], features)[0]
    # END A06

    # --------------------------------------------------------
    # CAM 계산: 채널 가중치 × features -> ReLU -> 32×32 -> [0,1]
    # --------------------------------------------------------
    # gradient.mean((2,3), keepdim=True) : 채널별 공간 평균 -> (1, 16, 1, 1)
    # (... * features).sum(1)            : 채널 가중합 -> (1, 16, 16)
    # .relu()                            : 해당 target에 양으로 기여한 반응만 남김
    cam=(gradient.mean((2,3),keepdim=True)*features).sum(1).relu()

    # 16×16 -> 32×32 입력 좌표로 확대 후 NumPy (32, 32)
    # .detach() : 미분 그래프에서 분리
    # .numpy()  : Tensor -> NumPy 배열
    cam=torch.nn.functional.interpolate(cam.unsqueeze(1),(32,32),mode="bilinear",align_corners=False)[0,0].detach().numpy()

    # 최댓값으로 나누어 [0,1] (양의 반응이 없으면 0 지도 유지, 1e-8로 0 나누기 방지)
    cam=cam/max(float(cam.max()),1e-8)

    # --------------------------------------------------------
    # 열지도 합성 및 저장
    # --------------------------------------------------------
    # rgb  : (32,32,3) uint8 원본
    # heat : JET 색상 열지도 (BGR)
    rgb=np.uint8(X_valid[index].permute(1,2,0).numpy()*255)
    heat=cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)

    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    # src1=원본 BGR : 같은 32×32 좌표의 원본
    # alpha=.6      : 원본 비중
    # src2=heat     : 열지도
    # beta=.4       : 열지도 비중
    # gamma=0       : 더할 상수 밝기
    overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.6,heat,.4,0)

    # 256×256으로 확대 저장 -> outputs/target_0_cam.png, target_1_cam.png
    cv2.imwrite(str(OUT/f"target_{target}_cam.png"),cv2.resize(overlay,(256,256)))

    # logit    : 해당 target 클래스 점수
    # mean CAM : [0,1] 정규화 지도의 평균 (반응이 넓게 퍼질수록 큼)
    print("target:",valid.classes[target],"logit:",round(out[0,target].item(),4),"mean CAM:",round(float(cam.mean()),4))
