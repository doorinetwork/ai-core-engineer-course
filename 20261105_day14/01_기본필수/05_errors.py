"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
# Path: 파일 위치 기준의 경로 계산
from pathlib import Path
# json: 지표 딕셔너리를 JSON 문자열로 저장·출력
import json
# NumPy: 배열 계산과 npz 가중치 읽기
import numpy as np
# pandas: 결과 표(DataFrame) 출력·CSV 저장
import pandas as pd
# PyTorch: Tensor 연산과 모델 학습
import torch
# nn: 층(Conv2d·Linear 등)과 손실 함수
from torch import nn
# ImageFolder: data/분할/클래스 폴더 구조를 데이터셋으로 읽기
from torchvision.datasets import ImageFolder
# transforms: 이미지 크기 변경·Tensor 변환
from torchvision import transforms as tr
# sklearn.metrics: 정확도·F1·분류 보고서·혼동행렬
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ============================================================
# 2. 실행 환경·경로 고정
# ============================================================
# set_num_threads(1) : CPU 스레드를 1개로 고정 -> 연산 순서가 같아 결과 재현에 유리
torch.set_num_threads(1)
# manual_seed(14) : 난수 시작값 고정 -> 같은 초기 가중치·같은 결과
torch.manual_seed(14)
# BASE : 이 .py 파일이 있는 폴더(다른 작업 폴더에서 실행해도 같은 위치를 가리킴)
BASE = Path(__file__).resolve().parent
# DATA : 입력 이미지 폴더 data/ (train·valid·test 하위 폴더)
DATA = BASE / "data"
# OUT : 결과 저장 폴더(이 파일과 같은 폴더)
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# tr.Resize((32, 32)) : 이미지를 높이 32 × 너비 32로 변경
# tr.ToTensor()       : PIL 이미지 [H,W,C] 0~255 -> Tensor [C,H,W] 0.0~1.0
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# ============================================================
# 3. train·valid 이미지 읽기
# ============================================================
# ImageFolder(root, transform) : 폴더 구조를 데이터셋으로 읽고, 꺼낼 때마다 transform 적용
#   root=DATA / "train" : 학습용 21장(circle 12·square 6·triangle 3)
#   root=DATA / "valid" : 비교·선택용 9장(클래스별 3장)
#   transform=transform : 32×32 Resize + [0,1] Tensor 변환
# 주의: test는 읽지 않음(최종 평가용으로 남겨 둠)
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# train[i] : (이미지 Tensor [3,32,32], 라벨) 튜플 -> [0]으로 이미지만 꺼냄
# torch.stack(...) : 이미지 21장을 한 묶음으로 쌓음 -> X_train 모양 [21, 3, 32, 32] (N,C,H,W)
X_train = torch.stack([train[i][0] for i in range(len(train))])
# torch.tensor(targets, dtype=torch.long) : 라벨 목록을 정수 Tensor로 변환(CrossEntropyLoss 요구 형식)
#   예: [0,0,...,1,...,2] -> 모양 [21]
y_train = torch.tensor(train.targets, dtype=torch.long)
# valid도 같은 방식 -> X_valid [9, 3, 32, 32], y_valid [9]
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)

# ============================================================
# 4. 모델 구성·source Backbone 가중치 읽기
# ============================================================
# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
# 번호별 층과 Tensor 모양(입력 [N, 3, 32, 32] 기준):
#   0 Conv2d(3, 8, 3, padding=1)  : RGB 3채널 -> 특징 8채널, 3×3 커널 -> [N, 8, 32, 32]
#   1 BatchNorm2d(8)              : 채널별 정규화(running 평균·분산 사용)
#   2 ReLU()                      : 음수를 0으로
#   3 MaxPool2d(2)                : 2×2 최댓값 -> [N, 8, 16, 16]
#   4 Conv2d(8, 16, 3, padding=1) : 두 번째 Conv(partial에서 여는 층) -> [N, 16, 16, 16]
#   5 BatchNorm2d(16), 6 ReLU()   : 정규화·활성화 (Grad-CAM은 6번 출력 사용)
#   7 AdaptiveAvgPool2d((4, 4))   : 크기와 무관하게 4×4로 평균 -> [N, 16, 4, 4]
#   8 Flatten()                   : 16×4×4 = 256개 특징 -> [N, 256]
#   9 Linear(256, 3)              : 256 특징 -> 3클래스 logit [N, 3]
# bias=False : 바로 뒤 BatchNorm이 이동값(beta)을 갖고 있어 Conv bias가 불필요
model = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
)

# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
# np.load(file, allow_pickle=False) : npz 배열 묶음 읽기
#   allow_pickle=False : 파이썬 객체(pickle) 실행을 막고 숫자 배열만 허용 -> 안전한 읽기
# state_dict() : 층 이름 -> 가중치 Tensor 사전, 예: "0.weight", "1.running_mean"
# saved.files  : npz 안의 배열 이름 목록(Backbone 0~8번 층 이름만 들어 있음)
# torch.from_numpy(saved[name].copy()) : NumPy 배열 -> Tensor 변환(.copy()로 파일과 분리된 배열 사용)
# state.update(...) : Backbone 가중치만 source 값으로 바꾸고, 9번 Head는 무작위 초기값 유지
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    state = model.state_dict()
    state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
# load_state_dict(state) : 수정한 사전을 모델에 다시 넣음 -> source Backbone 적용 완료
model.load_state_dict(state)

# ============================================================
# 5. Head-only 학습 설정(Freeze)
# ============================================================
# requires_grad는 가중치 갱신, eval()은 BatchNorm 통계 사용 방식을 제어합니다.
# 전체 파라미터를 requires_grad=False로 잠금 -> Backbone(0~8) 고정
for p in model.parameters():
    p.requires_grad = False
# 9번 Head(Linear 256->3)만 다시 열어 학습 대상으로 지정 -> 256×3 + 3 = 771개
for p in model[9].parameters():
    p.requires_grad = True
# model.eval() : BatchNorm이 배치 통계 대신 저장된 running 평균·분산을 사용
# 주의: eval()은 gradient를 끄지 않음 -> Head 학습·Grad-CAM 미분은 그대로 가능
model.eval()  # 고정 Backbone의 running 통계도 보존

# AdamW(params, lr, weight_decay) : 가중치 갱신 규칙
#   params=model[9].parameters() : Head 파라미터만 전달
#   lr=0.01                      : 학습률(한 번에 움직이는 크기)
#   weight_decay=0.0             : 가중치 감쇠 미사용
optimizer = torch.optim.AdamW(model[9].parameters(), lr=0.01, weight_decay=0.0)
# CrossEntropyLoss() : logit [N,3]과 정답 번호 [N]으로 분류 손실 계산(내부에서 softmax 처리)
criterion = nn.CrossEntropyLoss()

# ============================================================
# 6. 초기 모델 학습(1회)과 valid 예측
# ============================================================
# 오류 관찰용 초기 1회 학습. 07번 최종 8회 학습 모델과 구분합니다.
for epoch in range(1):
    # --------------------------------------------------------
    # 기울기 초기화 -> 손실 계산 -> 역전파 -> 갱신
    # --------------------------------------------------------
    # zero_grad() : 이전에 누적된 기울기를 0으로 초기화
    optimizer.zero_grad()
    # model(X_train) : train 21장 전체를 한 번에 넣어 logit [21, 3] 계산
    loss = criterion(model(X_train), y_train)
    # backward() : 손실을 미분해 requires_grad=True인 Head에만 기울기 저장
    loss.backward()
    # step() : 저장된 기울기로 Head 가중치 실제 갱신
    optimizer.step()

# torch.no_grad() : 예측만 하므로 미분 기록을 끔(메모리·시간 절약)
with torch.no_grad():
    # logits : valid 9장의 클래스별 점수 [9, 3]
    logits = model(X_valid)
    # softmax(dim=1) : 각 행을 합 1인 확률로 변환, 예: [2.0, 0.5, 0.1] -> [0.73, 0.16, 0.11]
    probabilities = logits.softmax(dim=1)
    # argmax(dim=1) : 각 행에서 가장 큰 열 번호 = 예측 클래스, 예: [0.2, 1.8, 0.1] -> 1
    predictions = logits.argmax(dim=1)

# ============================================================
# 7. 혼동행렬 계산 (TODO B05)
# ============================================================
# TODO B05: 실제 y_valid와 predictions로 혼동행렬을 만드세요. 행=실제, 열=예측.
# BEGIN B05
# confusion_matrix(y_true, y_pred, labels) : 3×3 개수 표
#   y_true=y_valid         : 실제 라벨(행)
#   y_pred=predictions     : 예측 라벨(열)
#   labels=[0, 1, 2]       : 행·열 순서를 circle·square·triangle로 고정
# 값 예시: matrix[2, 0] = 3 -> 실제 triangle 3장을 모두 circle로 예측
# 주의: 인자 순서를 바꾸면 행·열 의미가 뒤바뀜(행=예측, 열=실제)
matrix = confusion_matrix(y_valid, predictions, labels=[0, 1, 2])
# END B05
print("혼동행렬:\n", matrix)

# ============================================================
# 8. 클래스별 지표와 오답 목록
# ============================================================
# classification_report : valid 9장 기준 precision·recall·f1·support
#   target_names=valid.classes : ['circle', 'square', 'triangle'] (labels 순서와 같음)
#   zero_division=0            : 예측이 0건인 클래스의 precision 0/0을 0으로 보고
#   digits=3                   : 소수 셋째 자리
print(classification_report(y_valid, predictions, labels=[0, 1, 2], target_names=valid.classes, zero_division=0, digits=3))

# 오답 위치 찾기(메서드 체인 단계별):
#   (predictions != y_valid) : 예측과 실제가 다르면 True인 bool Tensor [9]
#   .nonzero()               : True 위치 번호 [[k], ...] 모양 [오답 수, 1]
#   .flatten()               : 1차원 [k, ...]으로 펼침
#   .tolist()                : 파이썬 리스트로 변환, 예: [5, 6, 7]
errors = (predictions != y_valid).nonzero().flatten().tolist()
print("오답 개수:", len(errors))
# errors[:3] : 앞의 최대 3건만 출력
# valid.samples[i][0] : i번째 파일 경로 -> Path(...).name으로 파일 이름만 표시
# valid.classes[번호] : 번호 -> 클래스 이름, 예: 0 -> 'circle'
for i in errors[:3]:
    print(Path(valid.samples[i][0]).name, "실제:", valid.classes[y_valid[i]], "예측:", valid.classes[predictions[i]])
