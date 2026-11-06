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

# OpenCV: 열지도 색 입히기·이미지 합성·PNG 저장
import cv2

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
# 6. 8 epoch 학습과 best epoch 선택 (TODO B07)
# ============================================================

# history   : epoch별 valid 손실 기록 목록
# best_loss : 지금까지 가장 낮은 valid 손실(처음은 무한대)
history = []
best_loss = float("inf")
# epochs=8 : epoch 1~8, 매번 train 21장 전체로 한 번 갱신
for epoch in range(1, 8+1):
    # --------------------------------------------------------
    # train 손실 계산·역전파·갱신 (TODO B07)
    # --------------------------------------------------------
    # zero_grad() : 이전 epoch 기울기 초기화
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    # TODO B07: 역전파와 Optimizer 갱신을 수행하세요.
    # BEGIN B07
    # backward() : 손실을 미분해 Head 파라미터의 .grad에 기울기 저장
    loss.backward()
    # step() : 저장된 기울기로 Head 가중치 실제 갱신
    # 주의: 원본 pass는 갱신이 없어 8 epoch 모두 초기 Head 그대로였음
    optimizer.step()
    # END B07

    # --------------------------------------------------------
    # valid 손실 기록과 best 상태 저장
    # --------------------------------------------------------
    # no_grad() : 평가만, .item() : 1개짜리 Tensor -> 파이썬 float
    with torch.no_grad():
        valid_loss = criterion(model(X_valid), y_valid).item()
    # 기록 값:
    #   epoch      : 학습 회차(1~8)
    #   valid_loss : 그 회차 갱신 후 valid 9장 평균 손실
    history.append({"epoch": epoch, "valid_loss": valid_loss})
    # valid 손실이 가장 낮으면 그 epoch과 가중치를 보관
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_epoch = epoch
        # 메서드 체인:
        #   v.detach() : 미분 그래프에서 분리
        #   .clone()   : 이후 학습으로 바뀌지 않도록 값 복사
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
# 가장 좋았던 epoch의 가중치로 되돌림
model.load_state_dict(best_state)

# ============================================================
# 7. 최종 valid 예측 (TODO B08)
# ============================================================

with torch.no_grad():
    # logits : valid 9장의 클래스별 점수 [9, 3]
    logits = model(X_valid)
    # softmax(1) : 행별 확률 [9, 3] -> 최대값을 confidence로 기록
    probabilities = logits.softmax(1)
    # TODO B08: 각 행의 가장 큰 logit 열을 예측 라벨로 고르세요.
    # BEGIN B08
    # argmax(dim=1) : 행(이미지)마다 가장 큰 logit의 열 번호 -> predictions [9]
    # 값 예시: [0.2, 1.8, -0.3] -> 클래스 1(square)
    # 주의: 원본 zeros는 모두 circle(0)로 예측한 가짜 결과였음
    predictions = logits.argmax(dim=1)
    # END B08

# ============================================================
# 8. 지표·기록 파일 저장 (TODO B09)
# ============================================================

# metrics 키:
#   accuracy   : valid 정확도
#   macro_f1   : 세 클래스 F1 동일 가중 평균(labels=[0,1,2], 분모 0이면 0)
#   best_epoch : valid 손실이 가장 낮았던 epoch
#   valid_n    : 평가에 쓴 valid 이미지 수(9)
#   source     : Backbone 출처(합성 이미지 사전학습)
#   test_used  : test 사용 여부(False)
metrics = {"accuracy": accuracy_score(y_valid, predictions), "macro_f1": f1_score(y_valid, predictions, labels=[0,1,2], average="macro", zero_division=0), "best_epoch": best_epoch, "valid_n": len(valid), "source": "synthetic_source_pretraining", "test_used": False}
# to_csv(path, index=False) : 행 번호 없이 CSV 저장
# metrics.csv : 지표 1행
pd.DataFrame([metrics]).to_csv(OUT / "metrics.csv", index=False)
# history.csv : epoch별 valid 손실 8행
pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
# predictions.csv 열:
#   file       : valid 파일 이름
#   actual     : 실제 라벨 번호
#   predicted  : 예측 라벨 번호
#   confidence : 예측 클래스 확률(행별 softmax 최댓값)
pd.DataFrame({"file": [Path(p).name for p,_ in valid.samples], "actual": y_valid.numpy(), "predicted": predictions.numpy(), "confidence": probabilities.max(1).values.numpy()}).to_csv(OUT / "predictions.csv", index=False)
# TODO B09: metrics를 UTF-8 JSON 문자열로 저장하세요. 아래 제공된 경로를 사용합니다.
# BEGIN B09
# json.dumps(obj, ensure_ascii, sort_keys) : 사전 -> JSON 문자열
#   ensure_ascii=False : 한글 등을 \uXXXX로 바꾸지 않고 그대로 저장
#   sort_keys=True     : 키를 알파벳 순서로 정렬 -> 재실행 시 문자열 비교가 쉬움
# write_text(..., encoding="utf-8") : metrics.json 파일에 UTF-8로 기록
metrics_json = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
(OUT / "metrics.json").write_text(metrics_json, encoding="utf-8")
# END B09
# confusion.csv : 행=실제, 열=예측, 이름은 circle·square·triangle
pd.DataFrame(confusion_matrix(y_valid, predictions, labels=[0,1,2]), index=valid.classes, columns=valid.classes).to_csv(OUT / "confusion.csv")

# ============================================================
# 9. 오답 이미지 저장
# ============================================================

# 이전 실행의 project_error_*.png 삭제 -> 이번 실행의 오답만 남김
for old in OUT.glob("project_error_*.png"):
    old.unlink()
# 오답 위치 번호 리스트, 예: [] (오답 0건) 또는 [5, 6]
errors = (predictions != y_valid).nonzero().flatten().tolist()
# 오답이 있을 때만 최대 8장 저장, 이름 예: project_error_05.png
# 주의: 오답이 없으면 이미지를 만들지 않음(억지로 만들지 않음)
for i in errors[:8]:
    image = (X_valid[i].permute(1,2,0).numpy()*255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"project_error_{i:02d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# ============================================================
# 10. 최종 모델 Grad-CAM
# ============================================================

# 대상 이미지: 오답이 있으면 첫 오답, 없으면 valid 0번(circle_000.png)
index = errors[0] if errors else 0
# hook 대신 Sequential을 둘로 나눕니다. 관심 클래스의 logit을 미분합니다.
# image    : 이미지 1장 [1, 3, 32, 32], 미분 기록 시작
# features : 0~6번 층 출력 [1, 16, 16, 16]
# logits_one : 7~9번 층 출력 [1, 3]
image = X_valid[index:index + 1].clone().requires_grad_(True)
features = model[:7](image)
logits_one = model[7:](features)
# target : 예측 클래스 번호(가장 큰 logit)
target = int(logits_one.argmax(1).item())
# 선택 logit을 특징 지도로 미분 -> [1, 16, 16, 16]
gradient = torch.autograd.grad(logits_one[0, target], features)[0]
# 채널별 공간 평균 -> 채널 중요도 [1, 16, 1, 1]
weights = gradient.mean(dim=(2, 3), keepdim=True)
# 채널 가중합 -> relu로 양의 반응만 -> [1, 16, 16]
cam = (weights * features).sum(dim=1).relu()
# 32×32로 확대 후 [32, 32]만 꺼냄
cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
cam = cam.detach().numpy()
cam = cam / max(float(cam.max()), 1e-8)  # 양의 반응이 없으면 0 지도를 유지
# 원본 RGB [32, 32, 3] uint8
rgb = (X_valid[index].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
# 열지도 색 입히기(JET, BGR)
heat_bgr = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
# 원본 0.6 + 열지도 0.4 합성(같은 32×32 좌표)
overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, heat_bgr, 0.4, 0)

# 256×256으로 확대해 project_cam.png 저장
cv2.imwrite(str(OUT / "project_cam.png"), cv2.resize(overlay,(256,256)))

# ============================================================
# 11. 결과 출력
# ============================================================

# metrics를 키 정렬 JSON 한 줄로 출력, 예: {"accuracy": 1.0, "best_epoch": 8, ...}
print(json.dumps(metrics,ensure_ascii=False,sort_keys=True))
print("오답 수:",len(errors),"예측 행:",len(valid),"test 사용: False")
