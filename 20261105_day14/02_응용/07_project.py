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

# 학습 시간 측정용 고정밀 타이머
import time
# OpenCV: 열지도 색 입히기, 합성, PNG 저장
import cv2


# ============================================================
# 3. train / valid 이미지를 Tensor 묶음으로 준비
# ============================================================

# ImageFolder(root, transform) : 폴더 구조로 이미지와 라벨을 읽고 transform 적용
#
# root=DATA / "train"   : 학습에만 사용 (180장: circle 120, square 40, triangle 20)
# root=DATA / "valid"   : epoch·후보 비교에만 사용 (90장: 클래스별 30)
# transform=transform   : Resize(32,32) + ToTensor -> (3, 32, 32), [0,1]
# 주의: test 폴더는 읽지 않음
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# train[i] -> (이미지 Tensor (3,32,32), 라벨 번호)
# torch.stack : 이미지 N장을 하나의 배치 Tensor로 쌓음
#
# X_train : (180, 3, 32, 32)  /  y_train : (180,) 라벨 번호 0·1·2
# X_valid : (90, 3, 32, 32)   /  y_valid : (90,)
# dtype=torch.long : CrossEntropyLoss가 요구하는 정수 라벨 자료형
X_train = torch.stack([train[i][0] for i in range(len(train))])
y_train = torch.tensor(train.targets, dtype=torch.long)
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
y_valid = torch.tensor(valid.targets, dtype=torch.long)


# ============================================================
# 4. 비교 후보 정의
# ============================================================

# TODO A07: Head 기준선·Partial·Head+weight 세 조건을 고정하세요.
# BEGIN A07
# (strategy, balance) 목록
#
# ("head", "none")    : 기준선, 새 Head만 학습, 클래스 가중치 없음
# ("partial", "none") : 마지막 Conv + Head 학습, 클래스 가중치 없음
# ("head", "weight")  : 새 Head만 학습, train 손실에 클래스 가중치 적용
# 주의: 기준선 head/none에서 한 가지 조건만 바꾼 두 후보를 함께 비교
experiments=[("head","none"),("partial","none"),("head","weight")]
# END A07


# ============================================================
# 5. 후보별 전이학습 및 valid 평가
# ============================================================

rows = []
all_history = []
for strategy, balance in experiments:

    # --------------------------------------------------------
    # 후보마다 seed 재고정 -> 같은 Head 초기값에서 출발
    # --------------------------------------------------------
    torch.manual_seed(14)

    # --------------------------------------------------------
    # 모델 생성: TinyShapeCNN Backbone + 새 3클래스 Head
    # --------------------------------------------------------

    # 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
    #
    # [0] Conv2d(3, 8, 3, padding=1)  : (B,3,32,32) -> (B,8,32,32)
    # [1] BatchNorm2d(8)              : 채널별 정규화 (running 통계 보유)
    # [2] ReLU
    # [3] MaxPool2d(2)                : (B,8,32,32) -> (B,8,16,16)
    # [4] Conv2d(8, 16, 3, padding=1) : (B,8,16,16) -> (B,16,16,16)  <- 마지막 Conv
    # [5] BatchNorm2d(16)
    # [6] ReLU
    # [7] AdaptiveAvgPool2d((4, 4))   : (B,16,16,16) -> (B,16,4,4)
    # [8] Flatten                     : (B,16,4,4) -> (B,256)
    # [9] Linear(256, 3)              : (B,256) -> (B,3) 클래스별 logit  <- 새 Head
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
    )

    # --------------------------------------------------------
    # source Backbone 가중치 불러오기
    # --------------------------------------------------------

    # 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
    #
    # allow_pickle=False : 파이썬 객체 복원 금지 -> 수치 배열만 읽음 (임의 코드 실행 방지)
    # saved.files        : npz 안의 배열 이름 목록 -> 예) "0.weight", "1.running_mean", ...
    # .copy()            : npz 파일을 닫은 뒤에도 쓸 수 있도록 배열 복사
    # state.update(...)  : Backbone 키만 source 값으로 덮어씀, Head(9.*)는 seed 14 초기값 유지
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
        state = model.state_dict()
        state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
    model.load_state_dict(state)

    # --------------------------------------------------------
    # 학습 범위 설정 (requires_grad)
    # --------------------------------------------------------

    # 전체 파라미터를 먼저 고정 -> 기울기 계산·갱신 대상에서 제외
    for p in model.parameters():
        p.requires_grad = False

    # 새 Head만 학습 대상 -> Linear(256,3): weight 768 + bias 3 = 771개
    for p in model[9].parameters():
        p.requires_grad = True

    # partial : 마지막 Conv(model[4])도 추가로 학습
    # Conv2d(8,16,3, bias=False): 16 × 8 × 3 × 3 = 1,152개 -> Head 771 + 1,152 = 1,923개
    # 주의: BatchNorm(model[5])의 affine은 응용에서 열지 않음
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True

    # eval() : BatchNorm이 배치 통계 대신 source의 running 통계를 사용 -> BN 통계 고정
    # 주의: eval()은 gradient를 끄지 않음 (학습 여부는 requires_grad가 결정)
    model.eval()

    # --------------------------------------------------------
    # Optimizer와 손실 함수 설정
    # --------------------------------------------------------

    # torch.optim.AdamW(params, lr, weight_decay)
    #
    # params=[requires_grad=True인 파라미터] : 학습 범위에 든 파라미터만 전달
    # lr=0.003                               : 한 번 갱신할 때의 보폭(학습률)
    # weight_decay=0.0001                    : 가중치를 0 쪽으로 조금씩 줄이는 규제
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003, weight_decay=0.0001)

    # 클래스 가중치 = train 개수의 역수를 평균 1로 맞춘 값
    #
    # counts        : torch.bincount(y_train, minlength=3) -> [120, 40, 20]
    # inverse       : 1 / counts -> [0.00833, 0.025, 0.05]
    # class_weights : inverse / inverse.mean() -> [0.3, 0.9, 1.8]
    # 예: triangle 1장 오답의 손실이 circle 1장 오답보다 1.8 / 0.3 = 6배 크게 반영
    counts = torch.bincount(y_train, minlength=3).float()
    inverse = 1.0 / counts
    class_weights = inverse / inverse.mean()

    # nn.CrossEntropyLoss(weight=...)
    #
    # weight=class_weights : balance == "weight"일 때만 train 손실에 클래스 가중치 적용
    # weight=None          : 모든 샘플을 같은 비중으로 평균
    criterion = nn.CrossEntropyLoss(weight=class_weights if balance == "weight" else None)

    # 지금까지 가장 낮은 valid 손실 (첫 epoch에서 무조건 갱신되도록 무한대로 시작)
    best_loss = float("inf")
    # 학습 시간 측정 시작 (기기별로 달라지는 참고값)
    started = time.perf_counter()

    # --------------------------------------------------------
    # 학습 루프: 8 epoch, epoch마다 train 전체 1배치
    # --------------------------------------------------------
    for epoch in range(1, 9):

        # zero_grad() : 이전 epoch의 누적 기울기 초기화
        # backward()  : 손실을 학습 대상 파라미터로 미분
        # step()      : 계산된 기울기로 실제 가중치 갱신
        # 주의: train만 backward에 사용
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        # valid 손실 계산: 가중치 없는 기본 CrossEntropy (후보끼리 같은 기준으로 비교)
        # torch.no_grad() : 기울기 기록을 하지 않아 평가 중 그래프·메모리 절약
        with torch.no_grad():
            vl = nn.CrossEntropyLoss()(model(X_valid), y_valid).item()

        # strategy   : 학습 범위 (head / partial)
        # balance    : 불균형 대응 (none / weight)
        # epoch      : 1~8
        # valid_loss : 해당 epoch 직후 valid 손실
        all_history.append({"strategy":strategy,"balance":balance,"epoch":epoch,"valid_loss":vl})

        # valid 손실이 가장 낮은 epoch의 가중치를 보관
        # .detach().clone() : 이후 학습으로 값이 바뀌지 않도록 독립 복사본 저장
        if vl < best_loss:
            best_loss=vl
            best_epoch=epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}

    # 8 epoch 학습에 걸린 시간(초)
    elapsed = time.perf_counter()-started

    # --------------------------------------------------------
    # best epoch 가중치로 복원 후 valid 평가
    # --------------------------------------------------------
    model.load_state_dict(best_state)

    # model(X_valid) : (90, 3) logit
    # .softmax(1)    : 클래스 축으로 확률 변환 -> 각 행 합 1
    # .argmax(1)     : 확률이 가장 큰 클래스 번호
    #
    # 예: 확률 [0.10, 0.85, 0.05] -> 예측 1(square)
    with torch.no_grad():
        prob=model(X_valid).softmax(1)
        pred=prob.argmax(1)

    # output_dict=True : 클래스별 precision·recall·f1을 dict로 반환 -> report['2']['recall']
    # zero_division=0  : 분모가 0인 지표는 0으로 보고
    report=classification_report(y_valid,pred,labels=[0,1,2],output_dict=True,zero_division=0)

    # --------------------------------------------------------
    # 후보별 결과 1행 기록
    # --------------------------------------------------------

    # strategy        : 학습 범위 (head / partial)
    # balance         : 불균형 대응 (none / weight)
    # params          : 학습한 파라미터 수 -> head 771, partial 1923
    # best_epoch      : valid 손실이 가장 낮았던 epoch
    # accuracy        : valid 전체 정확도
    # macro_f1        : 클래스별 F1의 동일 가중 평균 (소수 클래스도 같은 비중)
    # triangle_recall : 실제 triangle(라벨 2) 중 triangle로 맞힌 비율
    # seconds         : 8 epoch 학습 시간 (기기별로 달라지는 참고값)
    rows.append({
        "strategy":strategy,
        "balance":balance,
        "params":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch":best_epoch,
        "accuracy":accuracy_score(y_valid,pred),
        "macro_f1":f1_score(y_valid,pred,labels=[0,1,2],average="macro",zero_division=0),
        "triangle_recall":report['2']['recall'],
        "seconds":elapsed,
    })


# ============================================================
# 6. 비교표·학습 이력 저장
# ============================================================

# metrics.csv : 세 후보의 valid 지표 3행 (seconds 포함)
# history.csv : 세 후보 × 8 epoch = 24행의 valid 손실 기록
pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
pd.DataFrame(all_history).to_csv(OUT/"history.csv",index=False)


# ============================================================
# 7. 마지막 후보(head+weight)의 valid 예측표 저장
# ============================================================

# 마지막 head+weight 후보의 예측을 설명용으로 저장합니다. 자동 최고 모델이라고 부르지 않습니다.
# 주의: 반복문이 끝난 뒤의 pred·prob는 마지막 후보 ("head","weight")의 값
# TODO A08: 실제 y_valid와 pred를 표에 넣으세요. valid.samples와 같은 순서입니다.
# BEGIN A08
# file       : valid.samples의 파일 이름 -> 예) "circle_000.png"
# actual     : 실제 라벨 번호 (0 circle, 1 square, 2 triangle)
# predicted  : 예측 라벨 번호
# confidence : 예측 클래스의 확률 (softmax 최댓값)
#
# prob.max(1)  : 행마다 최댓값과 위치를 함께 반환
# .values      : 최댓값(확률)만 선택
# .tolist()    : Tensor -> 파이썬 float 목록
#
# 예: prob 행 [0.02, 0.03, 0.95] -> predicted 2, confidence 0.95
result=pd.DataFrame({
    "file":[Path(path).name for path,_ in valid.samples],
    "actual":y_valid.tolist(),
    "predicted":pred.tolist(),
    "confidence":prob.max(1).values.tolist(),
})
# END A08
result.to_csv(OUT/"predictions.csv",index=False)

# 혼동행렬 3×3: 행=실제, 열=예측, 순서 circle·square·triangle 고정
# 예: [triangle 행, circle 열] = 3 -> 실제 triangle 3장을 circle로 예측
pd.DataFrame(confusion_matrix(y_valid,pred,labels=[0,1,2]),index=valid.classes,columns=valid.classes).to_csv(OUT/"confusion.csv")


# ============================================================
# 8. 실제 오답 이미지 저장 (있을 때만)
# ============================================================

predictions=pred
probabilities=prob

# 이전 실행에서 남은 오답 이미지 삭제 -> 이번 실행의 오답만 남김
for old in OUT.glob("project_error_*.png"):
    old.unlink()

# (pred != y_valid) : 오답 위치 True
# .nonzero()        : True 위치 인덱스 -> (k, 1)
# .flatten()        : (k,) 1차원으로 펼침
# .tolist()         : 파이썬 int 목록 -> 예) [65, 72]
# 주의: 오답이 없으면 빈 목록 -> 이미지를 만들지 않음
errors=(pred!=y_valid).nonzero().flatten().tolist()

# 최대 8장 저장: Tensor (3,32,32) [0,1] -> (32,32,3) 0~255 uint8 -> RGB를 BGR로 바꿔 PNG 저장
for i in errors[:8]:
    rgb=np.uint8(X_valid[i].permute(1,2,0).numpy()*255)
    cv2.imwrite(str(OUT/f"project_error_{i:03d}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))


# ============================================================
# 9. Grad-CAM: 선택 이미지의 예측 클래스 logit 반응 지도
# ============================================================

# 오답이 있으면 첫 오답, 없으면 valid 0번 이미지(circle_000.png) 사용
index=errors[0] if errors else 0

# --------------------------------------------------------
# Sequential을 features 앞뒤로 나누어 순전파
# --------------------------------------------------------

# hook 대신 Sequential을 둘로 나눕니다. 관심 클래스의 logit을 미분합니다.
#
# image      : (1, 3, 32, 32), requires_grad_(True)로 미분 그래프 연결
# features   : model[:7] (Conv~ReLU) 출력 -> (1, 16, 16, 16)
# logits_one : model[7:] (Pool~Head) 출력 -> (1, 3)
# target     : 이 이미지의 예측 클래스 번호
image = X_valid[index:index + 1].clone().requires_grad_(True)
features = model[:7](image)
logits_one = model[7:](features)
target = int(logits_one.argmax(1).item())

# torch.autograd.grad(outputs, inputs)
#
# outputs=logits_one[0, target] : 미분할 scalar (선택 클래스 logit 하나)
# inputs=features               : 기울기를 구할 대상 Tensor
# [0]                           : 반환 tuple의 첫 원소 -> (1, 16, 16, 16)
gradient = torch.autograd.grad(logits_one[0, target], features)[0]

# --------------------------------------------------------
# 채널 가중치 × features 합 -> ReLU -> 32×32 확대 -> [0,1] 정규화
# --------------------------------------------------------

# weights : 채널별 기울기 공간 평균 -> (1, 16, 1, 1)
# cam     : 채널 가중합 후 양의 반응만 남김 -> (1, 16, 16)
weights = gradient.mean(dim=(2, 3), keepdim=True)
cam = (weights * features).sum(dim=1).relu()

# interpolate(..., size=(32, 32), mode="bilinear") : 16×16 지도를 입력 좌표 32×32로 확대
# [0, 0] : 배치·채널 축 제거 -> (32, 32)
cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=(32, 32), mode="bilinear", align_corners=False)[0, 0]

# .detach() : 미분 그래프에서 분리
# .numpy()  : Tensor -> NumPy 배열
cam = cam.detach().numpy()

# 최댓값으로 나누어 [0,1]로 맞춤
# 양의 반응이 없으면 0 지도를 유지
# 주의: 1e-8 하한으로 0 나누기 방지 (빈 지도를 억지로 키우지 않음)
cam = cam / max(float(cam.max()), 1e-8)

# --------------------------------------------------------
# 열지도 색 입히기 및 원본과 합성
# --------------------------------------------------------

# rgb      : (32,32,3) uint8 원본 이미지
# heat_bgr : cam 0~255를 JET 색상으로 변환 (BGR, 빨강일수록 반응 큼)
rgb = (X_valid[index].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
heat_bgr = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)

# cv2.addWeighted(src1, alpha, src2, beta, gamma)
#
# src1=원본 BGR : 같은 32×32 좌표의 원본
# alpha=0.6     : 원본 비중
# src2=heat_bgr : 열지도
# beta=0.4      : 열지도 비중
# gamma=0       : 결과에 더할 상수 밝기
overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, heat_bgr, 0.4, 0)


# ============================================================
# 10. CAM 저장 및 결과 출력
# ============================================================

# 보기 쉽게 256×256으로 확대해 저장
cv2.imwrite(str(OUT/"project_cam.png"),cv2.resize(overlay,(256,256)))

# 세 후보 비교표 (seconds 제외)
print(pd.DataFrame(rows).drop(columns="seconds").round(4).to_string(index=False))
# 예측 행 90 = valid 전체, test는 어떤 단계에서도 읽지 않음
print("예측 행:",len(result),"오답:",len(errors),"test 사용: False")
