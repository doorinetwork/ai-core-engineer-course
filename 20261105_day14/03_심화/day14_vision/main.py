# ============================================================
# 1. 라이브러리 및 경로 설정
# ============================================================

# 경로를 운영체제와 무관하게 다루는 Path 클래스
from pathlib import Path
# metrics.json·cam_index.json 저장
import json
# 수치 배열 (이 파일에서는 미사용, 공통 골격)
import numpy as np
# 비교표·history·예측표 CSV 저장
import pandas as pd
# PyTorch: 스레드 고정·모델 저장
import torch
# classification_report: 클래스별 지표 / confusion_matrix: 혼동행렬
from sklearn.metrics import classification_report,confusion_matrix
# 제공 학습 골격: 후보 1개 학습·평가
from engine import experiment
# 제공 설명 도구: 실제 오답 타일·Grad-CAM 저장
from explain import save_explanations

# BASE : 이 폴더 / DATA : 자체 포함 data / OUT : 직접 생성 결과
BASE=Path(__file__).resolve().parent;DATA=BASE/"data";OUT=BASE/"outputs"

# CPU 1-thread 고정 -> 재현성 확보
OUT.mkdir(exist_ok=True);torch.set_num_threads(1)


# ============================================================
# 2. 후보 정의 (TODO D07)
# ============================================================

# TODO D07: 같은 8회 예산에서 네 후보를 명시하세요. Scratch는 비교표의 별도 보조 후보입니다.
# BEGIN D07
# (strategy, balance) 후보 4개 - 이 순서가 동률일 때의 우선순위
# - ("head","none")     : Head만 학습, 보정 없음 (기준선)
# - ("partial","none")  : 마지막 Conv+BN affine+Head 학습
# - ("head","weight")   : Head만 학습 + 클래스 가중 손실
# - ("head","sampler")  : Head만 학습 + 균형 재추출
# 주의: weight와 sampler는 별개 후보로 두고 동시에 적용하지 않음
experiments=[("head","none"),("partial","none"),("head","weight"),("head","sampler")]
# END D07


# ============================================================
# 3. 후보별 학습 및 최선 후보 선택
# ============================================================

rows=[];histories=[];best_key=None
for strategy,balance in experiments:
    # --------------------------------------------------------
    # 후보 학습 (같은 seed·분할·8 epoch 예산)
    # --------------------------------------------------------
    # strategy=strategy : 학습 범위
    # balance=balance   : 불균형 대응 방식
    # epochs=8          : 8회 예산
    model,summary,history,prediction=experiment(DATA,strategy=strategy,balance=balance,epochs=8)
    rows.append(summary)
    # candidate 열 : 후보 이름  예: head_sampler
    history["candidate"]=strategy+"_"+balance
    histories.append(history)

    # TODO D08: Macro F1을 최대화하고 동률이면 loss가 작은 후보를 고르세요. test는 보지 않습니다.
    # BEGIN D08
    # --------------------------------------------------------
    # 선택 키 (튜플은 앞 원소부터 비교)
    # --------------------------------------------------------
    # 1순위 summary["macro_f1"] : 클수록 좋음
    # 2순위 -summary["loss"]    : loss가 작을수록 -loss가 커짐 -> 큰 값 우선
    # 예: (1.0, -0.0126) < (1.0, -0.0065) -> loss 0.0065 후보 선택
    # 3순위 후보 순서 : 아래 key>best_key가 엄격 비교라 완전 동률이면 먼저 나온 후보 유지
    key=(summary["macro_f1"],-summary["loss"])
    # END D08

    # 더 좋은 후보면 요약·예측·가중치를 CPU 복사본으로 보관
    if best_key is None or key>best_key:
        best_key=key;chosen=summary;chosen_pred=prediction;chosen_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


# ============================================================
# 4. 비교표·history·선택 모델 결과 저장
# ============================================================

# metrics.csv : 후보 4행 (accuracy·macro_f1·클래스별 recall·loss·best_epoch·trainable·seconds 등)
pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
# history.csv : 후보 4개 × 8 epoch = 32행, candidate 열로 구분
pd.concat(histories,ignore_index=True).to_csv(OUT/"history.csv",index=False)

# 마지막 후보 모델 객체에 선택 가중치를 올림 (구조가 같으므로 가능)
model.load_state_dict(chosen_state);model.eval()

# predictions.csv : 선택 후보의 valid 90행 예측
chosen_pred.to_csv(OUT/"predictions.csv",index=False)

# 클래스별 precision·recall·f1·support (target_names는 labels 순서와 같게)
report=classification_report(chosen_pred.actual,chosen_pred.predicted,labels=[0,1,2],target_names=["circle","square","triangle"],output_dict=True,zero_division=0)

# metrics.json
# - selected  : 선택 후보 요약
# - report    : 클래스별 리포트
# - test_used : False (test는 선택에 쓰지 않음)
(OUT/"metrics.json").write_text(json.dumps({"selected":chosen,"report":report,"test_used":False},indent=2,ensure_ascii=False),encoding="utf-8")

# confusion.csv : 행=실제, 열=예측 (0 circle, 1 square, 2 triangle)
pd.DataFrame(confusion_matrix(chosen_pred.actual,chosen_pred.predicted,labels=[0,1,2])).to_csv(OUT/"confusion.csv",index=False)

# 실제 오답 타일·CAM 저장, cam_index.json에 출처·target class 기록
explanations=save_explanations(DATA,OUT,model,chosen_pred)
(OUT/"cam_index.json").write_text(json.dumps(explanations,ensure_ascii=False,indent=2),encoding="utf-8")


# ============================================================
# 5. 선택 모델 저장 (TODO D09)
# ============================================================

# TODO D09: 선택 가중치와 클래스 매핑·입력 계약을 함께 저장하세요.
# BEGIN D09
# 클래스 매핑은 학습에 쓴 ImageFolder에서 그대로 가져옴 -> {'circle':0,'square':1,'triangle':2}
from engine import datasets
class_to_idx=datasets(DATA)[0].class_to_idx

# best_model.pt 내용
# - model        : 선택 후보 state_dict (CPU 텐서)
# - class_to_idx : 클래스 이름 -> 번호 계약
# - input        : 입력 계약 (RGB, Resize 32×32, ToTensor float [0,1], 정규화 없음)
# - architecture : 복원 시 같은 구조를 만들기 위한 이름
# - selected     : 선택 후보 요약 (strategy·balance·macro_f1·loss 등)
# - test_used    : False
# 주의: restore.py는 weights_only=True로 읽으므로 텐서·dict·list·str·숫자·bool만 넣음
torch.save({"model":chosen_state,
            "class_to_idx":class_to_idx,
            "input":{"color":"RGB","size":[32,32],"transform":"Resize((32,32)) -> ToTensor()","range":"float [0,1]","normalize":"none"},
            "architecture":"TinyShapeCNN_3-8-16_GAP4 (models.build_model)",
            "selected":chosen,
            "test_used":False},OUT/"best_model.pt")
# END D09


# ============================================================
# 6. 결과 출력
# ============================================================

# seconds는 실행 환경마다 달라지므로 출력표에서 제외
print(pd.DataFrame(rows).drop(columns="seconds").round(4).to_string(index=False))
# 예: 선택: head sampler test 사용: False
print("선택:",chosen["strategy"],chosen["balance"],"test 사용: False")
