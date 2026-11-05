# ============================================================
# 1. 경로·환경 설정
# ============================================================
# 경로를 운영체제와 무관하게 다루는 Path 클래스
from pathlib import Path
# sys: 모듈 검색 경로 추가 / json: 결과 기록용(이 파일에서는 미사용)
import sys,json
# 수치 배열 (이 파일에서는 미사용, 공통 골격)
import numpy as np
# 비교표·history CSV 저장
import pandas as pd
# PyTorch: CPU 스레드 고정
import torch

# BASE : 이 main.py가 있는 폴더
BASE=Path(__file__).resolve().parent

# 제공 모듈(day14_vision/)을 import할 수 있게 검색 경로에 추가
sys.path.insert(0,str(BASE.parent/"day14_vision"))

# DATA : 03_심화/data / OUT : 이 폴더의 outputs
DATA=BASE.parent/"data"
OUT=BASE/"outputs"
OUT.mkdir(exist_ok=True)

# CPU 1-thread 고정 -> 재현성 확보
torch.set_num_threads(1)

# experiment : balance 인자로 불균형 대응 방식 선택 ('none'/'weight'/'sampler')
from engine import experiment,datasets,metrics
# 제공 모델 함수 (이 실습에서는 import만 함)
from models import build_model,configure

# ============================================================
# 2. 불균형 대응 후보 정의 (TODO D04)
# ============================================================
# train 분포 : circle 120 / square 40 / triangle 20 (6:2:1)
# valid 분포 : 30 / 30 / 30 (원래 분포 그대로 평가)
rows=[]
# TODO D04: none, weight, sampler를 각각 독립 후보로 비교하세요. 동시에 적용하지 않습니다.
# BEGIN D04
# 후보 목록
# - none    : 기준선. 보정 없음
# - weight  : CrossEntropyLoss(weight=...)로 소수 클래스 손실을 크게 반영 (train 손실에만)
# - sampler : WeightedRandomSampler로 train 배치를 클래스 균형에 가깝게 재추출
# 주의: weight와 sampler를 한 후보에 동시에 적용하지 않음 -> 효과가 이중 보정됨
methods=["none","weight","sampler"]
# END D04

# ============================================================
# 3. 후보별 학습
# ============================================================
for method in methods:
    # --------------------------------------------------------
    # 같은 head 전략·8 epoch 예산으로 학습
    # --------------------------------------------------------
    # balance=method : 이번 후보의 불균형 대응 방식
    # epochs=8       : 전체 train(또는 같은 개수만큼 재추출)을 8번 학습
    _,summary,history,_=experiment(DATA,balance=method,epochs=8)
    rows.append(summary)

    # history 열
    # - train_objective : 학습 손실 (weight 후보는 가중 손실이라 valid_loss와 같은 척도가 아님)
    # - valid_loss      : 비가중 valid 평균 손실
    # - draw_circle/square/triangle : 그 epoch에 실제로 뽑힌 클래스별 이미지 수
    #   예: none 120/40/20, sampler 57/60/63처럼 균형에 가까워짐
    history.to_csv(OUT/f"{method}_history.csv",index=False)

# ============================================================
# 4. 비교표 저장 및 출력
# ============================================================
# imbalance_comparison.csv : 후보별 accuracy·macro_f1·클래스별 recall·loss·best_epoch·seconds
result=pd.DataFrame(rows)
result.to_csv(OUT/"imbalance_comparison.csv",index=False)
print(result[["balance","accuracy","macro_f1","triangle_recall"]].round(4).to_string(index=False))
print("sampler는 train만 재추출합니다. valid는 원래 분포 그대로입니다.")
