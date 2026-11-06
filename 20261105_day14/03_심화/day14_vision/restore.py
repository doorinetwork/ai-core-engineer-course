# ============================================================
# 1. 라이브러리 및 경로 설정
# ============================================================

# 경로를 운영체제와 무관하게 다루는 Path 클래스
from pathlib import Path
# 확률 배열 비교(allclose, array_equal)
import numpy as np
# 저장된 predictions.csv 읽기
import pandas as pd
# PyTorch: 모델 파일 읽기
import torch
# valid 배치 공급
from torch.utils.data import DataLoader
# 제공 데이터셋·평가 함수
from engine import datasets,evaluate
# 제공 모델 구조 생성
from models import build_model

# BASE : 이 폴더 / DATA : 자체 포함 data / OUT : main.py가 만든 결과
BASE=Path(__file__).resolve().parent;DATA=BASE/"data";OUT=BASE/"outputs"
torch.set_num_threads(1)


# ============================================================
# 2. 저장 파일 읽기 및 구조 생성
# ============================================================

# 직접 생성한 파일만 읽습니다. 임의 출처의 직렬화 파일을 로드하지 않습니다.
# map_location="cpu" : GPU에서 저장했어도 CPU로 읽음
# weights_only=True  : 텐서·기본 자료형만 허용 (임의 객체 실행 차단)
saved=torch.load(OUT/"best_model.pt",map_location="cpu",weights_only=True)

# 같은 구조(source Backbone + 새 Head)를 먼저 만듦 -> 아래에서 가중치를 덮어씀
model=build_model(DATA)


# ============================================================
# 3. 가중치 복원 (TODO D10)
# ============================================================

# TODO D10: model.load_state_dict로 저장 가중치를 복원하세요.
# BEGIN D10
# saved["model"] : main.py D09에서 저장한 선택 후보 state_dict
# load_state_dict : 키·모양이 모두 맞아야 성공 (strict=True 기본)
model.load_state_dict(saved["model"])
# END D10


# ============================================================
# 4. 새 프로세스에서 다시 평가
# ============================================================

# eval() : BN running 통계 사용
model.eval();_,valid=datasets(DATA)

# batch_size=32, shuffle=False : 저장 당시와 같은 순서로 평가
summary,y,prob=evaluate(model,DataLoader(valid,batch_size=32,shuffle=False))


# ============================================================
# 5. 저장 당시 예측과 비교
# ============================================================

# main.py가 저장한 선택 후보 예측표
expected=pd.read_csv(OUT/"predictions.csv")

# 현재 valid 이미지 경로 목록 (data/ 기준 상대 경로)
paths=[str(Path(p).relative_to(DATA)) for p,_ in valid.samples]

# ID 순서     : 파일 순서가 같은지
# 확률 일치   : atol=1e-6, rtol=1e-5 범위 안에서 p0~p2가 같은지
# 예측 일치   : argmax 클래스가 완전히 같은지
print("ID 순서:",paths==expected.path.tolist())
print("확률 일치:",np.allclose(prob,expected[["p0","p1","p2"]].to_numpy(),atol=1e-6,rtol=1e-5))
print("예측 일치:",np.array_equal(prob.argmax(1),expected.predicted.to_numpy()))
