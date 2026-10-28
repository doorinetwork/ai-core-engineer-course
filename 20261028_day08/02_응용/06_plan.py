"""실습 06 응용: 04번에서 만든 CV 표만 읽어 선택하고 평가 계획을 기록합니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 결과 기록을 위해 Python 표준 라이브러리 json을 가져온다.
import json

# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent

# 현재의 Python 파일이 존재하는 경로 아래에 outputs 디렉토리를 만드는 경로를 OUT에 저장한다.
OUT = HERE / "outputs"

# -----------------------------------------------------------------------------
# CV 비교표 읽기
# -----------------------------------------------------------------------------
# 선택 기준은 CV_F1 하나뿐이다. valid_F1은 진단용 참고값이라 선택에 쓰면 valid가 사실상 학습에 쓰인 셈이 된다.
# 04_features.py에서 만든 feature_comparison.csv로 CV 비교표를 읽어 table에 저장한다.
# 각 모델 후보가 하나의 행이며, model(이름), CV_F1(5-fold 평균 F1), CV_std(fold 간 표준편차), valid_F1(검증 F1) 열을 가진다. 
# 행 순서는 Logistic, Logistic_FE이다.
table = pd.read_csv(OUT / "feature_comparison.csv")

# -----------------------------------------------------------------------------
# 후보 선택 
# -----------------------------------------------------------------------------
# table["CV_F1"].idxmax()로 CV_F1이 가장 큰 행의 인덱스(행 번호)를 구하고, loc[그 행, "model"]로 그 행의 모델 이름을 꺼내 selected에 저장한다.
# idxmax는 최댓값이 여러 개(정확한 동점)이면 가장 앞에 있는 행의 인덱스를 돌려주므로, "동점이면 표의 첫 후보(Logistic)"라는 규칙을 별도 코드 없이 그대로 지킨다.
# selected: 선택된 모델 이름이다. 현재 표에서는 CV_F1이 Logistic 0.5219, Logistic_FE 0.5514이므로 "Logistic_FE"가 된다.
selected = table.loc[table["CV_F1"].idxmax(), "model"]

# -----------------------------------------------------------------------------
# 분할 ID 목록 기록
# -----------------------------------------------------------------------------
# manifest: request_id, split(train/valid/test), position(분할 안에서의 순서) 열을 가진 분할표이다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train, valid, test 각각에 대해 해당 split의 행만 고르고, position순으로 정렬한 뒤, request_id만 리스트로 꺼내 ids에 저장한다.
# ids: 키는 "train", "valid", "test"이고 값은 각 분할의 request_id 목록(60개, 20개, 20개)이다.
# position순으로 정렬하는 이유: 같은 id라도 순서가 다르면 재학습 결과를 똑같이 재현할 수 없으므로, 순서까지 고정해 기록한다.
ids = {name: manifest.loc[manifest["split"] == name].sort_values("position")["request_id"].tolist()
       for name in ["train", "valid", "test"]}

# -----------------------------------------------------------------------------
# 평가 계획
# -----------------------------------------------------------------------------
# 최종 평가에 쓸 설정을 모두 dict로 모아 plan에 저장한다.
# plan (type: dict): 아래 키를 가진 평가 계획서이다.
# - "selected" (type: str): 위에서 고른 모델 이름이다.
# - "C" (type: int): 로지스틱 회귀의 규제 강도 역수이며, 04번 비교와 같은 값 1로 고정한다.
# - "max_iter" (type: int): 학습 반복 최대 횟수 1000이며, 최적화가 끝까지 수렴하도록 넉넉하게 준다.
# - "features" (type: list[str]): 요청 시점에 알 수 있는 원본 입력 열 4개이다.
# - "derived" (type: bool): selected가 "Logistic_FE"이면 True로, 파생 특성 2개를 추가로 만들어야 한다는 표시이다.
# - "threshold" (type: float): 0.5이며, 예측 확률을 실패(1)/성공(0)으로 바꾸는 기준값이다.
# - "rule" (type: str): "score > threshold"로, 확률이 정확히 0.5이면 양성으로 보지 않는다는 비교 방향까지 고정한다.
# - "positive_label" (type: int): 1(실패)을 찾고 싶은 양성 클래스로 정한다.
# - "selection" (type: str): 선택 근거(train60 5-fold 평균 F1, 정확한 동점은 Logistic 우선)를 글로 남긴다.
# - "split_ids" (type: dict): 위에서 만든 분할별 id 목록으로, 누가 어느 분할에 속했는지 기록한다.
plan = {"selected": selected, "C": 1, "max_iter": 1000,
        "features": ["input_chars", "allowed_tools", "task_type", "priority"],
        "derived": selected == "Logistic_FE", "threshold": 0.5,
        "rule": "score > threshold", "positive_label": 1,
        "selection": "train60 5-fold mean F1; exact tie: Logistic first", "split_ids": ids}

# -----------------------------------------------------------------------------
# 계획 저장 및 출력
# -----------------------------------------------------------------------------
# plan을 JSON 문자열로 바꿔 outputs/selection.json에 UTF-8로 저장한다.
# ensure_ascii=False는 한글 등 비ASCII 글자를 \uXXXX로 바꾸지 않고 그대로 쓰게 하고, indent=2는 2칸 들여쓰기로 사람이 읽기 쉽게 한다.
# encoding="utf-8"은 운영체제 기본 인코딩과 상관없이 같은 파일이 만들어지게 한다.
(OUT / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
# 선택된 모델 이름을 화면에 출력한다.
print("선택:", selected)
# 고정한 설정을 화면에 요약 출력한다. 개발80(train60+valid20)으로 다시 학습하고, test20은 마지막에 한 번만 채점한다는 뜻이다.
print("설정: C=1 / 임계값 0.5 초과 / 개발80으로 재학습 / test20 최종 확인")
print("저장: outputs/selection.json (최종평가 이후 변경하지 않습니다.)")
