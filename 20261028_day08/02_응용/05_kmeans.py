"""실습 05 응용: train 수치2열만 결측 대체·표준화하여 K=3으로 묶습니다."""
# -----------------------------------------------------------------------------
# 라이브러리 불러오기
# -----------------------------------------------------------------------------
# 경로를 다루기 위해 Python 표준 라이브러리 pathlib에서 Path를 가져온다.
from pathlib import Path
# 데이터 처리를 위해 외부 라이브러리 Pandas를 import한다.
import pandas as pd
# 배열 생성 및 수치 연산을 위해 외부 라이브러리 numpy를 가져온다.
import numpy as np

# -----------------------------------------------------------------------------
# 데이터 읽기
# -----------------------------------------------------------------------------
# HERE에 현재의 Python 파일이 존재하는 경로를 저장한다.
HERE = Path(__file__).resolve().parent
# 이 Python 파일이 있는 경로에 있는 requests.csv 파일을 읽어 raw에 저장한다. 인덱스는 request_id로 정한다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# data/split_manifest.csv에 정의된 data 분할 규칙과 position 순서를 읽어 manifest에 저장한다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# -----------------------------------------------------------------------------
# 데이터 분할
# -----------------------------------------------------------------------------
# train에 해당하는 request_id를 읽어 position순으로 정렬해 train_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
# valid에 해당하는 request_id를 읽어 position순으로 정렬해 valid_ids에 저장한다. 분할표는 manifest에 저장된 규칙을 따른다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
# raw에서 train_ids와 valid_ids에 해당하는 행만 꺼내 각각 train, valid에 저장한다.
# train, valid (type: DataFrame): 각각 60행, 20행이다. copy()로 원본 raw와 분리된 복사본을 만들어, 이후 수정이 raw에 영향을 주지 않게 한다.
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
# 모델 입력에 쓸 열들을 FEATURES에 리스트로 저장한다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# 수치형 2열과 범주형 2열로 FEATURES를 분리한다.
NUM, CAT = FEATURES[:2], FEATURES[2:]
# train과 valid에서 입력 열만 골라 X_train, X_valid에 저장한다.
# X_train (type: DataFrame, 60행×4열), X_valid (type: DataFrame, 20행×4열): 모델에 들어갈 입력 표이다.
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
# train과 valid에서 정답 열 failed를 꺼내 y_train, y_valid에 저장한다.
# y_train, y_valid (type: Series[int]): 1은 실패, 0은 성공이며, 실패(1)가 찾고 싶은 양성 클래스이다.
y_train, y_valid = train["failed"], valid["failed"]

# -----------------------------------------------------------------------------
# 출력 경로 지정
# -----------------------------------------------------------------------------
# 현재의 Python 파일이 존재하는 경로 아래에 outputs 디렉토리를 만드는 경로를 OUT에 저장한다.
OUT = HERE / "outputs"
# 디렉토리가 이미 존재한다면 무시한다.
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 군집화 도구 불러오기
# -----------------------------------------------------------------------------
# 전처리와 군집화를 한 흐름으로 묶기 위해 외부 라이브러리 scikit-learn에서 Pipeline을 가져온다.
from sklearn.pipeline import Pipeline
# 결측치를 채우기 위해 외부 라이브러리 scikit-learn에서 SimpleImputer를 가져온다.
# 기능: 빈 칸을 열마다 정한 대표값(여기서는 중앙값)으로 채운다. KMeans는 NaN이 있으면 거리를 계산하지 못해 에러를 낸다.
from sklearn.impute import SimpleImputer
# 열마다 숫자 크기를 맞추기 위해 외부 라이브러리 scikit-learn에서 StandardScaler를 가져온다.
# 기능: 열마다 평균 0, 표준편차 1로 바꾼다. 글자 수(수천 단위)와 도구 수(한 자리)를 그대로 두면 거리가 글자 수에만 좌우된다.
from sklearn.preprocessing import StandardScaler
# 비슷한 행끼리 묶기 위해 외부 라이브러리 scikit-learn에서 KMeans를 가져온다.
# 기능: 중심점 K개를 두고, 각 행을 가장 가까운 중심에 배정한 뒤 중심을 배정된 행들의 평균으로 옮기는 일을 반복한다.
from sklearn.cluster import KMeans
# 군집이 얼마나 잘 나뉘었는지 재기 위해 외부 라이브러리 scikit-learn에서 silhouette_score를 가져온다.
# 기능: 각 행이 자기 군집과 가깝고 다른 군집과 먼 정도를 -1~1 사이 점수로 계산해 평균을 낸다. 1에 가까울수록 잘 나뉜 것이다.
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# 군집화 모델
# -----------------------------------------------------------------------------
# cluster: 중앙값 결측 대체 → 표준화 → KMeans 3단계 흐름이다. 아직 fit 전이라 중앙값·평균·중심점은 정해지지 않았다.
# strategy="median": 평균보다 극단값에 덜 흔들리는 중앙값으로 빈 칸을 채운다.
# "kmeans": n_clusters=3: 요청들을 3개 묶음으로 나눈다(교안이 정한 K=3이다).
#           n_init=10: 시작 중심점을 서로 다르게 10번 뽑아 학습하고, 묶음 안 거리 합(inertia)이 가장 작은 결과를 고른다. 
#                      시작점을 잘못 뽑아 나쁜 군집에 머무는 일을 줄인다.
#           random_state=42: 시작 중심점을 뽑는 난수를 고정해, 다시 실행해도 같은 군집 결과가 나오게 한다.
cluster = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=42))])

# -----------------------------------------------------------------------------
# 군집 학습
# -----------------------------------------------------------------------------
# 비지도학습이므로 정답 라벨 없이 train의 두 수치형 열만 넘긴다.
# fit에 y를 주지 않는 이유: KMeans는 비지도 학습이라 정답 없이 입력의 생김새만으로 묶는다. 
#                        failed를 넣으면 "실패 여부로 나눈 묶음"이 되어 비지도 군집의 뜻이 사라진다.
# elapsed_ms를 넣지 않는 이유: 처리 시간은 요청이 끝난 뒤에야 알 수 있는 사후 정보라서, 요청을 받는 시점에는 쓸 수 없다.
# train으로만 fit하는 이유: 중앙값·평균·표준편차·중심점이 모두 train에서만 정해져야 valid 정보가 섞이지 않는다.
cluster.fit(X_train[NUM])

# -----------------------------------------------------------------------------
# 군집 할당 결과 및 중심점 프로파일 추출
# -----------------------------------------------------------------------------
# labels: train 각 행이 배정된 군집 번호(0, 1, 2)이다. 순서는 X_train의 행 순서, 즉 train_ids 순서와 같다.
labels = cluster["kmeans"].labels_
# 실루엣 점수 계산을 위해 군집기 직전까지의 전처리 단계(대체 + 표준화)를 거친 표준화 좌표 배열 z를 얻는다.
# z: 마지막 단계(KMeans)를 뺀 앞 두 단계(결측 대체·표준화)만 적용한 값이다.
# cluster[:-1]: Pipeline을 잘라 마지막 단계 직전까지만 쓴다. KMeans가 실제로 거리를 잰 공간이 이 표준화 공간이므로, 실루엣도 같은 공간에서 계산해야 한다.
z = cluster[:-1].transform(X_train[NUM])
# centers: 군집 중심 3개를 원래 단위(글자 수, 도구 수)로 되돌린 값이다.
# inverse_transform을 쓰는 이유: cluster_centers_는 표준화된 값(평균 0 기준 몇 표준편차인지)이라 사람이 읽기 어렵다. 
#                              되돌리면 "평균 글자 수 약 몇 자" 식으로 해석할 수 있다.
centers = cluster["scaler"].inverse_transform(cluster["kmeans"].cluster_centers_)

# -----------------------------------------------------------------------------
# 결과 저장 및 출력
# -----------------------------------------------------------------------------
# profile: 군집마다 중심의 input_chars, allowed_tools 값을 담은 표이다.
profile = pd.DataFrame(centers, columns=NUM)
# 맨 앞(0번 위치)에 cluster 열을 넣어 각 행이 몇 번 군집인지 표시한다. range(3)은 0, 1, 2이다.
profile.insert(0, "cluster", range(3))
# n 열: 군집마다 배정된 train 행 수이다.
# np.bincount는 0, 1, 2가 각각 몇 번 나왔는지 센다. minlength=3은 어떤 군집에 행이 하나도 없어도 길이 3을 보장해 열 길이가 어긋나지 않게 한다.
profile["n"] = np.bincount(labels, minlength=3)
# 각 요청 ID별 할당된 군집 번호 표를 CSV 파일로 저장한다.
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
# 요청 id와 배정된 군집 번호를 짝지어 cluster_assignments.csv로 저장한다.
# train_ids와 labels는 둘 다 train_ids 순서이므로, 같은 위치끼리 짝지으면 행이 정확히 맞는다.
pd.DataFrame({"request_id": train_ids, "cluster": labels}).to_csv(OUT / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
# 군집 요약표를 소수점 셋째 자리까지 반올림해 행 번호 없이 출력한다.
print(profile.round(3).to_string(index=False))
# 표준화 공간 z에서 계산한 실루엣 점수를 소수점 셋째 자리까지 출력한다.
print("silhouette(표준화 공간):", round(silhouette_score(z, labels), 3))
print("군집 번호에는 실패·성공·우열의 뜻이 없습니다.")
