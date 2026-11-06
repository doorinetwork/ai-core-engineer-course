"""실습 03. 정상 라벨의 면적·비율·클래스 수와 그룹 분리를 확인합니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import pandas as pd

# ============================================================
# 2. manifest 읽기
# ============================================================
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
# manifest.csv : 이미지 60장의 목록. 열 의미
#   image_id : 이미지 이름 / split : train·val·test / group_id : 같은 원본 묶음 번호
#   image, label : data/ 기준 상대 경로 / width, height : 160, 128
#   objects : 라벨 객체 수(0 또는 3) / sha256 : 파일 해시(완전 중복 확인용)
manifest=pd.read_csv(BASE/"data/manifest.csv")
rows=[]

# ============================================================
# 3. 라벨 줄마다 면적·가로세로비 기록 (TODO A03)
# ============================================================
for image in manifest.itertuples():
    # 빈 TXT면 splitlines() -> [] -> 이 이미지는 행이 생기지 않음 (정상 negative)
    for line in (BASE/"data"/image.label).read_text(encoding="utf-8").splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        # >>> A03
        # 요구: 정규화 면적 w*h와 원본 pixel 기준 가로세로비를 기록하세요.
        # 기록하는 dict 키 의미
        #   image_id, split, group_id : 어느 이미지·분할·그룹의 박스인지
        #   class_id : 0=square, 1=circle, 2=triangle (float -> int)
        #   area     : 정규화 면적 w*h (이미지 전체 대비 비율). 예: 0.19375*0.2421875 = 0.0469
        #   aspect   : 픽셀 가로세로비 (w*W)/(h*H). 예: (0.19375*160)/(0.2421875*128) = 31/31 = 1.0
        # 주의: 정규화 값끼리 w/h로 나누면 160/128 비율이 빠져 1.25배 틀어진 값이 된다.
        rows.append(dict(image_id=image.image_id,split=image.split,group_id=image.group_id,class_id=int(cls),
                         area=w*h,aspect=(w*image.width)/(h*image.height)))
        # <<< A03

# ============================================================
# 4. 분할별 클래스 수·그룹 중복·파일 중복 확인
# ============================================================
r=pd.DataFrame(rows);r.to_csv(out/"label_audit.csv",index=False)
# groupby(["split","class_id"]).size() : (분할, 클래스)별 박스 수 -> unstack으로 클래스를 열로 펼침
print(r.groupby(["split","class_id"]).size().unstack(fill_value=0).to_string())
# 그룹 하나가 2개 이상 split에 나타나면 누수 -> 기대 0
print("split 간 그룹 중복:", int((manifest.groupby("group_id").split.nunique()>1).sum()))
# sha256.duplicated() : 앞에서 이미 나온 해시면 True -> 완전히 같은 파일 수. 기대 0
print("정확히 같은 파일:", int(manifest.sha256.duplicated().sum()))
print("빈 라벨은 정상 negative 이미지이며 누락 라벨과 다릅니다.")
