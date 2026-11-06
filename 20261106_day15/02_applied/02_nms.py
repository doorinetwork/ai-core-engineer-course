"""실습 02. 고전 NMS 원리 실험. YOLO26 기본 추론에 NMS가 있다고 가정하지 않습니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou, batched_nms

# ============================================================
# 2. 입력을 Tensor로 준비
# ============================================================
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
# p : 손으로 정한 후보 4개(boxes_fixture.csv), g : 정답 3개(boxes_gt.csv)
p=pd.read_csv(BASE/"data/boxes_fixture.csv");g=pd.read_csv(BASE/"data/boxes_gt.csv")
# boxes   : 후보 xyxy, shape [4, 4] float32
# scores  : 후보 confidence [0.95, 0.85, 0.80, 0.40]
# classes : 후보 클래스 [0, 0, 1, 2] (int64, batched_nms의 그룹 번호)
# gb      : 정답 xyxy, shape [3, 4]
boxes=torch.tensor(p[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
scores=torch.tensor(p.confidence.to_numpy(),dtype=torch.float32)
classes=torch.tensor(p.class_id.to_numpy(),dtype=torch.int64)
gb=torch.tensor(g[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
rows=[]

# ============================================================
# 3. (conf, NMS IoU) 조건 3개 비교
# ============================================================
# (0.25, 0.5) : 기준 / (0.5, 0.5) : conf만 올림 / (0.25, 0.95) : NMS IoU만 올림
for conf,nms_iou in [(0.25,0.5),(0.5,0.5),(0.25,0.95)]:
    # ids : confidence >= conf 인 후보의 원본 index. 예: conf 0.5 -> tensor([0, 1, 2])
    ids=torch.where(scores>=conf)[0]
    # >>> A02
    # 요구: 같은 클래스끼리만 중복을 제거하는 batched_nms를 적용하세요.
    # 반환 index는 ids로 줄인 배열 기준이므로 다시 원본 index로 바꿉니다.
    # batched_nms(boxes, scores, idxs, iou_threshold) : 같은 idxs(클래스) 안에서만 IoU > 임계값인 낮은 점수 후보 제거
    #   boxes[ids]   : conf를 통과한 후보 박스
    #   scores[ids]  : 그 후보들의 점수
    #   classes[ids] : 그 후보들의 클래스 (다른 클래스끼리는 겹쳐도 억제하지 않음)
    #   nms_iou      : 억제 기준 IoU (0.5 또는 0.95)
    # 반환값은 "ids로 줄인 배열" 안의 위치 -> ids[...]로 원본 index로 변환
    # 예: conf 0.25, IoU 0.5 -> 후보 0·1(square)이 서로 IoU 0.90으로 겹침(교집합 1368 / 합집합 1520) -> 1 제거 -> keep [0, 2, 3]
    # 주의: IoU 0.95면 0.90 < 0.95라 억제되지 않아 4개 모두 남음
    keep=ids[batched_nms(boxes[ids],scores[ids],classes[ids],nms_iou)]
    # <<< A02

    # --------------------------------------------------------
    # 남은 후보를 정답과 IoU 0.5 일대일 매칭 (NMS와 매칭은 별개 단계)
    # --------------------------------------------------------
    used=set();tp=0
    for i in keep.tolist():
        # box_iou(boxes1, boxes2) : [1, 3] IoU 표 -> [0]으로 정답 3개에 대한 IoU 1차원
        ovs=box_iou(boxes[[i]],gb)[0]
        # eligible : 같은 클래스이면서 아직 쓰지 않은 정답 index
        eligible=[j for j in range(len(g)) if int(classes[i])==int(g.iloc[j].class_id) and j not in used]
        # IoU가 큰 정답부터 확인해 0.5 이상이면 연결 후 중단
        for j in sorted(eligible,key=lambda j:-float(ovs[j])):
            if ovs[j]>=0.5:used.add(j);tp+=1;break
    # kept : NMS 후 남은 후보 수 / fp = kept - tp / fn = 정답 3 - tp
    n=len(keep);fp=n-tp;fn=len(g)-tp
    rows.append(dict(conf=conf,nms_iou=nms_iou,kept=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0,recall=tp/len(g)))

# ============================================================
# 4. 비교표 저장
# ============================================================
# nms_comparison.csv 열 : conf, nms_iou, kept, tp, fp, fn, precision, recall
r=pd.DataFrame(rows);r.to_csv(out/"nms_comparison.csv",index=False);print(r.round(4).to_string(index=False))
