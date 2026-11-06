"""실습 04. 무작위 초기화 YOLO26의 1 epoch 학습 연결 점검. 성능 실험이 아닙니다."""
# ============================================================
# 1. 라이브러리 불러오기
# ============================================================
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# ============================================================
# 2. 실행 진입점
# ============================================================
# Windows에서도 읽기 worker가 스크립트를 재실행하지 않도록 진입점을 구분합니다.
if __name__ == "__main__":
    # set_num_threads(1) : CPU 연산 스레드를 1개로 고정 -> 실행마다 연산 순서가 같아 재현에 유리
    torch.set_num_threads(1)

    # --------------------------------------------------------
    # 학습용 데이터 설정 YAML 만들기 (test 경로 없음)
    # --------------------------------------------------------
    # config 키 의미
    #   train : 학습 이미지 폴더 (라벨은 images -> labels로 바꾼 경로에서 자동 탐색)
    #   val   : 검증 이미지 폴더 (epoch마다 mAP 계산)
    #   names : 클래스 번호 순서의 이름 목록 -> 0=square, 1=circle, 2=triangle
    # 주의: test 키를 넣지 않는다 -> 학습·선택 과정에 test가 섞이지 않음
    config = {"train": str(BASE/"data/images/train"), "val": str(BASE/"data/images/val"),
              "names": ["square", "circle", "triangle"]}
    config_path = BASE / "p04_dataset.yaml"
    # yaml.safe_dump(dict) : 딕셔너리를 YAML 문자열로 변환해 파일로 저장
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # --------------------------------------------------------
    # 모델 구조 생성과 1 epoch 학습 (TODO B05)
    # --------------------------------------------------------
    # YOLO("yolo26n.yaml") : 설치 패키지에 든 구조 설정만으로 무작위 가중치 모델 생성 (다운로드 없음)
    # 주의: .yaml = 구조(빈 가중치), .pt = 학습된 가중치·메타정보가 든 checkpoint
    # 구조만 생성합니다. COCO 사전학습 .pt와 다릅니다.
    model = YOLO("yolo26n.yaml")
    # >>> B05
    # 요구: config_path로 1 epoch, imgsz=160, batch=4, CPU 학습을 호출하세요.
    # 나머지 옵션은 제공한 그대로 유지하세요. test 경로는 전달하지 않습니다.
    # model.train(...) : 학습 후 runs 폴더에 results.csv·weights/best.pt·last.pt 저장
    #   data=str(config_path)       : 위에서 만든 데이터 설정 YAML (이번 맞춤 데이터는 필수)
    #   epochs=1                    : train 12장을 1번만 학습 -> 연결 점검 목적
    #   imgsz=160                   : 입력 한 변 160px (원본 160×128을 letterbox로 맞춤)
    #   batch=4                     : 한 번에 4장 -> 12장 / 4 = 3 iteration
    #   device="cpu"                : CPU 학습 (GPU 없이 실행)
    #   workers=0                   : 데이터 로딩을 메인 프로세스에서 수행 (Windows 재실행 문제 방지)
    #   seed=42                     : 난수 시작값 고정
    #   project=str(BASE/"p04_runs"), name="smoke" : 결과 폴더 p04_runs/smoke/
    #   exist_ok=True               : 재실행 시 smoke2, smoke3을 만들지 않고 같은 폴더 사용
    model.train(data=str(config_path), epochs=1, imgsz=160, batch=4, device="cpu",
                workers=0, seed=42, project=str(BASE/"p04_runs"), name="smoke", exist_ok=True)
    # <<< B05
    # model.task : 'detect' / model.names : {0:'square', 1:'circle', 2:'triangle'} / model.device : cpu
    print("task:", model.task, "names:", model.names, "device:", model.device)
    print("학습 결과:", BASE/"p04_runs/smoke")

'''
Starting training for 1 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
        1/1         0G      3.105      4.262    0.04568         12        160: 100% ━━━━━━━━━━━━ 3/3 2.2it/s 1.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 11.8it/s 0.1s
                   all          6         18          0          0          0          0

1 epochs completed in 0.001 hours.
Optimizer stripped from C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke\weights\last.pt, 5.3MB
Optimizer stripped from C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke\weights\best.pt, 5.3MB

Validating C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke\weights\best.pt...
Ultralytics 8.4.203  Python-3.13.16 torch-2.14.0+cu130 CPU (Intel Core Ultra 9 275HX)
YOLO26n summary (fused): 120 layers, 2,375,421 parameters, 0 gradients, 5.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 20.2it/s 0.0s
                   all          6         18          0          0          0          0
                square          6          6          0          0          0          0
                circle          6          6          0          0          0          0
              triangle          6          6          0          0          0          0
Speed: 0.0ms preprocess, 6.3ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke
task: detect names: {0: 'square', 1: 'circle', 2: 'triangle'} device: cpu
학습 결과: C:\workspace\practice_assignment\20261106\01_basic\p04_runs\smoke
'''