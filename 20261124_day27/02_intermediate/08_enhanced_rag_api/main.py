"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))

import argparse

from coursekit.workshops import STAGE_RUNNERS
from coursekit.io import save_report

from steps import make_service_chain
from steps import wire_fastapi
from checks import check_http_response

def main():
    parser=argparse.ArgumentParser(description='Day27 실습 08 · 08_enhanced_rag_api')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    report=STAGE_RUNNERS[8](args.offline,[make_service_chain,wire_fastapi,check_http_response])
    path=save_report(HERE,'day27_08_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 08 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

if __name__=='__main__':main()
