"""Day23 실습 01 · 검색 대상과 ID (응용, mentee).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 5개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge
from day23kit.io import save_result

from tasks import select_original, make_passage_input, make_mapping, reject_duplicate_ids, reject_missing_fields



def run(mode,level,args):
    rows=load_knowledge()
    selected=select_original(rows)
    passage=make_passage_input(selected[1])
    mapping=make_mapping(selected if level==1 else rows)
    result=dict(exercise=1,level=level,source_mode='local_snapshot',
                selected_ids=[r['id'] for r in selected],example_passage=passage,
                total_mapped=len(mapping),mapping=mapping)
    if level>=2:
        result['unique_ids']=len(reject_duplicate_ids(rows))
        result['all_fields_valid']=reject_missing_fields(rows)
    if level>=3:
        result['versions']=audit_versions(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 01 · 검색 대상과 ID')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_01.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
