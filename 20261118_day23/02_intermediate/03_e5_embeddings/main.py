"""Day23 실습 03 · E5 문장 Embedding (응용, mentee).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 5개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge, E5_MODEL_ID
from day23kit.embedding import get_encoder
from day23kit.io import save_result

from tasks import make_passages, make_queries, encode_both, verify_alignment, check_shapes_vectors



def run(mode,level,args):
    rows=load_knowledge()[:3] if level==1 else load_knowledge()
    encoder=get_encoder(mode)
    passages=make_passages(rows)
    questions=make_queries(['서버는 어디에서 실행하나요?'])
    docs,queries=encode_both(encoder,passages,questions)
    result=dict(exercise=3,level=level,source_mode='e5_live' if mode=='live' else 'simulated',
                model_id=E5_MODEL_ID if mode=='live' else encoder.model_id,
                document_shape=list(docs.shape),query_shape=list(queries.shape),
                example_passage=passages[0],example_query=questions[0])
    if level>=2:
        result['alignment']=verify_alignment(rows,docs)
        result['vectors_valid']=check_shapes_vectors(docs,queries,len(rows))
    if level>=3:result['model_contract']=make_model_stamp(encoder,mode)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 03 · E5 문장 Embedding')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_03.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
