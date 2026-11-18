"""Day23 실습 05 · 원문·벡터 저장과 Top-k (응용, mentee).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 5개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge, E5_MODEL_ID, TABLE_NAME
from day23kit.fixtures import fixture_search
from day23kit.embedding import get_encoder, encode_passages, encode_question, validate_vectors
from day23kit.db import connect_db, ensure_table, upsert_rows, exact_topk
from day23kit.io import save_result

from tasks import schema_sql, prepare_document_rows, topk_sql, filtered_sql, check_search_options



def run(mode,level,args):
    rows=load_knowledge()
    sql_schema=schema_sql()
    inputs=prepare_document_rows(rows)
    sql_query=topk_sql()
    question='서버는 어디에서 실행하나요?'
    result=dict(exercise=5,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_exact',
                table=TABLE_NAME,model=E5_MODEL_ID,prepared_count=len(inputs),
                first_passage=inputs[0]['passage_input'],
                sql_schema=sql_schema,sql_topk=sql_query)
    category=None if level==1 else 'development'
    if level>=2:
        result['sql_filtered']=filtered_sql()
        result['options']=check_search_options(3,category)
    if mode=='fixture':
        hits=fixture_search(question,3,category=category)
    else:
        encoder=get_encoder('live')
        vectors=encode_passages(encoder,rows)
        if level>=3:check_id_vector_integrity(rows,vectors)
        else:validate_vectors(vectors,len(rows))
        qvec=encode_question(encoder,question)
        with connect_db(Path(__file__).with_name('.env')) as conn:
            ensure_table(conn)
            upsert_rows(conn,rows,vectors,E5_MODEL_ID)
            hits=exact_topk(conn,qvec,k=3,category=category)
    result['hits']=[{'id':h['id'],'content':h['content'],'distance':round(h['distance'],6)} for h in hits]
    if level>=3:
        # 이전 단계의 벡터를 재사용합니다. live에서는 같은 모델을 다시 다운로드/로드하지 않습니다.
        if mode=='fixture':
            encoder=get_encoder('fixture')
            vectors=encode_passages(encoder,rows)
        result['id_vector_integrity']=check_id_vector_integrity(rows,vectors)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 05 · 원문·벡터 저장과 Top-k')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_05.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
