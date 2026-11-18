"""Day23 실습 06 · Exact·HNSW·EXPLAIN (응용, mentee).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 5개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import E5_MODEL_ID
from day23kit.embedding import get_encoder, encode_question
from day23kit.db import connect_db, exact_topk, check_data_ready
from day23kit.io import save_result

from tasks import create_index_sql, explain_sql, read_scan_type, check_operator_compatibility, compare_topk



def run(mode,level,args):
    sql_index=create_index_sql()
    sql_plan=explain_sql()
    result=dict(exercise=6,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_live',
                index_sql=sql_index,explain_sql=sql_plan)
    if mode=='fixture':
        sample=['Seq Scan on day23_search_docs (cost=...)','Sort Key: (embedding <=> ...)']
        result['scan_type']=read_scan_type(sample)
        result['plan_is_simulated']=True
    else:
        encoder=get_encoder('live')
        qvec=encode_question(encoder,'서버는 어디에서 실행하나요?')
        with connect_db(Path(__file__).with_name('.env')) as conn:
            check_data_ready(conn)
            if getattr(args,'create_index',False):
                conn.execute(sql_index)
                conn.execute('ANALYZE day23_search_docs')
            plan_lines=[str(r[0]) for r in conn.execute(sql_plan,(E5_MODEL_ID,qvec)).fetchall()]
            result['scan_type']=read_scan_type(plan_lines)
            result['explain_lines']=plan_lines
            if level>=3:
                # Exact 기준선과 인덱스 유도 계획을 같은 질문/데이터/필터에서 실제 비교합니다.
                # SET LOCAL은 수업용 트랜잭션 동안만 적용되며 결과의 우열을 단정하지 않습니다.
                conn.execute('SET LOCAL enable_indexscan=off')
                conn.execute('SET LOCAL enable_bitmapscan=off')
                exact_ids=[h['id'] for h in exact_topk(conn,qvec,k=3,category='development')]
                conn.execute('SET LOCAL enable_indexscan=on')
                conn.execute('SET LOCAL enable_bitmapscan=on')
                conn.execute('SET LOCAL enable_seqscan=off')
                indexed_plan=[str(r[0]) for r in conn.execute(sql_plan,(E5_MODEL_ID,qvec)).fetchall()]
                indexed_ids=[h['id'] for h in exact_topk(conn,qvec,k=3,category='development')]
                conn.execute('SET LOCAL enable_seqscan=on')
                confirmed_index=(read_scan_type(indexed_plan)=='index')
                result['comparison']={
                    'category':'development','k':3,'exact_ids':exact_ids,
                    'forced_plan_ids':indexed_ids,
                    'forced_plan_scan':read_scan_type(indexed_plan),
                    'hnsw_index_observed':confirmed_index,
                    'exact_vs_ann_overlap':compare_topk(exact_ids,indexed_ids) if confirmed_index else None,
                    'note':'Index Scan이 관측됐을 때만 ANN 비교라 기록합니다. 9건에서 속도 향상 주장 금지.',
                }
    if level>=2:
        result['operator_compatible']=check_operator_compatibility(sql_plan,'vector_cosine_ops')
        result['example_overlap']=compare_topk([102,106,107],[102,107,101])
    if level>=3:
        result['ann_filtered_example']=filtered_ann_warning(k=3,returned=1,exact_count=4)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 06 · Exact·HNSW·EXPLAIN')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')

    parser.add_argument('--create-index',action='store_true',help='live에서만 HNSW 인덱스를 실제 생성')
    args=parser.parse_args()
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_06.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
