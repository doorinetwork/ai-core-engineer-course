"""Day23 실습 02 · Vector·Similarity·Distance (응용, mentee).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 5개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from day23kit.mathops import cosine_distance
from day23kit.io import save_result

from tasks import dot, cosine, l2, normalize, rank_dot_vs_cosine



def run(mode,level,args):
    q=np.array([1.,0.]);a=np.array([.8,.6]);u=np.array([1.,0.]);v=np.array([10.,10.])
    result=dict(exercise=2,level=level,source_mode='local_math',
                dot=round(dot(q,a),6),cosine=round(cosine(q,a),6),l2=round(l2(q,a),6),
                cosine_distance=round(1-cosine(q,a),6))
    if level>=2:
        result['normalized_v']=normalize(v).round(6).tolist()
        result['rankings']=rank_dot_vs_cosine(q,{'U':u,'V':v})
    if level>=3:
        result['validated']=validate_pair(q,a)
        errors={}
        for label,x,y in [('dimension_mismatch',q,np.array([1.,0.,2.])),
                           ('nonfinite',q,np.array([np.nan,1.])),
                           ('zero_vector',q,np.array([0.,0.]))]:
            try:validate_pair(x,y)
            except ValueError as e:errors[label]=str(e)
        result['invalid_inputs']=errors
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 02 · Vector·Similarity·Distance')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_02.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
