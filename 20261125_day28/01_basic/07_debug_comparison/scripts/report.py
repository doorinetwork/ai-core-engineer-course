"""
Day28 실습 07 · 01_basic · Debug 비교 결과 Markdown 보고서 작성 스크립트.

- 역할: main.py 가 만든 outputs/result.json 을 읽어 Question ID 별 비교 보고서(outputs/report_summary.md)를 만든다
  - 계산은 main.py(execute)가 한다. 이 스크립트는 읽기·표 만들기만 한다 (평균만 여기서 계산)
  - main.py 가 실습 08 에서 쓰는 report.md 와 겹치지 않게 report_summary.md 로 저장한다
- 입력 형식: {'mode', 'dataset_version', 'comparison_kind', 'comparisons': [{question_id, baseline_recall, rerank_recall, delta_recall, diagnosis}, ...]}
  - offline_fixture : Q01·Q07·Q06 3건 / live : Q01 1건
- None 값(No-answer 질문 Q06) 은 0 이 아니라 '해당 없음' 으로 쓰고 평균에서 뺀다
- 실행 위치: 실습 폴더에서 python scripts/report.py   (다른 result.json 을 보려면 경로를 인자로 준다)
  - 예: python scripts/report.py ../07_debug_comparison_주석/outputs/result.json
"""
from pathlib import Path
import json
import sys

# ============================================================
# 1. 경로 설정
# ============================================================
# UNIT_DIR : 이 파일 기준 한 단계 위 폴더 (<실습>/scripts/report.py -> <실습>)
UNIT_DIR = Path(__file__).resolve().parents[1]
# RESULT : main.py 결과 파일 (인자로 경로를 주면 그 파일)
RESULT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else UNIT_DIR / 'outputs' / 'result.json'
# REPORT : 이 스크립트가 쓰는 보고서 파일
REPORT = UNIT_DIR / 'outputs' / 'report_summary.md'

# ============================================================
# 2. diagnosis 값 범례 (coursekit/eval_experiments.py diagnose() 분기 순서)
# ============================================================
# - find_problem_stage() 는 diagnose(baseline 실행, Gold) 를 그대로 돌려준다
# - 위에서부터 차례로 검사해 처음 맞는 값을 반환한다
DIAGNOSIS = [
    ('system_failure', 'status 가 answered·insufficient_context 가 아님', '실행 오류 (Target·LLM 호출)'),
    ('no_answer_case', 'Gold 가 비어 있음 (답이 없는 질문)', '정보 부족 응답을 별도로 평가'),
    ('candidate_miss', 'candidate_ids 에 Gold 가 없음', 'Query Rewrite·검색 단계'),
    ('ranking_miss', 'ranked_ids 상위 2개에 Gold 가 없음', 'Reranker·Top-N (검색 순위)'),
    ('context_miss', 'included_ids(LLM Context) 에 Gold 가 없음', 'Context 조립·context_k'),
    ('generation_or_validation', 'Gold 는 Context 에 있으나 status 가 answered 가 아님', '생성·출력 검증'),
    ('inspect_answer_grounding', '위 검사를 모두 통과 (출처 ID 만으로 사실 충실성은 인증 불가)', '답변 근거(Claim) 직접 확인'),
]
DIAG_DESC = {k: (cond, fix) for k, cond, fix in DIAGNOSIS}


# ============================================================
# 3. 값 표시 함수
# ============================================================
# show : None 은 '해당 없음', 나머지는 문자열
def show(value):
    if value is None:
        return '해당 없음'
    return str(value)


# mean : None 을 뺀 평균과 개수 (값이 없으면 None)
def mean(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None, 0
    return round(sum(valid) / len(valid), 4), len(valid)


# ============================================================
# 4. 보고서 만들기
# ============================================================
# build_report : result dict -> Markdown 문자열
def build_report(result):
    rows = result.get('comparisons', [])
    lines = ['# Day28 실습 07 · Baseline vs Rerank 비교 보고서', '']
    # --------------------------------------------------------
    # 실행 정보와 해석 경고
    # --------------------------------------------------------
    lines += ['## 1. 실행 정보', '', '| 항목 | 값 |', '|:--|:--|',
              f"| mode | {show(result.get('mode'))} |",
              f"| dataset_version | {show(result.get('dataset_version'))} |",
              f"| comparison_kind | {show(result.get('comparison_kind'))} |",
              f"| 비교 질문 수 | {len(rows)} ({', '.join(r['question_id'] for r in rows)}) |",
              f"| 입력 파일 | {RESULT.name} |", '']
    if result.get('mode') == 'offline_fixture':
        lines.append('- 주의: mode 가 offline_fixture 다. 교육용 고정 데이터로 계산한 값이며 실제 모델·DB 성능이 아니다.')
    elif result.get('mode') == 'live':
        lines.append('- 주의: mode 가 live 다. 실제 RAG 를 한 번 실행한 값이라 반복 실행 전에는 결론을 확정하지 않는다.')
    if result.get('comparison_kind') == 'offline_version_comparison_not_online_AB':
        lines.append('- 주의: comparison_kind 는 같은 질문셋으로 두 버전(baseline·rerank_only)을 비교한 오프라인 버전 비교다. 실사용자 온라인 A/B 실험이 아니다.')
    lines.append('')
    # --------------------------------------------------------
    # Question ID 별 세로 표
    # --------------------------------------------------------
    lines += ['## 2. 질문별 비교', '']
    for r in rows:
        cond, fix = DIAG_DESC.get(r.get('diagnosis'), ('코드 정의에 없는 값', '-'))
        lines += [f"### {r['question_id']}", '', '| 항목 | 값 |', '|:--|:--|',
                  f"| baseline_recall (Recall@3) | {show(r.get('baseline_recall'))} |",
                  f"| rerank_recall (Recall@3) | {show(r.get('rerank_recall'))} |",
                  f"| delta_recall (rerank - baseline) | {show(r.get('delta_recall'))} |",
                  f"| diagnosis (baseline 실행 기준) | {show(r.get('diagnosis'))} |",
                  f"| diagnosis 조건 | {cond} |",
                  f"| 먼저 볼 곳 | {fix} |", '']
    # --------------------------------------------------------
    # 평균 (None 제외)
    # --------------------------------------------------------
    lines += ['## 3. 평균 (해당 없음 제외)', '', '| 항목 | 값 |', '|:--|:--|']
    for key in ('baseline_recall', 'rerank_recall', 'delta_recall'):
        m, n = mean(r.get(key) for r in rows)
        lines.append(f'| {key} 평균 | {show(m)} ({n}/{len(rows)}건) |')
    excluded = [r['question_id'] for r in rows if r.get('delta_recall') is None]
    lines += [f"| 평균에서 뺀 질문 | {', '.join(excluded) if excluded else '없음'} |", '']
    # --------------------------------------------------------
    # diagnosis 범례
    # --------------------------------------------------------
    lines += ['## 4. diagnosis 범례 (coursekit/eval_experiments.py diagnose(), 위에서부터 검사)', '',
              '| 값 | 조건 | 먼저 볼 곳 |', '|:--|:--|:--|']
    lines += [f'| {k} | {cond} | {fix} |' for k, cond, fix in DIAGNOSIS]
    lines.append('')
    return '\n'.join(lines)


# ============================================================
# 5. 실행 흐름 main()
# ============================================================
def main():
    if not RESULT.exists():
        sys.exit(f'{RESULT} 가 없습니다. 먼저 main.py를 실행하세요 (python main.py --offline)')
    result = json.loads(RESULT.read_text(encoding = 'utf-8'))
    if result.get('exercise') != 7 or 'comparisons' not in result:
        sys.exit('실습 07 result.json 이 아닙니다 (exercise 7, comparisons 필요)')
    REPORT.parent.mkdir(parents = True, exist_ok = True)
    REPORT.write_text(build_report(result), encoding = 'utf-8')
    print(f'보고서 저장: {REPORT}')


if __name__ == '__main__':
    main()
