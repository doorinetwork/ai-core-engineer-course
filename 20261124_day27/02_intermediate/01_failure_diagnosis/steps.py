"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# ============================================================
# 1. 실패 단계 판정 : diagnose_stage() 
# ============================================================
# diagnose_stage : 고정 실패 사례 1건에서 "가장 먼저" 발생한 문제 단계를 판정
# - 권한 없음(authorized False)            : 'authorization_filter'
# - 옛 버전 문서(current False)            : 'version_filter'
# - 색인 미공개(ready False)               : 'index_not_ready'
# - 질문이 모호함(ambiguous True)          : 'rewrite'
# - 정답 청크가 후보에 없음                : 'retrieval'
# - 후보에는 있으나 Context 에 못 들어감   : 'reranking'
# - 근거와 답이 맞지 않음(valid_source False): 'validation'
# - 위 조건에 하나도 걸리지 않음           : 'healthy'
# 예: F03 {gold_in_candidates: True, gold_in_context: False} -> 'reranking'
def diagnose_stage(case):
    # BEGIN D27-01-I01
    # TODO D27-01-I01 완료: 어떤 문제가 먼저 발생했는지 위에서부터 조건을 검사합니다.
    # --------------------------------------------------------
    # 1) 검색 이전 단계(권한·버전·색인 준비)를 먼저 검사
    # --------------------------------------------------------
    if not case['authorized']:
        return 'authorization_filter'
    if not case['current']:
        return 'version_filter'
    if not case['ready']:
        return 'index_not_ready'
    # --------------------------------------------------------
    # 2) 파이프라인 순서(Rewrite -> Retrieval -> Reranking)대로 검사
    # --------------------------------------------------------
    # 질문 자체가 모호하면 검색 결과를 보기 전에 Rewrite 문제로 판정
    if case['ambiguous']:
        return 'rewrite'
    # 정답 청크가 후보 목록에 없으면 Retriever 문제 (Reranker 는 없는 후보를 찾지 못함)
    if not case['gold_in_candidates']:
        return 'retrieval'
    # 후보에는 있는데 Context 에 못 들어가면 순위(Reranking) 문제
    if not case['gold_in_context']:
        return 'reranking'
    # --------------------------------------------------------
    # 3) 생성 후 검증 단계
    # --------------------------------------------------------
    # 주의: valid_source 는 None(검증 전) 일 수 있으므로 "is False" 로만 실패 판정
    if case['valid_source'] is False:
        return 'validation'
    return 'healthy'
    # END D27-01-I01


# ============================================================
# 2. 단계별 개선 조치 : recommend_action() 
# ============================================================
# recommend_action : 진단 단계 이름을 받아 그 단계에 맞는 개선 조치 문장을 반환
# - 'reranking' : 후보 순위를 다시 평가 (새 후보를 찾는 조치가 아님)
# - 'retrieval' : 검색 대상·필터·Top-k 점검 (Reranker 로는 누락 후보를 복구할 수 없음)
# - 없는 단계 이름 : dict 조회에서 KeyError 발생 (조용히 기본값을 주지 않음)
def recommend_action(stage):
    # BEGIN D27-01-I02
    # TODO D27-01-I02 완료: 문제 단계에 맞는 개선 방법을 반환합니다. Reranker는 누락된 후보를 새로 찾지 않습니다.
    # 단계 이름 : 조치 문장 (diagnose_stage() 가 돌려주는 8개 값과 1:1 대응)
    # - authorization_filter : 서버에서 권한 검사
    # - version_filter       : 현재 유효 문서만 검색
    # - index_not_ready      : Indexing 공개 상태 확인
    # - rewrite              : History 에서 명확한 대상만 복원
    # - retrieval            : 검색 대상·필터·Top-k 점검
    # - reranking            : 후보 관련성 순위 재평가
    # - validation           : 답변과 실제 근거 대조
    # - healthy              : 정상 흐름 유지
    return {'authorization_filter': '서버에서 접근 권한을 검사합니다.',
            'version_filter': '현재 유효한 문서만 검색합니다.',
            'index_not_ready': 'Indexing 공개 상태를 확인합니다.',
            'rewrite': 'History에서 명확한 대상만 복원합니다.',
            'retrieval': '검색 대상·필터·Top-k를 점검합니다.',
            'reranking': '후보 문서의 관련성 순위를 다시 평가합니다.',
            'validation': '답변과 실제 근거를 대조합니다.',
            'healthy': '정상 흐름을 유지합니다.'}[stage]
    # END D27-01-I02