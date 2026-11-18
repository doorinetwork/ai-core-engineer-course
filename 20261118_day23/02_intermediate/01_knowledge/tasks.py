"""Day23 실습 01 · 응용 · tasks

응용 수준의 TODO 함수 5개를 모은 모듈이다. main.py가 import 해서 run()에서 호출한다.
검색 대상 지식 101~109의 ID·원문·버전을 벡터 생성 전에 고정·검증하는 함수들이다.

이 파일의 TODO 함수:
- select_original (D23-01-I01): 기존 수업에서 사용한 101·102·103만 ID 오름차순으로 선택합니다.
- make_passage_input (D23-01-I02): E5 문서 입력 텍스트를 "passage: 제목 본문" 형식으로 만드세요.
- make_mapping (D23-01-I03): 각 지식의 id/title/content/category/source_version 및 passage_input을 기록하세요.
- reject_duplicate_ids (D23-01-I04): 모든 9개 지식의 ID 중복 여부를 점검하고 중복이면 ValueError를 내세요.
- reject_missing_fields (D23-01-I05): title/content/category/source_version의 누락 또는 공백 문자열을 거부하세요.
"""

# ============================================================
# 1. 원래 지식 101~103 선택
# ============================================================
# BEGIN D23-01-I01
def select_original(rows):
    # 전체 지식 목록에서 기존 수업 지식 101·102·103만 복사해 ID 오름차순으로 돌려준다.
    # - rows   : load_knowledge()가 읽은 지식 dict 목록 (101~109, 9건)
    # - 반환값 : 새 dict 3개 목록, ID 순서 [101, 102, 103]
    # - 원본 rows의 dict는 수정하지 않는다
    # [강사용 정답] D23-01-I01 · select_original
    #     기존 수업에서 사용한 101·102·103만 ID 오름차순으로 선택합니다.
    #     입력: 전체 지식 목록(list[dict]). 출력: 원래 지식 3개를 담은 list.
    #     원본 딕셔너리의 id/title/content를 수정하지 마세요.
    #     개념: 원문 ID 고정·검색할 자료의 범위.
    # 해설 D23-01-I01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # ① 기존 지식의 ID만 고르고 ② 순서 고정: 앞 차시의 ID/원문과 대응합니다.
    # (dict(r) for r in rows if ...) : 조건에 맞는 행만 얕은 복사 -> 원본 스냅샷 보존
    # key=lambda r: r['id']          : ID 오름차순 정렬 (입력 순서가 섞여 있어도 [101, 102, 103])
    return sorted((dict(r) for r in rows if r['id'] in (101, 102, 103)), key = lambda r: r['id'])
# END D23-01-I01

# ============================================================
# 2. E5 문서 입력 문자열 만들기
# ============================================================
# BEGIN D23-01-I02
def make_passage_input(row):
    # 지식 한 행을 E5 문서 입력 형식 "passage: 제목 본문" 문자열로 만든다.
    # - row    : 지식 dict 한 개 (title·content 사용)
    # - 반환값 : 'passage: ' + 제목 + 공백 1개 + 본문
    # 예: 102 -> "passage: 서버 실행 프로젝트 폴더에서 서버를 실행합니다."
    # [강사용 정답] D23-01-I02 · make_passage_input
    #     E5 문서 입력 텍스트를 "passage: 제목 본문" 형식으로 만드세요.
    #     입력: 지식 한 행. 출력: 문자열. 제목과 본문을 공백 하나로 구분하세요.
    #     개념: FAQ 질문과 달리 지식 본문을 검색 대상으로 사용합니다.
    # 해설 D23-01-I02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # E5 비대칭 검색은 문서에는 passage: 접두어가 필요합니다.
    # 주의: 질문 쪽은 'query: ' 접두어를 쓴다. 문서와 질문의 접두어를 바꿔 쓰면 E5 검색 계약이 깨진다
    return 'passage: ' + row['title'] + ' ' + row['content']
# END D23-01-I02

# ============================================================
# 3. ID·원문·버전·passage 입력 대응표 만들기
# ============================================================
# BEGIN D23-01-I03
def make_mapping(rows):
    # 지식마다 ID·원문·분류·버전과 E5 입력 문자열을 한 dict로 묶은 대응표를 만든다.
    # - rows   : 대응표를 만들 지식 목록 (기본필수는 101~103, 응용·심화는 101~109)
    # - 반환값 : dict 목록, 입력과 같은 순서
    # 대응표 키 : 의미
    # - id             : 지식 ID (벡터 행 번호가 아니라 원문을 찾는 열쇠)
    # - title/content  : 원문 제목·본문
    # - category       : 분류 (course / development / support)
    # - source_version : 원문 버전 (101~103은 knowledge_v1, 104~109는 knowledge_search_v2)
    # - passage_input  : make_passage_input()이 만든 E5 문서 입력
    # [강사용 정답] D23-01-I03 · make_mapping
    #     각 지식의 id/title/content/category/source_version 및 passage_input을 기록하세요.
    #     입력: 선택된 지식 목록. 출력: 딕셔너리 목록.
    #     개념: 벡터를 생성하기 전에도 원문·ID·버전을 추적해야 합니다.
    # 해설 D23-01-I03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 각 ID의 원문과 Embedding 입력을 함께 보존합니다.
    return [dict(id = r['id'], title = r['title'], content = r['content'],
                 category = r['category'], source_version = r['source_version'],
                 passage_input = make_passage_input(r)) for r in rows]
# END D23-01-I03

# ============================================================
# 4. ID 타입·중복 검사
# ============================================================
# BEGIN D23-01-I04
def reject_duplicate_ids(rows):
    # 모든 지식의 id가 양의 정수(bool 제외)이고 중복이 없는지 검사한다.
    # - id가 int가 아니거나 0 이하 : ValueError('id 타입 오류')
    # - 같은 id가 두 번 이상        : ValueError('duplicate_id')
    # - 정상                        : ID 집합 set 반환 (예: {101, ..., 109} -> len 9)
    # [강사용 정답] D23-01-I04 · reject_duplicate_ids
    #     모든 9개 지식의 ID 중복 여부를 점검하고 중복이면 ValueError를 내세요.
    #     단순 len만 비교하지 말고 id의 타입이 int(True 제외)인지도 확인합니다.
    #     출력: 중복이 없으면 실제 ID 집합(set).
    # 해설 D23-01-I04: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # r.get('id') : id 키가 없으면 None -> 아래 타입 검사에서 걸림
    ids = [r.get('id') for r in rows]
    # bool은 int의 하위 타입이므로 type(...) is int로 엄격하게 검사합니다.
    # 예: True는 isinstance(True, int)가 True지만 type(True) is int는 False
    if any(type(x) is not int or x <= 0 for x in ids):
        raise ValueError('id 타입 오류')
    # 목록 길이와 집합 크기가 다르면 중복 ID가 있다는 뜻
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_id')
    return set(ids)
# END D23-01-I04

# ============================================================
# 5. 필수 문자열 누락·공백 검사
# ============================================================
# BEGIN D23-01-I05
def reject_missing_fields(rows):
    # title·content·category·source_version 네 필드가 비어 있지 않은 문자열인지 검사한다.
    # - 필드 없음·문자열 아님·공백만 있음 : ValueError('missing_or_blank:필드명')
    # - 모든 행이 정상                    : True
    # [강사용 정답] D23-01-I05 · reject_missing_fields
    #     title/content/category/source_version의 누락 또는 공백 문자열을 거부하세요.
    #     입력: 전체 지식 목록. 정상 출력: True. 잘못된 항목은 ValueError.
    #     개념: 잘못된 원문은 정상적인 Embedding 입력이 될 수 없습니다.
    # 해설 D23-01-I05: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    for r in rows:
        for name in ('title', 'content', 'category', 'source_version'):
            # r.get(name) : 키가 없으면 None -> isinstance 검사에서 걸림
            # r[name].strip() : 공백만 있는 문자열 "  " 은 빈 문자열이 되어 거부
            if not isinstance(r.get(name), str) or not r[name].strip():
                raise ValueError('missing_or_blank:' + name)
    return True
# END D23-01-I05
