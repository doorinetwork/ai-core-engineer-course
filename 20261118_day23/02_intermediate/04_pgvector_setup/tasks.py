"""Day23 실습 04 · 응용 · tasks

응용 수준의 TODO 함수 5개를 모은 모듈이다. main.py가 import 해서 run()에서 호출한다.
pgAdmin에서 실행할 조회용 SQL 3개와 vector 확장 상태 판정·설치 오류 위치 분류를 담당한다.

이 파일의 TODO 함수:
- sql_server_info (D23-04-I01): pgAdmin에서 실행할 서버 버전·현재 DB 조회 SQL을 작성하세요.
- sql_extension_state (D23-04-I02): vector 확장의 제공 여부와 설치 여부를 확인하는 SQL 문자열을 작성하세요.
- sql_distance_probe (D23-04-I03): SQL에서 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 계산하세요.
- check_extension_row (D23-04-I04): fetchone()으로 읽은 (name, default_version, installed_version) 행을 판정하세요.
- classify_install_error (D23-04-I05): 대표 오류를 실패 위치(compile, server_files, db_permission,
"""
# ============================================================
# 1. 모듈 import
# ============================================================
# 제공 모듈 common/day23kit/mathops.py: 2차원 벡터 검사와 Cosine distance 계산 도우미 (cosine_distance)
from day23kit.mathops import cosine_distance

# ============================================================
# 2. 서버 버전·현재 DB 조회 SQL
# ============================================================
# BEGIN D23-04-I01
def sql_server_info():
    # pgAdmin Query Tool에서 실행할 서버 버전·현재 DB·현재 사용자 조회 SQL 문자열을 돌려준다.
    # - 반환값 : SELECT 문 두 개 (version(), current_database()·current_user)
    # - 주의   : 조회만 한다. DROP·CREATE DATABASE 같은 변경 SQL은 넣지 않는다
    # [강사용 정답] D23-04-I01 · sql_server_info
    #     pgAdmin에서 실행할 서버 버전·현재 DB 조회 SQL을 작성하세요.
    #     반드시 SELECT version()과 current_database()를 포함하세요.
    #     DB 변경이나 DROP/CREATE DATABASE는 하지 않습니다.
    # 해설 D23-04-I01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 조회만 하므로 기존 DB 내용을 변경하지 않습니다.
    return 'SELECT version(); SELECT current_database(), current_user;'
# END D23-04-I01

# ============================================================
# 3. vector 확장 제공·설치 상태 조회 SQL
# ============================================================
# BEGIN D23-04-I02
def sql_extension_state():
    # 서버에 vector 확장 파일이 제공되는지(default_version)와 현재 DB에 설치됐는지(installed_version) 조회하는 SQL을 돌려준다.
    # - 행 없음                     : 서버에 확장 파일이 없음 (빌드·복사 단계 문제)
    # - installed_version = NULL    : 파일은 있으나 현재 DB에서 CREATE EXTENSION 전
    # - installed_version = '0.8.7' : 현재 DB에서 사용 가능
    # [강사용 정답] D23-04-I02 · sql_extension_state
    #     vector 확장의 제공 여부와 설치 여부를 확인하는 SQL 문자열을 작성하세요.
    #     pg_available_extensions, installed_version, WHERE name='vector' 사용.
    # 해설 D23-04-I02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return "SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name='vector';"
# END D23-04-I02

# ============================================================
# 4. SQL 거리 계산 probe
# ============================================================
# BEGIN D23-04-I03
def sql_distance_probe():
    # 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 SQL로 계산하는 문자열을 돌려준다.
    # - '...'::vector : 문자열을 pgvector vector 타입으로 변환
    # - <=>           : Cosine distance 연산자 (1 - Cosine similarity)
    # - 기대값        : 두 벡터가 직교하므로 Cosine 0 -> distance 1.0
    # 주의: 3차원 수동 예제다. 384차원 E5 출력이 아니다
    # [강사용 정답] D23-04-I03 · sql_distance_probe
    #     SQL에서 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 계산하세요.
    #     문자열 반환. CAST는 ::vector, 연산자는 <=>를 사용합니다.
    #     예상값 1.0이며 384차원 E5 출력이 아닙니다.
    # 해설 D23-04-I03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return "SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector AS cosine_distance;"
# END D23-04-I03

# ============================================================
# 5. 확장 조회 결과 행 판정
# ============================================================
# BEGIN D23-04-I04
def check_extension_row(row):
    # fetchone()으로 읽은 (name, default_version, installed_version) 행을 설치 상태 dict로 바꾼다.
    # - None                              : {'state': 'unavailable', 'version': None}
    # - ('vector', '0.8.7', None)         : {'state': 'available_not_installed', 'version': None}
    # - ('vector', '0.8.7', '0.8.7')      : {'state': 'installed', 'version': '0.8.7'}
    # [강사용 정답] D23-04-I04 · check_extension_row
    #     fetchone()으로 읽은 (name, default_version, installed_version) 행을 판정하세요.
    #     None이면 unavailable, installed_version이 None이면 available_not_installed,
    #     설치됐으면 installed 및 version을 반환합니다.
    # 해설 D23-04-I04: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    if row is None:
        return {'state': 'unavailable', 'version': None}
    # row[2] : installed_version (None이면 현재 DB에 아직 CREATE EXTENSION 안 됨)
    return {'state': 'installed' if row[2] else 'available_not_installed', 'version': row[2]}
# END D23-04-I04

# ============================================================
# 6. 설치 오류의 실패 위치 분류
# ============================================================
# BEGIN D23-04-I05
def classify_install_error(message):
    # 대표 오류 문구를 보고 어느 설치 단계에서 실패했는지 이름 하나로 분류한다.
    # - 'nmake' / 'postgres.h'              : compile        (빌드 도구·PGROOT 단계)
    # - 'vector.control'                    : server_files   (서버 확장 파일 복사 단계)
    # - 'permission denied' / 'must be superuser' : db_permission (DB 권한)
    # - 'type "vector" does not exist'      : db_extension   (현재 DB에서 확장 미활성화)
    # - 'no module named'                   : python_env     (Python 패키지 누락)
    # - 그 밖                               : other
    # [강사용 정답] D23-04-I05 · classify_install_error
    #     대표 오류를 실패 위치(compile, server_files, db_permission,
    #     db_extension, python_env, other)로 분류하세요.
    #     키나 비밀번호가 들어 있는 예외 원문은 저장하지 않습니다.
    # 해설 D23-04-I05: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 로그 원문 대신 주어진 짧은 오류 종류만 분류합니다.
    # message.lower() : 대소문자 차이를 없애고 비교 (예: 'No module named' -> 'no module named')
    msg = message.lower()
    if 'nmake' in msg or 'postgres.h' in msg:
        return 'compile'
    if 'vector.control' in msg:
        return 'server_files'
    if 'permission denied' in msg or 'must be superuser' in msg:
        return 'db_permission'
    if 'type "vector" does not exist' in msg:
        return 'db_extension'
    if 'no module named' in msg:
        return 'python_env'
    return 'other'
# END D23-04-I05
