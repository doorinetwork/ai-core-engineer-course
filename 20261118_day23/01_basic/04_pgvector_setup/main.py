"""Day23 실습 04 · 서버 확장과 Python 연결.

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live  (필요한 선행 설치 후)
수강생 TODO: 3개 / 상세 요구사항은 각 함수의 주석 및 README.md.
fixture는 모의 시연이지 모델·PostgreSQL 실측 결과가 아닙니다.

실습 04 기본필수: pgAdmin에서 확장 제공 상태를 조회하고, 준비된 지침에 따라 활성화·거리 계산을 확인합니다.
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
import argparse
import sys
from pathlib import Path
# 공통 모듈 폴더를 import 경로 맨 앞에 추가
# Path(__file__).resolve().parents[2] : <수준>/<교시>_주석/main.py 기준 두 단계 위 = mentee 폴더
# -> mentee/common 을 넣어 day23kit 패키지를 import 할 수 있게 함
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
# 제공 모듈 common/day23kit/mathops.py: 2차원 벡터 검사와 Cosine distance 계산 도우미 (cosine_distance)
from day23kit.mathops import cosine_distance
# 제공 모듈 common/day23kit/db.py: 실제 PostgreSQL/pgvector 연결과 테이블·Top-k·HNSW SQL 상수 (connect_db)
from day23kit.db import connect_db
# 제공 모듈 common/day23kit/io.py: 결과 dict를 outputs/ 아래 JSON 파일로 저장 (save_result)
from day23kit.io import save_result


# ============================================================
# 2. 서버 버전·현재 DB 조회 SQL
# ============================================================
# BEGIN D23-04-B01
def sql_server_info():
    # pgAdmin Query Tool에서 실행할 서버 버전·현재 DB·현재 사용자 조회 SQL 문자열을 돌려준다.
    # - 반환값 : SELECT 문 두 개 (version(), current_database()·current_user)
    # - 주의   : 조회만 한다. DROP·CREATE DATABASE 같은 변경 SQL은 넣지 않는다
    # [강사용 정답] D23-04-B01 · sql_server_info
    #     pgAdmin에서 실행할 서버 버전·현재 DB 조회 SQL을 작성하세요.
    #     반드시 SELECT version()과 current_database()를 포함하세요.
    #     DB 변경이나 DROP/CREATE DATABASE는 하지 않습니다.
    # 해설 D23-04-B01: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    # 조회만 하므로 기존 DB 내용을 변경하지 않습니다.
    return 'SELECT version(); SELECT current_database(), current_user;'
# END D23-04-B01

# ============================================================
# 3. vector 확장 제공·설치 상태 조회 SQL
# ============================================================
# BEGIN D23-04-B02
def sql_extension_state():
    # 서버에 vector 확장 파일이 제공되는지(default_version)와 현재 DB에 설치됐는지(installed_version) 조회하는 SQL을 돌려준다.
    # - 행 없음                     : 서버에 확장 파일이 없음 (빌드·복사 단계 문제)
    # - installed_version = NULL    : 파일은 있으나 현재 DB에서 CREATE EXTENSION 전
    # - installed_version = '0.8.7' : 현재 DB에서 사용 가능
    # [강사용 정답] D23-04-B02 · sql_extension_state
    #     vector 확장의 제공 여부와 설치 여부를 확인하는 SQL 문자열을 작성하세요.
    #     pg_available_extensions, installed_version, WHERE name='vector' 사용.
    # 해설 D23-04-B02: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return "SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name='vector';"
# END D23-04-B02

# ============================================================
# 4. SQL 거리 계산 probe
# ============================================================
# BEGIN D23-04-B03
def sql_distance_probe():
    # 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 SQL로 계산하는 문자열을 돌려준다.
    # - '...'::vector : 문자열을 pgvector vector 타입으로 변환
    # - <=>           : Cosine distance 연산자 (1 - Cosine similarity)
    # - 기대값        : 두 벡터가 직교하므로 Cosine 0 -> distance 1.0
    # 주의: 3차원 수동 예제다. 384차원 E5 출력이 아니다
    # [강사용 정답] D23-04-B03 · sql_distance_probe
    #     SQL에서 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 계산하세요.
    #     문자열 반환. CAST는 ::vector, 연산자는 <=>를 사용합니다.
    #     예상값 1.0이며 384차원 E5 출력이 아닙니다.
    # 해설 D23-04-B03: 입력을 먼저 확인하고, 핵심 개념을 적용한 다음 결과를 반환합니다.
    return "SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector AS cosine_distance;"
# END D23-04-B03


# ============================================================
# 5. 실행 흐름 (run)
# ============================================================
def run(mode, level, args):
    # pgAdmin용 SQL을 기록하고, live에서만 실제 DB에 접속해 서버·확장·거리 probe를 확인한다.
    # - mode='fixture' : DB에 접속하지 않음 -> source_mode·db_check 모두 'not_run' (설치 미검증)
    # - mode='live'    : 이 파일 옆 .env로 실제 PostgreSQL 접속 -> 버전·DB 이름·확장 상태 기록
    # - level          : 2 이상은 예시 확장 행 판정·오류 분류, 3은 설치 5단계 미확인 목록 추가
    # 주의: 세 수준이 같은 run()을 쓴다. 기본필수 main.py에는 check_extension_row 등이 없지만 level 분기에 들어가지 않으므로 오류가 나지 않는다
    # ----------------------------------------------------
    # 5-1. pgAdmin에서 실행할 SQL 기록
    # ----------------------------------------------------
    # sqls 키 : 의미
    # - server    : 서버 버전·현재 DB 조회
    # - extension : vector 확장 제공·설치 상태 조회
    # - distance  : [1,0,0] <=> [0,1,0] 거리 probe
    sqls = {'server': sql_server_info(), 'extension': sql_extension_state(), 'distance': sql_distance_probe()}
    result = dict(exercise = 4, level = level, source_mode = 'not_run' if mode == 'fixture' else 'pgvector_live',
                  sql_for_pgadmin = sqls, db_check = 'not_run')
    # ----------------------------------------------------
    # 5-2. 응용·심화: 예시 입력으로 판정 함수 시연
    # ----------------------------------------------------
    # illustrated_extension_state : ('vector', '0.8.7', None) -> available_not_installed
    # example_error_stage         : 'postgres.h not found' -> compile
    if level >= 2:
        result['illustrated_extension_state'] = check_extension_row(('vector', '0.8.7', None))
        result['example_error_stage'] = classify_install_error('postgres.h not found')
    # manual_pending : 아무 단계도 확인하지 않은 {} -> 5단계 전부 미확인
    if level >= 3:
        result['manual_pending'] = audit_install_steps({})
    # ----------------------------------------------------
    # 5-3. live: 실제 DB 접속 확인 (이번 작업에서는 미실행)
    # ----------------------------------------------------
    if mode == 'live':
        # live만 실제 DB 연결. Windows 빌드 및 CREATE EXTENSION은 사전 수동 절차.
        # connect_db(.env 경로) : 제공 함수. PGHOST·PGPORT·PGDATABASE·PGUSER·PGPASSWORD를 읽어 psycopg 연결 + register_vector
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # fetchone()[0] : 결과 첫 행의 첫 열 값
            ver = conn.execute('SELECT version()').fetchone()[0]
            dbname = conn.execute('SELECT current_database()').fetchone()[0]
            ext = conn.execute("SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name='vector'").fetchone()
            # 응용 이상은 확장 상태를 판정 dict로, 기본필수는 제공 여부(True/False)만 기록
            result['db_check'] = {'server_version': ver, 'database': dbname, 'extension': check_extension_row(ext)} if level >= 2 else {'server_version': ver, 'database': dbname, 'extension_provided': ext is not None}
            # ext[2] : installed_version. 설치된 DB에서만 거리 probe 실행 (기대값 1.0)
            if ext is not None and ext[2]:
                result['db_check']['distance_probe'] = float(conn.execute(sql_distance_probe()).fetchone()[0])
            else:
                result['db_check']['distance_probe'] = 'not_run_extension_not_activated'
    return result

# ============================================================
# 6. 실행 진입점 (main)
# ============================================================
def main():
    # 명령행 인자를 읽어 run()을 실행하고 결과를 화면에 출력한 뒤 JSON 파일로 저장한다.
    # - --mode fixture : 기본값. 모의 벡터·오프라인 경로 (PostgreSQL·E5 모델을 쓰지 않음)
    # - --mode live    : 실제 E5 모델·PostgreSQL이 필요한 경로
    # - 저장 위치      : 이 파일 옆 outputs/exercise_04.json
    parser = argparse.ArgumentParser(description = 'Day23 실습 04 · 서버 확장과 Python 연결')
    parser.add_argument('--mode', choices = ['fixture', 'live'], default = 'fixture',
        help = 'fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args = parser.parse_args()
    # run 인자
    # - args.mode : 'fixture' 또는 'live'
    # - level=1   : 수준 번호 (1=기본필수, 2=응용, 3=심화). 이 파일은 기본필수
    # - args      : 명령행 인자 전체
    result = run(args.mode, 1, args)
    # save_result(폴더, 파일명, dict) : 폴더/outputs/파일명 으로 JSON을 저장하고 경로를 돌려준다
    path = save_result(Path(__file__).resolve().parent, 'exercise_04.json', result)
    import json
    # json.dumps 인자
    # - ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 출력
    # - indent=2           : 사람이 읽기 쉽게 두 칸 들여쓰기
    # - default=str        : JSON으로 바꿀 수 없는 값(Path 등)은 문자열로 변환
    print(json.dumps(result, ensure_ascii = False, indent = 2, default = str))
    print('결과 저장:', path)

# ============================================================
# 7. 스크립트 진입점
# ============================================================
if __name__ == '__main__':
    main()
