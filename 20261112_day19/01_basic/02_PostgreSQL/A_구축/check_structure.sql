-- Day19 오전 실습 2 · 기본필수 | check_structure.sql : 구축 결과를 catalog로 읽어 명세와 대조 (읽기 전용)
-- 실행 DB: day19_build_b (pgAdmin Query Tool 또는 psql)
-- ============================================================
-- 1. 안전 확인: 실행 대상 DB 검사
-- ============================================================
-- 제공 안전장치: 아래 DO 구문은 수정 과제가 아닙니다.
-- current_database()가 day19_build_b가 아니면 RAISE EXCEPTION으로 즉시 중단
-- -> knowledge_lab이나 다른 DB에 실수로 실행하는 것을 막는다 (데이터는 바꾸지 않음)
DO $guard$
BEGIN
    IF current_database() <> 'day19_build_b' THEN
        RAISE EXCEPTION 'Wrong database: expected day19_build_b, got %', current_database();
    END IF;
END;
$guard$;
-- ============================================================
-- 2. 열 정의 확인 (information_schema.columns)
-- ============================================================
-- 각 결과를 스키마명세서와 비교합니다. column_default의 내부 시퀀스명은 채점하지 않습니다.
-- 열별 자료형·최대 길이·NULL 허용·IDENTITY 여부 조회
-- 기대 6행: categories(id bigint ALWAYS, name character varying 30) / knowledge(id bigint ALWAYS, title 80, content text, category_id bigint)
-- 모든 열 is_nullable=NO
SELECT table_name, column_name, data_type, character_maximum_length,
       is_nullable, is_identity, identity_generation
FROM information_schema.columns
WHERE table_schema='public' AND table_name IN ('categories','knowledge')
ORDER BY table_name, ordinal_position;
-- ============================================================
-- 3. 제약조건 확인 (pg_constraint)
-- ============================================================
-- contype : p=PRIMARY KEY, u=UNIQUE, c=CHECK, f=FOREIGN KEY
-- pg_get_constraintdef(oid) : 제약 정의 문장 예: FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT
-- 기대: PK 2, UNIQUE 1(categories_name_key), CHECK 3, FK 1
SELECT conrelid::regclass AS table_name, contype, conname,
       pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid IN ('public.categories'::regclass, 'public.knowledge'::regclass)
ORDER BY conrelid::regclass::text, contype, conname;
-- PostgreSQL 버전에 따라 NOT NULL이 catalog 결과에 더 표시될 수 있습니다.
-- PK/UQ의 자동 이름이나 CHECK 표현의 괄호 개수 대신 실제 열·규칙·FK 삭제 정책을 확인합니다.
