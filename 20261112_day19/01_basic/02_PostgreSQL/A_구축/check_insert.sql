-- Day19 오전 실습 2 · 기본필수 | check_insert.sql : 구축한 두 표에 임시 행을 넣어 PK·FK·JOIN 확인 후 취소
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
-- 2. 임시 데이터로 연결 확인 후 취소
-- ============================================================
-- 구축 직후 비어 있는 두 테이블에서 수행하는 제공 점검입니다.
-- 임시 데이터는 ROLLBACK되어 남지 않습니다. 시퀀스 번호는 증가할 수 있습니다.
-- 트랜잭션 시작: 아래 INSERT는 마지막 ROLLBACK으로 모두 취소된다
BEGIN;
-- 부모 분류 1행 추가: id는 IDENTITY가 발급, RETURNING id,name으로 바로 확인 (예: 1, __build_check__)
INSERT INTO public.categories(name) VALUES ('__build_check__') RETURNING id,name;
-- INSERT ... SELECT : VALUES 대신 조회 결과(방금 만든 분류 id)를 category_id로 사용 -> FK 통과 확인
INSERT INTO public.knowledge(title,content,category_id)
SELECT '구축 확인','가상의 점검 내용입니다.',id
FROM public.categories WHERE name='__build_check__'
RETURNING id,title;
-- FK 열로 JOIN: 기대 1행 (구축 확인, __build_check__)
SELECT k.title,c.name AS category FROM public.knowledge k
JOIN public.categories c ON c.id=k.category_id WHERE c.name='__build_check__';
-- 두 INSERT 모두 취소 -> 표는 다시 빈 상태 (IDENTITY 번호는 되돌아가지 않을 수 있음)
ROLLBACK;
-- ============================================================
-- 3. 취소 확인
-- ============================================================
-- 기대: knowledge_count=0
SELECT COUNT(*) AS knowledge_count FROM public.knowledge;
-- 기대: 생성 각각 1행, JOIN 1행, 취소 후 knowledge_count=0 (빈 구축 DB 기준).
