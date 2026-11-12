-- Day19 오전 실습 2 · 기본필수 | build.sql : ERD·스키마 명세대로 categories·knowledge 구축
-- 실행 DB: day19_build_b (pgAdmin Query Tool 또는 psql)
-- ============================================================
-- 1. 트랜잭션 시작과 안전 확인
-- ============================================================
-- 맨 앞 BEGIN; : 이 파일의 문장 전체를 한 트랜잭션으로 묶음 -> 중간 오류 시 ROLLBACK으로 전부 취소
BEGIN;
-- 제공 안전장치: 아래 DO 구문은 수정 과제가 아닙니다.
-- current_database()가 day19_build_b가 아니면 RAISE EXCEPTION으로 즉시 중단 -> 다른 DB 보호
DO $guard$
BEGIN
    IF current_database() <> 'day19_build_b' THEN
        RAISE EXCEPTION 'Wrong database: expected day19_build_b, got %', current_database();
    END IF;
END;
$guard$;
-- ============================================================
-- 2. 부모 테이블 categories(분류) 만들기
-- ============================================================
-- categories : 부모 테이블. 지식(knowledge) table이 참조
CREATE TABLE public.categories (
    -- BEGIN B-DDL-01
    -- TODO B-DDL-01: categories의 id를 BIGINT 자동 생성 IDENTITY와 PRIMARY KEY로 선언하세요.
    -- 쉼표까지 포함합니다.
    -- id = BIGINT + GENERATED ALWAYS AS IDENTITY + PRIMARY KEY, 끝에 쉼표
    -- GENERATED ALWAYS AS IDENTITY : INSERT 때 DB가 1, 2, 3 ... 번호 발급 (직접 id를 넣으면 428C9 오류)
    -- PRIMARY KEY : 중복·NULL 금지 + 자동 인덱스(categories_pkey)
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- END B-DDL-01
    -- name = VARCHAR(30) 최대 30자 + NOT NULL + UNIQUE, 끝에 쉼표
    -- UNIQUE : 같은 분류명 'learning'을 다시 넣으면 23505 (자동 제약 이름 categories_name_key)
    name VARCHAR(30) NOT NULL UNIQUE,
    -- BEGIN B-DDL-02
    -- TODO B-DDL-02: categories.name의 1~30자 길이 CHECK를 categories_name_length라는
    -- 이름으로 완성하세요.
    CONSTRAINT categories_name_length CHECK (char_length(name) BETWEEN 1 AND 30)
    -- END B-DDL-02
);
-- ============================================================
-- 3. 자식 테이블 knowledge(지식) 만들기
-- ============================================================
-- knowledge : 자식 테이블. category_id로 categories table을 참조
CREATE TABLE public.knowledge (
    -- 제공 코드: 지식 id도 BIGINT ALWAYS IDENTITY PK (제목과 별개인 식별자, 제목 중복은 허용)
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- 제공 코드: 세 열 모두 NOT NULL, 각 줄 끝 쉼표
    -- title VARCHAR(80) : 최대 80자 / content TEXT : 길이 제한 없는 문자열 / category_id BIGINT : 부모 id와 같은 자료형
    title VARCHAR(80) NOT NULL,
    content TEXT NOT NULL,
    category_id BIGINT NOT NULL,
    -- BEGIN B-DDL-03
    -- TODO B-DDL-03: title은 1~80자, content는 1~2000자인 두 CHECK를 명세의 이름으로 완성하세요. 각 줄
    -- 끝 쉼표가 필요합니다.
    -- NOT NULL과 길이 CHECK는 다른 규칙입니다. 빈 문자열은 NULL이 아니므로 길이 하한으로 막습니다.
    -- 제목 중복 금지나 공백 제거를 이 요구사항에 추가하지 않습니다.
    CONSTRAINT knowledge_title_length CHECK (char_length(title) BETWEEN 1 AND 80),
    CONSTRAINT knowledge_content_length CHECK (char_length(content) BETWEEN 1 AND 2000),
    -- END B-DDL-03
    -- 제공 코드: FK 제약 이름 knowledge_category_fk, 자식 열 (category_id)
    CONSTRAINT knowledge_category_fk FOREIGN KEY (category_id)
        -- BEGIN B-DDL-04
        -- TODO B-DDL-04: category_id가 public.categories(id)를 참조하도록 REFERENCES와 부모 삭제
        -- RESTRICT를 완성하세요.
        -- FK는 실제 부모 존재를 검사합니다. category_id의 NOT NULL과 함께 필수 관계를 만들고
        -- RESTRICT는 참조 중인 부모 삭제를 막습니다. FK가 JOIN 결과를 자동으로 만들지는 않습니다.
        REFERENCES public.categories(id) ON DELETE RESTRICT
        -- END B-DDL-04
);
-- ============================================================
-- 4. 확정
-- ============================================================
-- 앞의 CREATE가 모두 성공했을 때만 확정. 오류가 났다면 ROLLBACK을 따로 실행하고 고친 뒤 다시 실행
COMMIT;
