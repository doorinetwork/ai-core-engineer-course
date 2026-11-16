"""Day21 실습 01: Prompt 구조와 요구사항. 실행: python main.py"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys
import json
from string import Template

# HERE      : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# ROLE_ROOT : HERE.parents[1] = mentee(또는 mentor) 폴더. coursekit/·data/가 이 위치에 있다
# sys.path.insert(0, ...) : 어느 작업 폴더에서 실행해도 coursekit 패키지를 import하도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
# 제공 모듈 coursekit/io.py: data/ 경로(DATA)·UTF-8 JSON/텍스트 저장·실행 래퍼(run) 등 파일 도구
from coursekit.io import DATA, read_json, write_json, write_text, run
# 제공 모듈 coursekit/prompting.py: 네 필드 출력 계약 문장(CONTRACT_TEXT)
from coursekit.prompting import CONTRACT_TEXT

# 기준: Day21 PPT 7~13장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

# ============================================================
# 2. 요구사항 여섯 칸 작성
# ============================================================
# 지식 답변 Prompt의 요구사항을 역할·작업·입력·제약·실패·출력 여섯 칸 dict로 만든다.
# 반환값: 6개 키가 모두 있고 값이 빈 문자열이 아닌 dict
# 이 값은 Prompt Template의 $role, $task 등 자리에 그대로 들어간다.
def build_requirements():
    # TODO D21-01-B01: 역할·작업·입력·제약·실패·출력 여섯 칸을 완성하세요.
    # 입력·대상: PPT 8장의 여섯 요소
    # 완료 기준: 6개 키가 모두 있고 각 문장이 비어 있지 않습니다.
    # 핵심: 역할을 전문가로 꾸미는 것과 정확한 근거를 제공하는 것은 다릅니다.
    # BEGIN D21-01-B01
    requirements = {
        "role": "초보 수강생의 실습 도움말 작성자",                                  # TODO: 누구를 돕는 작성자인가?
        "task": "제공된 지식으로 질문에 한국어 2문장 이내로 답합니다.",                 # TODO: 어떤 질문에, 어떤 언어·길이로 답할까?
        "input": "사용자 질문과 선택한 지식 목록",                                   # TODO: 질문과 선택 지식
        "constraints": "제공된 근거만 사용하고 예제 ID를 답에 복사하지 않습니다.",      # TODO: 근거 범위와 예제 ID 사용 금지
        "failure": "답을 뒷받침하는 사실이 없으면 정보 부족을 명시합니다.",             # TODO: 없는 사실을 추측하지 않는 처리
        "output": "status, answer, source_ids, missing_information 네 필드 JSON", # TODO: 출력할 네 필드 이름
    }
    # END D21-01-B01
    return requirements

# ============================================================
# 3. Prompt 원문 조립
# ============================================================
# 질문·지식·요구사항·출력 계약을 Template에 대입해 Prompt 원문 문자열을 만든다.
# question     : 사용자 질문 (예: "파일을 어디에 보관하나요?")
# rows         : 이번에 전달할 지식 목록 (예: knowledge_v1의 101 한 건)
# requirements : build_requirements()의 여섯 칸 dict
# 반환값       : [역할]~[출력] 여덟 줄 머리표가 붙은 Prompt 원문
def build_prompt(question, rows, requirements):
    # --------------------------------------------------------
    # 3-1. Template에 질문·지식·계약 대입
    # --------------------------------------------------------
    # TODO D21-01-B02: Template에 질문·직렬화한 지식·출력 계약을 대입하세요.
    # 입력·대상: question, rows, requirements; CONTRACT_TEXT는 제공 코드
    # 완료 기준: 질문·지식 원문·네 필드 약속이 prompt에 있으며 $question이 남지 않습니다.
    # 핵심: 지식 본문은 데이터입니다. Template 대입은 코드 실행 또는 보안 권한 부여가 아닙니다.
    # BEGIN D21-01-B02
    context_json = json.dumps(rows, ensure_ascii=False)  # TODO: rows
    # JSON 중괄호와 충돌하지 않도록 $변수 형식인 Template을 사용합니다.
    template = Template("[역할] $role\n[작업] $task\n[입력] $input\n[질문] $question\n[지식 데이터] $context_json\n[제약] $constraints\n[실패] $failure\n[출력] $contract")
    # template.substitute 인자
    # - **requirements             : 여섯 칸 dict를 role=..., task=... 키워드 인자로 펼침
    # - question=question          : [질문] 자리
    # - context_json=context_json  : [지식 데이터] 자리 (데이터일 뿐 실행 명령이 아님)
    # - contract=CONTRACT_TEXT     : [출력] 자리에 네 필드 계약 문장 9줄
    prompt = template.substitute(
        **requirements,
        question=question,              # TODO: question
        context_json=context_json,      # TODO: 위 context_json
        contract=CONTRACT_TEXT,
    )
    # END D21-01-B02
    return prompt

# ============================================================
# 4. 실행 흐름 (main)
# ============================================================
# 요구사항과 Prompt를 만들어 outputs/에 저장한다.
# requirements.json : 여섯 칸 요구사항
# prompt.txt        : 질문 "파일을 어디에 보관하나요?" + 지식 101 Prompt 원문
def main():
    # --------------------------------------------------------
    # 4-1. 요구사항·지식·Prompt 준비
    # --------------------------------------------------------
    requirements=build_requirements()
    rows=read_json(DATA / "knowledge_v1.json")[:1]
    prompt=build_prompt("파일을 어디에 보관하나요?",rows,requirements)
    # --------------------------------------------------------
    # 4-2. 요구사항·Prompt 원문 저장
    # --------------------------------------------------------
    # TODO D21-01-B03: 요구사항과 Prompt 원문을 저장하고 기준 질문을 확인하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: outputs/requirements.json 및 prompt.txt 생성. 아직 실제 모델 품질 결과는 없습니다.
    # BEGIN D21-01-B03
    assert len(requirements) == 6 and all(requirements.values())
    # TODO: prompt에 질문과 '개인 폴더'가 있는지 확인하세요.
    assert "파일을 어디에 보관하나요?" in prompt and "개인 폴더" in prompt
    write_json(HERE / "outputs" / "requirements.json", requirements)    # TODO: requirements
    write_text(HERE / "outputs" / "prompt.txt", prompt)                 # TODO: prompt
    # END D21-01-B03
    print("[01] Prompt 요구사항과 원문 저장 완료. 모델 결과가 아닙니다.")

if __name__ == "__main__":
    run(main)
