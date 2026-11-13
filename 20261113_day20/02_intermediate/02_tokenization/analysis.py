"""지식 문장을 Token·ID로 바꾸고 Padding·Truncation·문맥 예산을 분석해 token_report.json에 저장한다.

- records             : 문장별 Token·ID·개수 (A=8, B=7, C=9, BOS/EOS 포함)
- batch, batch_report : Padding 결과와 실제 길이·PAD 위치
- truncation          : max_length=5, 12 조건의 encode 결과
- language_comparison : 한국어·영어·코드 입력 비교 (심화)
- overflow_budget     : Window 4,096에서 예산 초과 확인 (심화)
"""
# ============================================================
# 1. 경로 설정 및 모듈 import
# ============================================================
from pathlib import Path
import sys

# HERE       : 이 파일이 있는 교시 폴더 (결과는 HERE/outputs 에 저장)
# LEVEL_ROOT : 수준 폴더 (common/ 패키지가 있는 위치)
# sys.path.insert(0, ...) : common.support 를 import할 수 있도록 검색 경로 맨 앞에 추가
HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 수준 공통 모듈 common/support.py: 지식 3건(DATA)·교육용 Tokenizer·문맥 예산 계산·JSON 저장
from common.support import DATA, ToyTokenizer, context_budget, write_json

# ============================================================
# 2. 실습 본문: Token·ID·Mask·문맥 예산 분석
# ============================================================
# 주의: 여기의 ToyTokenizer는 규칙 기반 학습용이며 실제 BERT 서브워드 출력과 다릅니다.
def run_analysis():
    # --------------------------------------------------------
    # 2-1. 교육용 어휘표 준비
    # --------------------------------------------------------
    # texts: 지식 A/B/C의 content 문장 3개
    # ToyTokenizer(말뭉치) : 말뭉치 조각으로 고정 어휘표 생성
    # - texts                 : 지식 문장 3개
    # - "AI", "FastAPI와 PostgreSQL", "print(x) // code" : Padding·코드 비교용 추가 예문
    # 예: "[BOS]"=2, "[EOS]"=3, "."=6 (특수 Token 4개 뒤에 정렬된 조각이 4번부터)
    texts = [item["content"] for item in DATA]
    tok = ToyTokenizer(texts + ["AI", "FastAPI와 PostgreSQL", "print(x) // code"])

    # --------------------------------------------------------
    # 2-2. 지식 A/B/C 문장의 Token·ID 변환
    # --------------------------------------------------------
    # BEGIN D20-02-I01
    # TODO D20-02-I01 완료: 제공된 지식 A/B/C 세 문장을 학습용 Token·Token ID로 변환하세요.
    # rows 원소 키 : 의미
    # - id            : 지식 ID (A, B, C)
    # - text          : 원문 문장
    # - tokens        : Token 문자열 (예: A -> ['[BOS]', '실습', '파일은', '개인', '폴더에', '보관합니다', '.', '[EOS]'])
    # - ids           : Token ID (예: A -> [2, 19, 25, 14, 26, 16, 6, 3])
    # - count         : BOS/EOS 포함 Token 수 (A=8, B=7, C=9)
    # - special_count : 특수 Token 수 ([BOS], [EOS] 2개)
    rows=[]
    for item in DATA:
        # tok.encode(문장) : Token·ID·Mask·길이 정보를 담은 dict 반환
        record=tok.encode(item["content"])
        rows.append({"id":item["id"], "text":item["content"],
                     "tokens":record["tokens"], "ids":record["input_ids"],
                     "count":record["valid_length"],"special_count":2})
    # 해설: [BOS]/[EOS]는 이 교육용 토크나이저의 규칙이며 실제 모델마다 다릅니다.
    # END D20-02-I01

    # --------------------------------------------------------
    # 2-3. 길이가 다른 두 입력 Padding
    # --------------------------------------------------------
    # BEGIN D20-02-I02
    # TODO D20-02-I02 완료: 길이가 다른 두 입력을 Padding하고 attention_mask의 0/1을 확인하세요.
    # tok.batch([...]) : 가장 긴 입력 길이로 모두 Padding
    # - "AI"                   : 실제 3 Token -> [PAD] 2개 추가, attention_mask=[1, 1, 1, 0, 0]
    # - "FastAPI와 PostgreSQL" : 실제 5 Token -> Padding 없음, attention_mask=[1, 1, 1, 1, 1]
    batch = tok.batch(["AI", "FastAPI와 PostgreSQL"])
    # 해설: batch의 모든 input_ids 길이는 같고 [PAD] 위치의 attention_mask는 0입니다.
    # 확인: 두 입력의 길이가 같고(5), 각 입력의 ID 수와 Mask 수가 같다
    assert len(batch[0]["input_ids"]) == len(batch[1]["input_ids"])
    assert all(len(b["input_ids"]) == len(b["attention_mask"]) for b in batch)
    # END D20-02-I02

    # --------------------------------------------------------
    # 2-4. 문맥 예산 계산
    # --------------------------------------------------------
    # BEGIN D20-02-I03
    # TODO D20-02-I03 완료: PPT 문맥 예시(300+1200+2000+150, 출력 예약 850)의 합과 남는 예산을 계산하세요.
    # context_budget 인자
    # - window=5000           : 한 요청의 Context Window 크기 (PPT 예시)
    # - system=300            : 시스템 지시 Token
    # - history=1200          : 이전 대화 Token
    # - evidence=2000         : 검색 근거 Token
    # - user=150              : 사용자 질문 Token
    # - output_reserved=850   : 답변 출력용으로 비워 둘 Token
    # 결과: input_used=3650, remaining=500, fits=True
    budget=context_budget(5000,{"system":300,"history":1200,"evidence":2000,"user":150},850)
    # 해설: 입력 3,650 + 출력 예약 850 = 4,500; 예시 창 5,000에서 500 남음.
    # END D20-02-I03

    # --------------------------------------------------------
    # 2-5. max_length 두 조건의 Truncation 비교
    # --------------------------------------------------------
    # BEGIN D20-02-I04
    # TODO D20-02-I04 완료: max_length=5와 12의 Truncation 여부·실제 길이를 비교하세요.
    # trunc_cases 키 = max_length 문자열 ("5", "12"), 값 = encode 결과 dict
    # - max_length=5  : 원래 길이 5와 같으므로 잘리지 않음 (truncated=False)
    # - max_length=12 : 상한보다 짧아 그대로 (truncated=False)
    # 주의: 이 입력에서는 두 조건 모두 잘림이 일어나지 않는다. 잘림을 보려면 5보다 작은 값(예: 4)이 필요하다
    trunc_cases={
        str(n):tok.encode("FastAPI와 PostgreSQL",max_length=n)
        for n in (5,12)
    }
    # 해설: max_length는 이 예제에서 BOS/EOS를 포함한 입력 상한입니다.
    # END D20-02-I04

    # --------------------------------------------------------
    # 2-6. Padding 결과의 실제 길이·PAD 위치 정리
    # --------------------------------------------------------
    # BEGIN D20-02-I05
    # TODO D20-02-I05 완료: 실제 Token 수와 패딩 포함 Tensor 길이를 나누어 기록하세요.
    # batch_report 원소 키 : 의미
    # - valid         : PAD를 뺀 실제 Token 수 (예: "AI" -> 3)
    # - padded        : Padding 후 전체 길이 (예: 5)
    # - pad_positions : attention_mask=0인 위치 (예: [3, 4])
    batch_report=[{
        "valid":it["valid_length"], "padded":len(it["input_ids"]),
        "pad_positions":[i for i,v in enumerate(it["attention_mask"]) if v==0],
    } for it in batch]
    # END D20-02-I05
    # 심화 전용 결과 자리: 이 수준에서는 빈 값으로 JSON에 기록
    mixed_rows = []
    small_budget = None
    policy = "예산과 잘림의 차이를 설명"

    # --------------------------------------------------------
    # 2-7. 결과 JSON 저장
    # --------------------------------------------------------
    # BEGIN D20-02-I06
    # TODO D20-02-I06 완료: Token·Mask·문맥 예산 결과를 outputs/token_report.json에 저장하세요.
    # result 키 : 의미
    # - tokenizer_kind      : 사용한 Tokenizer 종류 (사전학습 모델 아님)
    # - records             : 문장별 Token·ID·개수
    # - batch               : Padding 배치
    # - budget              : Window 5,000 예산 계산
    # - truncation          : max_length 조건별 결과 (응용·심화)
    # - batch_report        : 실제 길이·PAD 위치 (응용·심화)
    # - language_comparison : 한국어·영어·코드 비교 (심화)
    # - overflow_budget     : Window 4,096 예산 초과 결과 (심화)
    # - policy              : 예산 초과 대응 규칙
    result={"tokenizer_kind":"교육용 regex·고정 어휘표 (사전학습 모델이 아님)",
            "records":rows,"batch":batch,"budget":budget,"truncation":trunc_cases,
            "batch_report":batch_report,"language_comparison":mixed_rows,
            "overflow_budget":small_budget,"policy":policy}
    write_json(HERE/"outputs"/"token_report.json",result)
    # 콘솔 확인: [02] [('A', 8), ('B', 7), ('C', 9)] remaining 500
    print("[02]",[(x["id"],x["count"]) for x in rows],"remaining",budget["remaining"])
    # END D20-02-I06

