# 실습 03 예상 결과·검증 기준

실습 목적: Top-k 검색 품질 지표의 분자·분모를 직접 계산합니다.

- `outputs/result.json`에서 `exercise=3`, `level=02_intermediate` 확인
- `mode=offline_fixture`는 **교육용 고정 데이터**, `mode=live`만 실제 서비스 실행
- 단계별 `question_id`, Gold, 후보/Context, 출처는 반드시 구분
- TODO ID는 수강생 코드와 강사용 코드에서 같은 함수에 대응
- 비교와 실행시간은 **같은 데이터·버전·실행 모드**끼리만 논의
- 정답 ID는 Target에 직접 전달하지 않습니다.

## 강의 체크

Recall@k · Precision@k · Hit@k: Recall의 분모는 전체 Gold 개수, Precision의 분모는 k입니다.

## 흔한 오개념

- No-answer를 Recall 0으로 합산하면 평균이 왜곡됩니다.
- 모델이 반환한 Citation ID만으로 Faithfulness가 자동 검증되지는 않습니다.
- 오프라인 Fixture의 Latency는 실측 지표가 아닙니다.
