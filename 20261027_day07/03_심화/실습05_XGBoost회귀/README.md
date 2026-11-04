# 실습 05 · XGBoost 회귀

**목표:** 분류·회귀 목적함수, 정답, 지표의 짝을 설명합니다.

**실제 정답 실행으로 확인한 출력**

```text
model  MAE_ms  RMSE_ms
  XGB 610.197   700.41
고정 80회차·깊이2입니다. valid를 조기 종료에 사용하지 않았습니다.
```

**생성 파일:** `outputs/xgb_regression.csv`

**완료·기록:** 

- binary:logistic과 reg:squarederror를 각각 어떤 정답에 쓰나요?  
  binary:logistic은 범주형 종속변수에 사용한다. 이 실습에서는 failed에 해당한다.  
  reg:squarederror는 연속형 종속변수에 사용한다. 이 실습에서는 elapsed_ms에 해당한다.
