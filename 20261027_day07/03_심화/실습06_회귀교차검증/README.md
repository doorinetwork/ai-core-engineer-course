# 실습 06 · 회귀 5-fold

**목표:** 음수 scorer, 양의 오차, fold 평균을 구분합니다.

**실제 정답 실행으로 확인한 출력**

```text
 model  CV_RMSE_ms  CV_std_ms
  Tree     599.635     96.116
Forest     624.011     96.974
   XGB     650.892     92.295
CV_RMSE는 fold별 RMSE의 단순 평균입니다. 전체 예측을 합친 RMSE와 다를 수 있습니다.
```

**생성 파일:** `outputs/regression_cv.csv`, `outputs/cv_scores.csv`

**완료·기록:** 

- 분류에서 쓰던 StratifiedKFold가 아니라 KFold를 쓰는 이유는 무엇인가요?  

- 단위 ms를 표에 남겨야 하는 이유는 무엇인가요?  
