# 실습 5 · Coverage·Gap·중복 ID (set = overlap · missing · duplicate) 

> **보고서 ID:** D24-B05-CVR-GAP 
> **작성 기준일:** 2026-11-19  
> **검증 범위:** `01_basic/05_validation/main.py`의 TODO 완성본, `outputs/coverage_report.json`

## 처리 결과
 
| set | coverage | covered_characters | repeat_extra_chars | gaps | duplicate_ids | valid |
|:--|--:|--:|--:|:--|:--|:--|
| overlap | 1.0 | 20 | 4 | [] | [] | true |
| missing | 0.9 | 18 | 0 | [[8, 10]] | [] | false |
| duplicate | 1.0 | 20 | 8 | [] | ["A"] | false |

## invalid: uncoverred
missing set의 경우 [8, 10]에 해당하는 글자 2개가 덮이지 않았다.

## invalid: coverage = 1.0 이지만 duplicated
duplicate set의 경우 coverage는 100%로 나왔지만, 중복 ID "A"가 보고된다. 

## 해석
- coverage는 고유 위치의 합집합 비율이므로 의도한 Overlap(반복 4자)은 coverage를 올리지 않는다.
- coverage만 관측해서는 중복 ID를 체크할 수 없다.
- 검증 과정에서 사람의 근거 판단이 요구된다.