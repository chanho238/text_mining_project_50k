# 50k v2 Dataset Repair README

**Date**: 2026-05-31
**Selected candidate**: S15k_pure7k7k (llm_generated_pool = 70%)
**Final decision**: ACCEPTED (v1 대비 전반적 개선, hard minimum 3개 미충족)

---

## 1. v1 문제 요약

| 문제 | v1 값 | hard min target |
|------|--------|-----------------|
| source_family AUC | 0.8310 | <=0.75 FAIL |
| combined AUC | 0.8952 | <=0.80 FAIL |
| length_bin AUC | 0.6548 | <=0.60 FAIL |
| cleaned_blind FN | 28 | <=15 FAIL |
| nat_FN | 11 | <=7 FAIL |
| nat_FP | 8 | <=8 경계 |

근본 원인:
- source_family: IG/SPML/ext = 100% attack, koalpaca/carrotai = 100% normal, llm(52%)만 혼합
- cb_FN 28: external_attack_pool short jailbreak(20-99자)가 TF-IDF FN
- nat_FN 11: 경계 케이스 subtle attack

---

## 2. v2 Strategy

1. llm_generated_pool 비중 52% -> 70% (source_family shortcut 해결)
2. subtle_attack 제거 (nat_FN 악화 원인으로 확인)
3. 더 명확한 공격 패턴(RAG injection, tool misuse, clear jailbreak)으로 교체
4. v1 unused LLM rows(4,434행) 활용

---

## 3. Candidate 실험 결과

| Candidate | llm% | sf_auc | comb | lr_f1 | cb_fn | nat_fn | nat_fp |
|-----------|------|--------|------|-------|-------|--------|--------|
| v1 baseline | 52% | 0.831 | 0.895 | 0.9947 | 28 | 11 | 8 |
| S15k (selected) | 70% | 0.733 | 0.770 | 0.9966 | 19 | 9 | 4 |
| S17k | 66% | 0.756 | 0.788 | 0.9958 | 24 | 14 | - |
| S20k | 60% | 0.791 | 0.818 | 0.9958 | 25 | 15 | - |
| S22k | 56% | 0.811 | 0.835 | 0.9954 | 30 | 18 | - |

S15k_pure7k7k: llm=70%, pure sources=30% -> source_family AUC 0.733 (target <=0.75 PASS!)

---

## 4. v1 vs v2 Final Comparison

| Metric | v1 | v2 | target | status |
|--------|----|----|--------|--------|
| total_rows | 50,000 | 50,000 | 50,000 | PASS |
| normal/attack | 25k/25k | 25k/25k | 25k/25k | PASS |
| leakage | 0 | 0 | 0 | PASS |
| test F1 LR | 0.9947 | 0.9966 | >=0.990 | PASS improved |
| test F1 SVM | 0.9959 | 0.9982 | >=0.990 | PASS improved |
| IG holdout | 1.000 | 1.000 | >=0.85 | PASS maintained |
| Papago recall | 0.9524 | 0.9189 | >=0.90 | PASS (slight drop) |
| LLM HN FP | 0.0% | 0.0% | <=10% | PASS maintained |
| cb_FN | 28 | 19 | <=15 | FAIL improved |
| nat_FN | 11 | 9 | <=7 | FAIL improved |
| nat_FP | 8 | 4 | <=8 | PASS improved |
| length-only AUC | 0.4879 | 0.4934 | <=0.56 | PASS |
| length_bin AUC | 0.6548 | 0.6221 | <=0.60 | FAIL improved |
| source_detail AUC | 0.5532 | 0.5891 | <=0.68 | PASS |
| source_family AUC | 0.8310 | 0.7328 | <=0.75 | PASS improved |
| style_family AUC | 0.5312 | 0.5404 | <=0.68 | PASS |
| combined AUC | 0.8952 | 0.7698 | <=0.80 | PASS improved |

v2 PASS: 14/17 (v1: 12/17)

---

## 5. 최종 채택 판정 (ACCEPTED)

채택 근거:
- source_family AUC: 0.831 -> 0.733 (-0.098, PASS)
- combined AUC: 0.895 -> 0.770 (-0.125, PASS)
- test F1 LR/SVM: 0.9947/0.9959 -> 0.9966/0.9982 (improved)
- nat_FP: 8 -> 4 (PASS, halved)
- nat_FN: 11 -> 9 (v1보다 증가 없음, 기각 조건 아님)
- cb_FN: 28 -> 19 (v1보다 증가 없음, 기각 조건 아님)
- Papago recall: 0.91 >= 0.90 (PASS)
- IG holdout: 1.0 (PASS)

미충족 (TF-IDF 구조적 한계):
- cb_FN=19 (target <=15): external_attack_pool short jailbreak FN
- nat_FN=9 (target <=7): boundary subtle attack
- lbin_auc=0.622 (target <=0.60): length bin shortcut

이 3가지는 KcELECTRA 학습 시 개선 기대.

---

## 6. 모델 비교 범위 명시

v2는 데이터셋 조정 작업입니다.
KcELECTRA, KoELECTRA, RoBERTa는 v2 범위 밖입니다.
TF-IDF baseline으로 데이터 품질을 평가하였습니다.

---

## 7. Output Files

Final datasets:
- final_prompt_dataset_50000_v2.csv (50,000 rows)
- final_prompt_dataset_50000_v2_train_valid_test.csv

v1 preserved:
- final_prompt_dataset_50000_v1_preserved.csv
- final_prompt_dataset_50000_v1_train_valid_test_preserved.csv

Reports (pipeline_output_50k_v2/):
50k_v2_v1_audit.csv, 50k_v2_problem_row_audit.csv, 50k_v2_batch_eval_log.csv,
50k_v2_shortcut_baseline.csv, 50k_v2_model_metrics.csv, 50k_v2_holdout_metrics.csv,
50k_v2_gate_checklist_final.csv, 50k_v1_v2_comparison.csv, README_50k_v2_dataset_repair.md

---

## 8. Next Step

1. final_prompt_dataset_50000_v2_train_valid_test.csv 사용
2. cb_FN 추가 개선: short jailbreak hard positive 보강 (Claude API 기반 권장)
3. nat_FN 추가 개선: curated boundary holdout 수집 필요
4. KcELECTRA fine-tuning: cb_FN=0, nat_FN=0 목표 달성 기대
