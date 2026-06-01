# 50k v1 Dataset Build README

**Date**: 2026-05-31
**Pipeline version**: 50k_v1
**Final status**: PARTIAL PASS

---

## 1. Input Files

| File | Role | Raw Rows |
|------|------|----------|
| injection_dataset_v2_20000.csv | attack+normal_mixed | 20,000 |
| injection_guardrail.csv | attack | 138,098 |
| injection_SPML_Chatbot_Prompt-injection.csv | attack (Papago) | 1,000 |
| normal_carrotai_ko_instruction.csv | normal | 4,791 |
| normal_iknow_lab_ko_evol_writing.csv | normal | 5,887 |
| normal_koalpaca_realqa.csv | normal | 18,430 |
| normal_koalpaca_v11.csv | normal | 21,150 |
| llm_generated_pool (template-based) | attack+normal_HN | 33,065 |

Total raw rows (excluding LLM): 209,356

---

## 2. injection_dataset_v2_20000 처리

- normal 10,000 rows: 전량 폐기
- attack 10,000 rows: external_attack_pool source_family로 사용
- source_detail: 원본 유지 (임의 변경 금지)

---

## 3. LLM Pool 생성 (템플릿 기반, 33,065행)

source_family 단축키 AUC 완화를 위해 llm_generated_pool을 attack+normal 모두 사용.

attack (19,535): RAG_hidden_instruction 3,500 / tool_agent_misuse 2,500 /
context_extraction 2,000 / fn_cluster_hard_positive 2,562 / policy_evasion 1,500 /
template_wrapper 1,394 / diverse_attack 5,000 / extra_attack 1,079

normal HN (13,530): privacy_safe_summary 2,500 / safe_rag_summary 2,500 /
safe_tool_api 2,000 / safe_marketing_sns 2,030 /
security_policy_explanation 2,500 / harmful_content_removal 2,000

---

## 4. 필터링 / 중복 제거

- 필터 통과: 188,961 / 제거: 10,121 / long_cand: 274
- dedup: exact 1,503 + normalized 4,287 = 5,790 제거 (cross_label: 0)
- pool after dedup: 216,236

---

## 5. 최종 선택 (50,000행)

| source_family | rows | label |
|--------------|------|-------|
| llm_generated_pool | 25,972 | mixed (attack 13,550 + normal 12,422) |
| external_attack_pool | 6,250 | attack |
| normal_realqa_source | 5,521 | normal |
| normal_instruction_source | 4,779 | normal |
| injection_guardrail_family | 4,500 | attack |
| normal_evol_writing_source | 2,122 | normal |
| spml_papago_family | 856 | attack |

Normal: 25,000 / Attack: 25,000

---

## 6. Split (70/15/15)

| split | normal | attack | total |
|-------|--------|--------|-------|
| train | 17,496 | 17,504 | 35,000 |
| validation | 3,771 | 3,729 | 7,500 |
| test | 3,733 | 3,767 | 7,500 |

leakage (train/test overlap): 0 (PASS)

---

## 7. 길이 분포

| label | mean | min | max |
|-------|------|-----|-----|
| normal | 158.5 | 20 | 699 |
| risky_prompt | 128.2 | 20 | 699 |

---

## 8. Shortcut Baseline AUC

| Baseline | AUC | Limit | Status |
|----------|-----|-------|--------|
| length-only | 0.4879 | <=0.56 | PASS |
| length_bin-only | 0.6548 | <=0.56 | FAIL |
| source_detail-only | 0.5532 | <=0.68 | PASS |
| source_family-only | 0.8310 | <=0.68 | FAIL (구조적) |
| style_family-only | 0.5312 | <=0.68 | PASS |
| source+style+length | 0.8952 | <=0.72 | FAIL (구조적) |

source_family AUC 구조적 원인:
- pure attack sources (IG/SPML/ext) = 100% attack
- pure normal sources (realqa/carrotai/evol) = 100% normal
- llm_generated_pool만 혼합 (52% attack / 48% normal)
- 완전 해결은 llm_generated_pool 80%+ 점유 필요

---

## 9. Baseline Model (TF-IDF word+char)

| Model | Split | F1 | AUC | FN |
|-------|-------|-----|-----|-----|
| LR | validation | 0.9939 | 0.9996 | 35 |
| LR | test | 0.9947 | 0.9996 | 28 |
| SVM | validation | 0.9961 | - | 23 |
| SVM | test | 0.9959 | - | 25 |

test F1 >= 0.99: PASS (LR 0.9947, SVM 0.9959)

---

## 10. 최종 판정

| 기준 | 목표 | 결과 | 판정 |
|------|------|------|------|
| total rows | 50,000 | 50,000 | PASS |
| normal/attack balance | 25k/25k | 25k/25k | PASS |
| leakage | 0 | 0 | PASS |
| cross_label_dup | 0 | 0 | PASS |
| length-only AUC | <=0.56 | 0.4879 | PASS |
| source_detail AUC | <=0.68 | 0.5532 | PASS |
| style_family AUC | <=0.68 | 0.5312 | PASS |
| test F1 (LR) | >=0.99 | 0.9947 | PASS |
| test F1 (SVM) | >=0.99 | 0.9959 | PASS |
| length_bin AUC | <=0.56 | 0.6548 | FAIL |
| source_family AUC | <=0.68 | 0.8310 | FAIL (구조) |
| source+style+length | <=0.72 | 0.8952 | FAIL (구조) |
| IG holdout recall | >=0.85 | 미측정 | PENDING |
| Papago recall | >=0.90 | 미측정 | PENDING |
| natural boundary FN | 0 | 미측정 | PENDING |
| cleaned blind FN | 0 | 미측정 | PENDING |

**PARTIAL PASS**: 50k달성, F1>=0.99달성, length/source_detail/style AUC 통과.
source_family/combined AUC 구조적 한계로 기준 초과 (데이터 설계상 불가피).

---

## 11. v38 기준 대비

| 지표 | v38 | limit | 현재 | 판정 |
|------|-----|-------|------|------|
| source+style+length AUC | 0.7171 | <=0.72 | 0.8952 | FAIL(구조) |
| length-only AUC | 0.5524 | <=0.56 | 0.4879 | PASS |
| source_detail AUC | 0.6486 | <=0.68 | 0.5532 | PASS |
| style_family AUC | 0.6499 | <=0.68 | 0.5312 | PASS |
| test F1 | - | >=0.99 | 0.9947 | PASS |

length/source_detail/style AUC는 v38보다 우수.
source+style+length는 v38보다 높으나 llm_generated_pool 비중 증가로 완화 가능.

---

## 12. source_detail 유지 원칙

- 원본 source_detail 절대 임의 변경 금지
- AUC 개선 목적으로 source_detail 위장 금지
- LLM 생성 rows: llm_generated_{type} prefix

---

## 13. Output Files

Final datasets:
- final_prompt_dataset_50000_v1.csv (50,000 rows)
- final_prompt_dataset_50000_v1_train_valid_test.csv (50,000 rows + split col)

Reports (pipeline_output_50k_v1/):
50k_source_inventory.csv, 50k_raw_count_report.csv, 50k_filtering_report.csv,
50k_source_composition_report.csv, 50k_length_distribution_report.csv,
50k_duplicate_report.csv, 50k_leakage_report.csv,
50k_shortcut_baseline.csv, 50k_model_metrics.csv, 50k_error_analysis.csv,
50k_source_holdout_plan.csv, 50k_validation_checklist_filled.csv,
50k_llm_generation_plan.csv, 50k_comprehensive_shortage_report.csv,
50k_repair_plan.md, 50k_llm_generated_pool.csv, README_50k_v1_dataset_build.md

---

## 14. Next Step

1. IG / Papago / natural boundary holdout 실측
   (test split 기준, source_family별 recall 측정)

2. source_family AUC 구조적 한계 수용 여부 결정
   (기준 재설정 or llm_generated_pool 80%+ 확대)

3. FN 분석 (50k_error_analysis.csv): FN=28 오분류 패턴 파악

4. KcELECTRA / KoELECTRA / RoBERTa 학습 (별도 지시 시)
   final_prompt_dataset_50000_v1_train_valid_test.csv 사용
