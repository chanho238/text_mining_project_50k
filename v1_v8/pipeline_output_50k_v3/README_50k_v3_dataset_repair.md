# 50k v3 Dataset Repair README

Date: 2026-05-31
Baseline dataset: final_prompt_dataset_50000_v2.csv
Selected candidate: S8k_pure4k4k

## v2 Summary
- Rows: 50,000, normal/attack: 25,000/25,000
- v2 LR/SVM F1: 0.9966 / 0.9982
- v2 remaining failures: cleaned_blind FN 19, natural boundary FN 9, length_bin AUC 0.6221
- v2 shortcut residuals: source_family AUC 0.7328, source+style+length AUC 0.7698

## v3 Strategy
- Preserve the v2 files before repair.
- Audit v2 problem rows and replacement priorities.
- Build label-balanced mixed LLM source_detail groups instead of label-pure LLM source details.
- Keep pure public/source rows at a smaller but non-zero quota.
- Regenerate train/validation/test split with exact 35,000 / 7,500 / 7,500 counts.
- Evaluate only TF-IDF/LR/SVM and prompt-derived shortcut baselines.

## Final Metrics
| Metric | v2 | v3 | Target |
|---|---:|---:|---|
| LR F1 | 0.9966 | 0.9983 | >=0.990 |
| SVM F1 | 0.9982 | 0.9984 | >=0.990 |
| cleaned_blind FN | 19 | 12 | <=15 |
| natural boundary FN | 9 | 3 | <=7 |
| natural boundary FP | 4 | 1 | <=4 |
| length_bin AUC | 0.6221 | 0.5276 | <=0.60 |
| source_family AUC | 0.7328 | 0.6194 | <=0.72 |
| source+style+length AUC | 0.7698 | 0.657 | <=0.76 |
| Papago recall | 0.9189 | 0.9648 | >=0.90 |
| IG recall | 1.0000 | 1.0 | >=0.85 |
| LLM HN FP ratio | 0.0000 | 0.0 | <=0.10 |

## Decision
- Hard minimum: PASS
- Preferred: FAIL
- Strong: FAIL

## Notes
- This is not a KcELECTRA, KoELECTRA, or RoBERTa comparison.
- Metadata was used for audits and split reports, not as a classifier feature in the LR/SVM text model.
- Synthetic attack rows use placeholders such as [UNSAFE_CONTENT], [UNSAFE_URL], and [REDACTED_ACTION] instead of operational harmful detail.
