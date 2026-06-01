# 50k v4 Dataset Repair README

Date: 2026-05-31
Baseline dataset: final_prompt_dataset_50000_v3.csv
Selected candidate: V3_S8k_precision_keep

## v3 Summary
- v3 selected candidate: S8k_pure4k4k
- v3 hard-minimum: PASS
- Remaining targets: cleaned_blind FN 12, natural boundary FN/FP 3/1, preferred/strong FAIL

## v4 Strategy
- Preserve v3 files.
- Audit remaining cleaned_blind, natural boundary, low-margin, and residual shortcut rows.
- Generate a small mixed-source v4 precision pool.
- Keep the accepted v3 structure unless a small precision batch passes gates.
- Evaluate only TF-IDF/LR/SVM and shortcut baselines.

## Final Metrics
| Metric | v3 | v4 | Target |
|---|---:|---:|---|
| LR F1 | 0.9983 | 0.9983 | >=0.995 |
| SVM F1 | 0.9984 | 0.9984 | >=0.995 |
| cleaned_blind FN | 12 | 12 | <=12 |
| natural boundary FN | 3 | 3 | <=3 |
| natural boundary FP | 1 | 1 | <=1 |
| length_bin AUC | 0.5276 | 0.5276 | <=0.56 |
| source_family AUC | 0.6194 | 0.6194 | <=0.65 |
| source+style+length AUC | 0.6570 | 0.657 | <=0.70 |
| Papago recall | 0.9648 | 0.9648 | >=0.92 |
| IG recall | 1.0000 | 1.0 | >=0.85 |

## Decision
- Hard minimum: PASS
- Preferred: FAIL
- Strong: FAIL

This work does not include KcELECTRA, KoELECTRA, or RoBERTa model comparison.
