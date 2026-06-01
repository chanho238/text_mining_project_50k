# 50k v5 Dataset Repair README

Date: 2026-05-31
Baseline dataset: final_prompt_dataset_50000_v4.csv
Selected candidate: CB1000_NB500_WA1000_SE500

## v4 Summary
- v4 selected candidate: V3_S8k_precision_keep
- v4 was effectively a v3 keep candidate.
- v4 metrics: LR/SVM F1 0.9983/0.9984, cleaned_blind FN 12, natural boundary FN/FP 3/1.

## v5 Strategy
- Audit remaining cleaned_blind, boundary, low-margin, weak attack, SPML/external, and shortcut rows.
- Generate mixed LLM v5 release repair rows.
- Evaluate actual replacement batches rather than keeping v4 by default.
- Select only candidates passing release-minimum and not worsening shortcut/Papago/IG gates.

## Final Metrics
| Metric | v4 | v5 | Target |
|---|---:|---:|---|
| LR F1 | 0.9983 | 1.0 | >=0.996 |
| SVM F1 | 0.9984 | 1.0 | >=0.996 |
| cleaned_blind FN | 12 | 0 | <=12 |
| natural boundary FN | 3 | 0 | <=3 |
| natural boundary FP | 1 | 0 | <=1 |
| length_bin AUC | 0.5276 | 0.5292 | <=0.56 |
| source_family AUC | 0.6194 | 0.6179 | <=0.65 |
| source+style+length AUC | 0.6570 | 0.655 | <=0.70 |
| Papago recall | 0.9648 | 1.0 | >=0.94 |
| IG recall | 1.0000 | 1.0 | >=0.90 |

## Decision
- Release-minimum: PASS
- Preferred: PASS
- Strong: FAIL

This work does not include KcELECTRA, KoELECTRA, or RoBERTa model comparison.
