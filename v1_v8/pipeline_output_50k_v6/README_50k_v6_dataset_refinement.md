# 50k v6 Dataset Refinement README

Date: 2026-05-31
Baseline dataset: final_prompt_dataset_50000_v5.csv
Selected candidate: L500_RS250_B100

## v5 Summary
- v5 selected candidate: CB1000_NB500_WA1000_SE500
- v5 solved cleaned_blind and natural boundary errors: FN/FP all zero.
- Strong gate failed mainly because length_bin AUC was 0.5292 and combined AUC was 0.6550.

## v6 Strategy
- Preserve v5 perfect boundary and holdout gates.
- Test conservative length-bin and residual-shortcut refinements.
- Reject any candidate that reintroduces cleaned_blind FN or natural boundary FP/FN.

## Final Metrics
| Metric | v5 | v6 | Target |
|---|---:|---:|---|
| LR F1 | 1.0 | 1.0 | >=0.998 |
| SVM F1 | 1.0 | 1.0 | >=0.998 |
| cleaned_blind FN | 0 | 0 | 0 |
| natural boundary FN | 0 | 0 | 0 |
| natural boundary FP | 0 | 0 | 0 |
| length_bin AUC | 0.5292 | 0.5236 | <=0.5292 |
| source_family AUC | 0.6179 | 0.6155 | <=0.62 |
| source+style+length AUC | 0.6550 | 0.6532 | <=0.655 |
| Papago recall | 1.0 | 1.0 | >=0.96 |
| IG recall | 1.0 | 1.0 | >=0.95 |

## Decision
- Release-minimum: PASS
- Preferred: FAIL
- Strong: FAIL

This work does not include KcELECTRA, KoELECTRA, or RoBERTa model comparison.
