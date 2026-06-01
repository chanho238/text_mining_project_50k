# 50k v7 Dataset Release-Plus Refinement README

Date: 2026-05-31
Baseline dataset: final_prompt_dataset_50000_v6.csv
Selected candidate: SF250_COMB250_L250

## v6 Summary
- v6 selected candidate: L500_RS250_B100
- Perfect gates were preserved: cleaned_blind FN 0, natural boundary FN/FP 0/0, Papago 1.0, IG 1.0.
- Preferred/strong failed only on very strict residual shortcut thresholds.

## v7 Strategy
- Audit residual high-purity source, style, and length cells.
- Evaluate conservative SF/COMB/L release-plus candidates.
- Reject candidates that reintroduce boundary or holdout regression.

## Final Metrics
| Metric | v6 | v7 |
|---|---:|---:|
| length_bin AUC | 0.5236 | 0.5222 |
| source_family AUC | 0.6155 | 0.6115 |
| source+style+length AUC | 0.6532 | 0.6485 |
| source_detail AUC | 0.5324 | 0.5304 |
| style_family AUC | 0.5622 | 0.5596 |
| cleaned_blind FN | 0 | 0 |
| natural boundary FN/FP | 0/0 | 0/0 |
| Papago / IG | 1.0 / 1.0 | 1.0 / 1.0 |

## Decision
- Release-minimum: PASS
- Preferred: FAIL
- Strong: FAIL

This work does not include KcELECTRA, KoELECTRA, or RoBERTa model comparison.
