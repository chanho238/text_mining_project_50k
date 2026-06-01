# 50k v8 Unseen-Indirect Robustness Repair

Date: 2026-05-31
Selected candidate: U3000_T1500_A

## Why v8
The v7 release-plus dataset had excellent internal gates, but the first unseen indirect holdout mixed defensive requests with true attacks. It produced attack recall 0.083, so v8 separates label audit, clean holdout rebuilding, and generalized training support.

## Holdout Separation
Unseen holdout is final-evaluation only and was not included in train/validation/test. v8 training support uses generalized threat-family pairs with different wording from the clean holdout.

## Final Metrics
- LR/SVM F1: 1.0 / 0.9997
- IG/Papago: 1.0 / 1.0
- cleaned_blind FN: 0
- natural boundary FP/FN: 0 / 0
- length/source/combined AUC: 0.5224 / 0.6115 / 0.6551
- clean unseen accuracy: 1.0
- clean unseen attack recall: 1.0
- clean unseen normal recall: 1.0

## Limitations
v8 improves text-based Korean prompt-injection robustness in this dataset setting. It is not a guarantee for all future attacks. Real agent execution environments, long-memory inputs, and tool side effects need additional validation.
