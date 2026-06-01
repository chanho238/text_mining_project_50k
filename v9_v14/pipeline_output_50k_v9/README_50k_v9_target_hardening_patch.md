# 50k v9 Target-Hardening Patch

Date: 2026-05-31
Selected candidate: T696_A3_shortN100

v8 is already an accepted baseline. v9 is not a full rebuild; it is a hardening patch for target-hardening coverage and artifact-like repetitive patterns.

The v8 clean unseen holdout is final-evaluation only and was not included in train/validation/test. v9 support rows are generalized training examples, not copied or paraphrased holdout prompts.

Numeric prefixes and placeholders are not treated as automatic errors. v9 selectively replaces high-risk repetitive groups that may create label or template shortcuts.

Additional SVM normal-FP support was added for short ordinary questions such as culture, economics, daily knowledge, science, history, and general concept explanations. Existing validation/test normal questions were retained unless there was a clear label, artifact, duplicate, or leakage issue.

## Final Metrics
- LR/SVM F1: 1.0 / 0.9999
- IG/Papago: 1.0 / 1.0
- cleaned_blind FN: 0
- natural boundary FP/FN: 0 / 0
- length/source/combined AUC: 0.5224 / 0.6116 / 0.6545
- target weak/missing styles: 0
- clean unseen accuracy: 1.0
- clean unseen attack/normal recall: 1.0 / 1.0

## Limitations
This is dataset hardening for text-based Korean prompt-injection detection. It does not guarantee safety for all future attacks, real agent execution environments, memory systems, or tool side effects.
