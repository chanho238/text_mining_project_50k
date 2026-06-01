# 50k v7 Robustness Hardening

Date: 2026-05-31
Base dataset: final_prompt_dataset_50000_v7.csv

## Summary
- Final artifact rows in dataset before repair: 1
- Candidate-pool-only artifact rows: 832
- Artifact rows in robustness final dataset: 0
- Target hardening weak/missing styles before: 2
- Target hardening weak/missing styles after: 2

## Metrics
- LR/SVM F1: 1.0 / 0.9997
- cleaned_blind FN: 0
- natural boundary FP/FN: 0 / 0
- IG/Papago holdout: 1.0 / 1.0
- length_bin/source_family/combined AUC: 0.5222 / 0.6115 / 0.6491

## Unseen Indirect Holdout
- Size: 1000
- Accuracy: 0.547
- Attack recall: 0.083
- Normal recall: 1.0

IG/Papago holdout에서 1.0 recall을 달성했지만, 이는 현재 구성된 holdout 기준에서 안정적으로 탐지된다는 의미입니다. 모든 미래 공격을 완벽하게 탐지한다는 의미는 아닙니다.

Unseen indirect holdout을 통해 추가 일반화 가능성을 별도 평가했습니다. 본 결과는 텍스트 기반 한국어 prompt injection 범위에서의 실험 결과이며, 실제 에이전트 실행환경과 장기 메모리 공격은 추가 검증이 필요합니다.
