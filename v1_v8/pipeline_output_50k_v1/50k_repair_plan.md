# 50k v1 Repair Plan

## Bug Fixes from v38
1. DataFrame bool 오류 방지: `if best_result is not None:` 사용
2. _gate_ok: nat_FP > 1 → 기각, nat_FN > 0 → 기각
3. batch별 재학습 모델로 gate 평가 (기존 모델 재사용 금지)

## Repair Triggers
- IG holdout recall < 0.85 → IG FN cluster hard positive 추가
- short GN 부족 → short_general_question LLM 생성
- llm_hard_negative FP > 10% → hard negative 품질 필터 강화
- source+style+length AUC > 0.72 → source 분포 재조정
- natural boundary FN > 0 → 즉시 기각, 원인 분석

## Priority Repair Actions
1. LLM attack 생성 (RAG/tool/context/policy attack)
2. normal hard negative LLM 생성 (privacy/SNS/RAG)
3. short GN 20_99 구간 추가
4. FN cluster 기반 hard positive 수집
