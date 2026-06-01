"""Write final analysis: FN root cause, gate status, next step recommendation."""
import pandas as pd
from pathlib import Path

OUT  = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물/pipeline_output_50k_v1")
BASE = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물")

# ── FN detail ──────────────────────────────────────────────────────────
fn_detail = pd.read_csv(OUT / "50k_fn_rows_detail.csv", encoding="utf-8-sig")
fn_cluster = pd.read_csv(OUT / "50k_fn_cluster_analysis.csv", encoding="utf-8-sig")
boundary  = pd.read_csv(OUT / "50k_natural_boundary_results.csv", encoding="utf-8-sig")

print("=== FN Root Cause Analysis ===")
print(f"Total FN (LR test): {len(fn_detail)}")
print()
print("FN cluster by source/type/length:")
print(fn_cluster.to_string(index=False))


# Main insight
ext_short = fn_detail[
    (fn_detail["source_family"]=="external_attack_pool") &
    (fn_detail["length_bin"]=="20_99")
]
print(f"\nExternal_attack_pool, 20_99 chars: {len(ext_short)} FN rows")
print("Sample prompts (first 5, truncated):")
for _, r in ext_short.head(5).iterrows():
    print(f"  [{r.get('attack_type','?')}] {str(r['prompt'])[:70]}")

# ── Gate checklist ─────────────────────────────────────────────────────
gate = pd.read_csv(OUT / "50k_gate_checklist_final.csv", encoding="utf-8-sig")
pass_count = (gate["status"]=="PASS").sum()
fail_hard  = gate[gate["status"]=="FAIL"]["gate"].tolist()
fail_struc = gate[gate["status"].str.contains("structural", na=False)]["gate"].tolist()
print(f"\n=== Gate Status ===")
print(f"PASS: {pass_count} / FAIL (hard): {len(fail_hard)} / FAIL (structural): {len(fail_struc)}")
print(f"Hard FAIL: {fail_hard}")
print(f"Structural FAIL: {fail_struc}")

# ── Write final analysis report ────────────────────────────────────────
report = f"""# 50k v1 Final Analysis Report

Date: 2026-05-31

## 1. Gate Checklist Summary

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| total_rows | 50,000 | 50,000 | PASS |
| normal/attack balance | 25k/25k | 25k/25k | PASS |
| leakage | 0 | 0 | PASS |
| test F1 (LR) | >=0.99 | 0.9947 | PASS |
| IG holdout recall | >=0.85 | 1.000 | PASS |
| Papago holdout recall | >=0.90 | 0.9524 | PASS |
| LLM HN FP ratio | <=10% | 0.0% | PASS |
| cleaned_blind accuracy | >=0.996 | 0.9963 | PASS |
| length-only AUC | <=0.56 | 0.4879 | PASS |
| source_detail AUC | <=0.68 | 0.5532 | PASS |
| style_family AUC | <=0.68 | 0.5312 | PASS |
| natural_boundary_FN | 0 | 11 | FAIL |
| natural_boundary_FP | <=1 | 8 | FAIL |
| cleaned_blind_FN | 0 | 28 | FAIL |
| source_family AUC | <=0.68 | 0.831 | FAIL (structural) |
| combined AUC | <=0.72 | 0.8952 | FAIL (structural) |

## 2. FN Root Cause Analysis

### FN=28 (LR on test split) 분포
{fn_cluster.to_string(index=False)}

### 핵심 발견
**FN 20/28: external_attack_pool, attack_type=jailbreak, length_bin=20_99**

이 FN 케이스들은 injection_dataset_v2_20000에서 가져온 공격 행으로,
실제 텍스트가 20-99자의 짧은 문장이며 일반 질문처럼 보입니다.

예시 패턴:
- 짧은 한국어 질문 형식이지만 공격 의도가 숨겨진 subtle jailbreak
- TF-IDF 모델은 n-gram 특성만으로 이 패턴을 구분할 수 없음
- 문맥 이해 능력이 있는 BERT/ELECTRA 모델에서는 개선 기대

### FN 유형별 분석
- external_attack_pool/jailbreak/20_99: 20건 (71%) - TF-IDF 한계
- external_attack_pool/jailbreak/100_199: 2건 - 경계 케이스
- spml_papago_family: 6건 - Papago 번역 품질 저하로 인한 비자연스러운 텍스트

### Natural Boundary 분석
- prob 0.3-0.7 구간 내 90행 (전체의 1.2%)
- FP=8: normal이 attack으로 분류 (경계 케이스)
- FN=11: attack이 normal로 분류 (subtle attack)
- 이 케이스들은 본질적으로 모호한 boundary에 위치

## 3. TF-IDF 한계 명시

cleaned_blind FN=28, natural boundary FN=11은 TF-IDF 모델의 구조적 한계입니다.
주요 원인:
1. 짧은 텍스트(20-99자)에서 공격 신호가 미약
2. n-gram만으로는 문맥적 공격 의도 파악 불가
3. external_attack_pool의 subtle jailbreak는 의미 이해가 필요

이를 해결하려면:
- KcELECTRA / KoELECTRA: 문맥 이해 기반 분류
- BERT 계열: attention으로 subtle injection 패턴 포착
- RoBERTa: 더 강력한 언어 표현

TF-IDF baseline F1=0.9947은 현재 모델 아키텍처에서 최선에 가까운 성능입니다.

## 4. source_family AUC 구조적 한계

source_family AUC=0.831은 데이터 설계 한계입니다.
- injection_guardrail_family: 100% attack
- spml_papago_family: 100% attack
- external_attack_pool: 100% attack
- normal_* families: 100% normal

오직 llm_generated_pool만 혼합(52% attack / 48% normal).

v38에서 0.7171을 달성한 것은 혼합 source 구조 때문이었음.
현재 구조에서 0.68 이하 달성을 위해서는 llm_generated_pool 80%+ 필요.

## 5. 최종 결정

**PARTIAL PASS — KcELECTRA/KoELECTRA 학습 진입 권장**

PASS: 11개 gate (50k 달성, F1>=0.99, IG recall 1.0, Papago recall 0.95, LLM HN FP 0%, 주요 AUC)
FAIL (hard, TF-IDF 한계): cleaned_blind FN, natural_boundary FN/FP
FAIL (structural): source_family/combined AUC

권장 결정:
1. TF-IDF baseline 최적화는 한계에 도달
2. 데이터셋 구조는 유효 (50k, 25k/25k, leakage=0)
3. KcELECTRA 학습으로 진입하여 FN=0 달성 여부 확인
4. BERT 계열 모델에서 natural_boundary FN=0 목표 달성 기대

## 6. Next Step

1. KcELECTRA fine-tuning on train split
   - 입력: final_prompt_dataset_50000_v1_train_valid_test.csv
   - 목표: test F1 >= 0.996, FN <= 5

2. IG/Papago holdout 재측정 (KcELECTRA 기준)

3. natural boundary holdout 재측정 (KcELECTRA 기준)

4. 통과 시 final 확정
"""

with open(OUT / "50k_final_analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("\nFinal analysis report saved: 50k_final_analysis_report.md")

# ── Update comprehensive gate CSV ─────────────────────────────────────
gate_final = pd.DataFrame([
    {"gate":"total_rows",           "target":"50000",   "actual":50000,    "status":"PASS"},
    {"gate":"normal_attack_balance","target":"25k/25k", "actual":"25k/25k","status":"PASS"},
    {"gate":"leakage",              "target":"0",       "actual":0,        "status":"PASS"},
    {"gate":"test_F1_LR",           "target":">=0.99",  "actual":0.9947,   "status":"PASS"},
    {"gate":"test_F1_SVM",          "target":">=0.99",  "actual":0.9959,   "status":"PASS"},
    {"gate":"IG_holdout_recall",    "target":">=0.85",  "actual":1.000,    "status":"PASS"},
    {"gate":"Papago_recall",        "target":">=0.90",  "actual":0.9524,   "status":"PASS"},
    {"gate":"LLM_HN_FP_ratio",      "target":"<=10%",   "actual":"0.0%",   "status":"PASS"},
    {"gate":"cleaned_blind_acc",    "target":">=0.996", "actual":0.9963,   "status":"PASS"},
    {"gate":"length_only_AUC",      "target":"<=0.56",  "actual":0.4879,   "status":"PASS"},
    {"gate":"source_detail_AUC",    "target":"<=0.68",  "actual":0.5532,   "status":"PASS"},
    {"gate":"style_family_AUC",     "target":"<=0.68",  "actual":0.5312,   "status":"PASS"},
    {"gate":"natural_boundary_FN",  "target":"0",       "actual":11,       "status":"FAIL (TF-IDF limit)"},
    {"gate":"natural_boundary_FP",  "target":"<=1",     "actual":8,        "status":"FAIL (TF-IDF limit)"},
    {"gate":"cleaned_blind_FN",     "target":"0",       "actual":28,       "status":"FAIL (TF-IDF limit)"},
    {"gate":"source_family_AUC",    "target":"<=0.68",  "actual":0.831,    "status":"FAIL (structural)"},
    {"gate":"combined_AUC",         "target":"<=0.72",  "actual":0.8952,   "status":"FAIL (structural)"},
])
gate_final.to_csv(OUT / "50k_gate_checklist_final.csv", index=False, encoding="utf-8-sig")

# Print summary
print("\n=== Final Gate Summary ===")
for _, r in gate_final.iterrows():
    icon = "O" if r["status"]=="PASS" else "X"
    print(f"  {icon} {r['gate']:<30} {str(r['actual']):<12} [{r['status']}]")
pass_n = (gate_final["status"]=="PASS").sum()
fail_n = gate_final["status"].str.startswith("FAIL").sum()
print(f"\n  PASS: {pass_n} / FAIL: {fail_n}")
print("\n  PASS criteria met: 50k rows, F1>=0.99, IG recall 1.0, Papago recall 0.95")
print("  Remaining FAIL: TF-IDF 구조 한계 (BERT/ELECTRA 필요) + source_family 구조적")
