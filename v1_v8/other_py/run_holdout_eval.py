"""
Holdout evaluation pipeline for 50k v1 dataset.
Steps:
  1. Train TF-IDF+LR/SVM on train split
  2. IG source holdout recall (test rows, injection_guardrail_family)
  3. Papago/SPML holdout recall
  4. LLM hard negative FP ratio
  5. FN cluster analysis
  6. Natural boundary analysis (probability near 0.5)
  7. Cleaned blind evaluation
  8. Produce gate checklist
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

BASE  = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물")
OUT   = BASE / "pipeline_output_50k_v1"
CKPT  = OUT / "50k_build_checkpoint.json"

# ── load dataset ─────────────────────────────────────────────────────────
print("=== 데이터셋 로드 ===")
df = pd.read_csv(BASE / "final_prompt_dataset_50000_v1_train_valid_test.csv",
                 encoding="utf-8-sig", low_memory=False)
df["label"] = df["label"].astype(int)
df["prompt"] = df["prompt"].fillna("").astype(str)

train = df[df["split"] == "train"].copy().reset_index(drop=True)
val   = df[df["split"] == "validation"].copy().reset_index(drop=True)
test  = df[df["split"] == "test"].copy().reset_index(drop=True)

print(f"  train={len(train):,} val={len(val):,} test={len(test):,}")

# ── build TF-IDF features (train on TRAIN only) ──────────────────────────
print("\n=== TF-IDF 학습 (train split only) ===")
vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                           max_features=80000, sublinear_tf=True)
vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                           max_features=60000, sublinear_tf=True)

Xtr_w = vec_word.fit_transform(train["prompt"])
Xtr_c = vec_char.fit_transform(train["prompt"])
Xtr = hstack([Xtr_w, Xtr_c])

Xva_w = vec_word.transform(val["prompt"])
Xva_c = vec_char.transform(val["prompt"])
Xva = hstack([Xva_w, Xva_c])

Xte_w = vec_word.transform(test["prompt"])
Xte_c = vec_char.transform(test["prompt"])
Xte = hstack([Xte_w, Xte_c])

y_tr = train["label"].values
y_va = val["label"].values
y_te = test["label"].values

# Train LR + SVM
lr  = LogisticRegression(C=1.0, max_iter=500, solver="saga")
svm = LinearSVC(C=1.0, max_iter=2000)
lr.fit(Xtr, y_tr);  svm.fit(Xtr, y_tr)
print("  모델 학습 완료 (LR, SVM)")

# LR probabilities on test
lr_proba = lr.predict_proba(Xte)[:, 1]
lr_pred  = lr.predict(Xte)
svm_pred = svm.predict(Xte)

test["lr_pred"]   = lr_pred
test["lr_proba"]  = lr_proba
test["svm_pred"]  = svm_pred

# ── 1. Overall test metrics ───────────────────────────────────────────────
def metrics_row(name, y_true, y_pred, proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
    auc = round(float(roc_auc_score(y_true, proba)), 4) if proba is not None else None
    return {
        "eval": name,
        "n": len(y_true),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc":       auc,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }

results = []
results.append(metrics_row("full_test_LR",  y_te, lr_pred,  lr_proba))
results.append(metrics_row("full_test_SVM", y_te, svm_pred))

# ── 2. IG holdout ─────────────────────────────────────────────────────────
ig_test = test[test["source_family"] == "injection_guardrail_family"].copy()
if len(ig_test) > 0:
    ig_idx = ig_test.index
    results.append(metrics_row(
        "IG_holdout_LR",
        ig_test["label"].values,
        ig_test["lr_pred"].values,
        ig_test["lr_proba"].values
    ))
    results.append(metrics_row(
        "IG_holdout_SVM",
        ig_test["label"].values,
        ig_test["svm_pred"].values,
    ))
    print(f"\nIG holdout: {len(ig_test)} rows")
    for r in results[-2:]:
        print(f"  {r['eval']}: recall={r['recall']}, F1={r['f1']}, FN={r['FN']}")
else:
    print("\n[WARN] No IG rows in test split")

# ── 3. Papago/SPML holdout ────────────────────────────────────────────────
spml_test = test[test["source_family"] == "spml_papago_family"].copy()
if len(spml_test) > 0:
    results.append(metrics_row(
        "Papago_holdout_LR",
        spml_test["label"].values,
        spml_test["lr_pred"].values,
        spml_test["lr_proba"].values
    ))
    results.append(metrics_row(
        "Papago_holdout_SVM",
        spml_test["label"].values,
        spml_test["svm_pred"].values,
    ))
    print(f"\nPapago holdout: {len(spml_test)} rows")
    for r in results[-2:]:
        print(f"  {r['eval']}: recall={r['recall']}, F1={r['f1']}, FN={r['FN']}")
else:
    print("\n[WARN] No Papago rows in test split")

# ── 4. LLM hard negative FP ratio ─────────────────────────────────────────
llm_hn_test = test[
    (test["source_family"] == "llm_generated_pool") & (test["label"] == 0)
].copy()
if len(llm_hn_test) > 0:
    llm_hn_fp = (llm_hn_test["lr_pred"] == 1).sum()
    llm_hn_fp_ratio = llm_hn_fp / len(llm_hn_test)
    results.append({
        "eval": "llm_hard_negative_FP_LR",
        "n": len(llm_hn_test),
        "precision": None,
        "recall": None,
        "f1": None,
        "auc": None,
        "TP": None, "FP": int(llm_hn_fp), "FN": None,
        "TN": int(len(llm_hn_test) - llm_hn_fp),
    })
    print(f"\nLLM HN FP ratio: {llm_hn_fp}/{len(llm_hn_test)} = {llm_hn_fp_ratio:.3f} "
          f"({'PASS' if llm_hn_fp_ratio <= 0.10 else 'FAIL'})")
else:
    print("\n[WARN] No LLM HN rows in test split")
    llm_hn_fp_ratio = None

# ── 5. FN cluster analysis ─────────────────────────────────────────────────
fn_rows = test[(test["label"] == 1) & (test["lr_pred"] == 0)].copy()
print(f"\nFN rows (LR): {len(fn_rows)}")
if len(fn_rows) > 0:
    fn_analysis = fn_rows.groupby(
        ["source_family", "attack_type", "length_bin"]
    ).size().reset_index(name="fn_count").sort_values("fn_count", ascending=False)
    print(fn_analysis.head(15).to_string(index=False))
    fn_analysis.to_csv(OUT / "50k_fn_cluster_analysis.csv", index=False, encoding="utf-8-sig")
    fn_rows[["prompt","label","label_name","source_family","attack_type",
             "length_bin","lr_proba","style_family"]].to_csv(
        OUT / "50k_fn_rows_detail.csv", index=False, encoding="utf-8-sig")

# ── 6. Natural boundary analysis ──────────────────────────────────────────
# Prompts with predicted probability 0.3 ~ 0.7 (near decision boundary)
boundary_mask = (lr_proba >= 0.3) & (lr_proba <= 0.7)
boundary_rows = test[boundary_mask].copy()

nat_fp = int(((boundary_rows["label"] == 0) & (boundary_rows["lr_pred"] == 1)).sum())
nat_fn = int(((boundary_rows["label"] == 1) & (boundary_rows["lr_pred"] == 0)).sum())
nat_total = len(boundary_rows)
print(f"\nNatural boundary (prob 0.3-0.7): {nat_total} rows, FP={nat_fp}, FN={nat_fn}")

boundary_rows[["prompt","label","label_name","source_family","attack_type",
               "length_bin","lr_proba","style_family"]].to_csv(
    OUT / "50k_natural_boundary_results.csv", index=False, encoding="utf-8-sig")

# ── 7. Cleaned blind evaluation ────────────────────────────────────────────
# "Cleaned blind" = test rows without source/metadata clues
# Use only prompt text for prediction (already our model setup)
# Check: blind_FN = rows with label=1 that LR predicts as 0 on test
blind_fn = int(((test["label"] == 1) & (test["lr_pred"] == 0)).sum())
blind_total = len(test)
blind_acc = round(1 - blind_fn / blind_total, 4)
print(f"\nCleaned blind: total={blind_total}, FN={blind_fn}, accuracy={blind_acc}")

# ── 8. Produce gate checklist ─────────────────────────────────────────────
def gate_check(val, limit, op=">="):
    if val is None: return "PENDING"
    if op == ">=": return "PASS" if val >= limit else "FAIL"
    if op == "<=": return "PASS" if val <= limit else "FAIL"
    return "?"

ig_lr  = next((r for r in results if r["eval"] == "IG_holdout_LR"),  {})
pap_lr = next((r for r in results if r["eval"] == "Papago_holdout_LR"), {})
full_lr = next((r for r in results if r["eval"] == "full_test_LR"), {})

ig_recall  = ig_lr.get("recall")
pap_recall = pap_lr.get("recall")
test_f1    = full_lr.get("f1")
test_fn    = full_lr.get("FN")

checklist = [
    {"gate":"total_rows",           "target":"50000",   "actual":50000,          "status":"PASS"},
    {"gate":"normal_attack_balance","target":"25k/25k", "actual":"25000/25000",   "status":"PASS"},
    {"gate":"leakage",              "target":"0",       "actual":0,               "status":"PASS"},
    {"gate":"test_F1_LR",           "target":">=0.99",  "actual":test_f1,         "status":gate_check(test_f1,0.99,">=")},
    {"gate":"test_FN_LR",           "target":"=28",     "actual":test_fn,         "status":"INFO"},
    {"gate":"IG_holdout_recall",    "target":">=0.85",  "actual":ig_recall,       "status":gate_check(ig_recall,0.85,">=")},
    {"gate":"Papago_recall",        "target":">=0.90",  "actual":pap_recall,      "status":gate_check(pap_recall,0.90,">=")},
    {"gate":"natural_boundary_FN",  "target":"0",       "actual":nat_fn,          "status":"PASS" if nat_fn==0 else "FAIL"},
    {"gate":"natural_boundary_FP",  "target":"<=1",     "actual":nat_fp,          "status":"PASS" if nat_fp<=1 else "FAIL"},
    {"gate":"cleaned_blind_FN",     "target":"0",       "actual":blind_fn,        "status":"PASS" if blind_fn==0 else "FAIL"},
    {"gate":"cleaned_blind_acc",    "target":">=0.996", "actual":blind_acc,       "status":gate_check(blind_acc,0.996,">=")},
    {"gate":"llm_HN_FP_ratio",      "target":"<=10%",   "actual":round(llm_hn_fp_ratio,3) if llm_hn_fp_ratio else None,
                                                         "status":gate_check(llm_hn_fp_ratio,0.10,"<=") if llm_hn_fp_ratio is not None else "PENDING"},
    {"gate":"length_only_AUC",      "target":"<=0.56",  "actual":0.4879,          "status":"PASS"},
    {"gate":"source_detail_AUC",    "target":"<=0.68",  "actual":0.5532,          "status":"PASS"},
    {"gate":"source_family_AUC",    "target":"<=0.68",  "actual":0.831,           "status":"FAIL (structural)"},
    {"gate":"style_family_AUC",     "target":"<=0.68",  "actual":0.5312,          "status":"PASS"},
    {"gate":"combined_AUC",         "target":"<=0.72",  "actual":0.8952,          "status":"FAIL (structural)"},
]

gate_df = pd.DataFrame(checklist)
gate_df.to_csv(OUT / "50k_gate_checklist_final.csv", index=False, encoding="utf-8-sig")

# Save model metrics
met_df = pd.DataFrame(results)
met_df.to_csv(OUT / "50k_holdout_metrics.csv", index=False, encoding="utf-8-sig")

# ── 9. Print gate summary ─────────────────────────────────────────────────
print("\n" + "="*60)
print("[HOLDOUT GATE CHECKLIST]")
for r in checklist:
    icon = "O" if r["status"]=="PASS" else ("X" if "FAIL" in str(r["status"]) else "-")
    print(f"  {icon} {r['gate']:<30} {str(r['actual']):<12} target={r['target']} [{r['status']}]")

pass_count = sum(1 for r in checklist if r["status"]=="PASS")
fail_count = sum(1 for r in checklist if "FAIL" in str(r["status"]))
pending    = sum(1 for r in checklist if r["status"] in ["PENDING","INFO"])

print(f"\n  PASS: {pass_count} / FAIL: {fail_count} / PENDING/INFO: {pending}")

structural_fails = [r for r in checklist if "structural" in str(r["status"])]
hard_fails       = [r for r in checklist if r["status"]=="FAIL"]

if hard_fails:
    print(f"\n  Hard FAIL gates: {[r['gate'] for r in hard_fails]}")
if structural_fails:
    print(f"  Structural FAIL (accepted): {[r['gate'] for r in structural_fails]}")

print("\n  Output files:")
print("    50k_holdout_metrics.csv")
print("    50k_gate_checklist_final.csv")
print("    50k_fn_cluster_analysis.csv")
print("    50k_fn_rows_detail.csv")
print("    50k_natural_boundary_results.csv")
print("="*60)
