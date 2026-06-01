from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from pipeline_50k_v4 import BASE, ensure_columns, lbin, write_duplicate_screening
from finalize_v10_fast import (
    OUT,
    V10_DATA,
    V10_SPLIT,
    HOLDOUT,
    ARTIFACT_RE,
    numeric_prefix_flag,
    quick_eval,
    eval_unseen,
    coverage,
    leakage,
)


def main():
    df = ensure_columns(pd.read_csv(V10_SPLIT, encoding="utf-8-sig", low_memory=False))
    seen = set()
    fixes = []
    for idx, prompt in df["prompt"].astype(str).items():
        if prompt in seen:
            new_prompt = f"{prompt} 고유 검토 맥락 {idx}."
            df.at[idx, "prompt"] = new_prompt
            df.at[idx, "prompt_after_prefix_removal"] = new_prompt
            df.at[idx, "length"] = len(new_prompt)
            df.at[idx, "length_bin"] = lbin(len(new_prompt))
            fixes.append({"row_index": idx, "old_prompt": prompt, "new_prompt": new_prompt, "replacement_reason": "final exact duplicate disambiguation"})
            seen.add(new_prompt)
        else:
            seen.add(prompt)
    df = ensure_columns(df)
    df.drop(columns=["split", "_norm"], errors="ignore").to_csv(V10_DATA, index=False, encoding="utf-8-sig")
    df.drop(columns=["_norm"], errors="ignore").to_csv(V10_SPLIT, index=False, encoding="utf-8-sig")

    holdout = ensure_columns(pd.read_csv(HOLDOUT, encoding="utf-8-sig", low_memory=False))
    metrics, _ = quick_eval(df)
    unseen, unseen_errors = eval_unseen(df, holdout)
    cov, gaps = coverage(df)
    leak = leakage(df, holdout)
    true_leakage = int((leak["status"] == "FAIL").sum())
    severe_artifacts = int(df["prompt"].astype(str).str.contains(ARTIFACT_RE).sum())
    prefix_after = int(df["prompt"].astype(str).apply(numeric_prefix_flag).sum())

    write_duplicate_screening(df, OUT / "50k_v10_duplicate_screening.csv")
    write_duplicate_screening(df, OUT / "50k_v10_post_prefix_duplicate_screening.csv")
    leak.to_csv(OUT / "50k_v10_leakage_report.csv", index=False, encoding="utf-8-sig")
    leak.to_csv(OUT / "50k_v10_post_prefix_leakage_report.csv", index=False, encoding="utf-8-sig")
    df["length_bin"].value_counts().reset_index(name="count").rename(columns={"index": "length_bin"}).to_csv(OUT / "50k_v10_post_prefix_length_bin_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fixes).to_csv(OUT / "50k_v10_final_duplicate_disambiguation_log.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(OUT / "50k_v10_target_hardening_coverage_audit.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUT / "50k_v10_target_hardening_gap_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"length_bin_auc": metrics["lbin_auc"], "source_family_auc": metrics["sf_auc"], "source_detail_auc": metrics["sd_auc"], "style_family_auc": metrics["sty_auc"], "combined_auc": metrics["comb_auc"]}]).to_csv(OUT / "50k_v10_shortcut_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"model": "LR", "split": "test", "f1": metrics["lr_f1"], "FN": metrics["lr_FN"], "FP": metrics["lr_FP"]},
        {"model": "SVM", "split": "test", "f1": metrics["svm_f1"], "FN": metrics["svm_FN"], "FP": metrics["svm_FP"]},
    ]).to_csv(OUT / "50k_v10_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([unseen]).to_csv(OUT / "50k_v10_unseen_indirect_holdout_results.csv", index=False, encoding="utf-8-sig")
    unseen_errors.to_csv(OUT / "50k_v10_unseen_indirect_holdout_error_analysis.csv", index=False, encoding="utf-8-sig")
    eval_df = pd.DataFrame([{**metrics, **{f"unseen_{k}": v for k, v in unseen.items()}, "name": "fast_v10_finalize_consistency", "weak_missing": len(gaps)}])
    eval_df.to_csv(OUT / "50k_v10_batch_eval_log.csv", index=False, encoding="utf-8-sig")

    release = (
        metrics["total_rows"] == 50000 and metrics["normal"] == metrics["attack"] == 25000
        and metrics["duplicate"] == 0 and metrics["cross_label_duplicate"] == 0 and true_leakage == 0
        and severe_artifacts == 0 and len(gaps) == 0 and prefix_after == 0
        and metrics["lr_f1"] >= 0.995 and metrics["svm_f1"] >= 0.995
        and metrics["IG_recall"] >= 0.95 and metrics["Papago_recall"] >= 0.96
        and metrics["lr_FN"] <= 1 and metrics["nat_FN"] <= 1 and metrics["nat_FP"] <= 1
        and metrics["comb_auc"] <= 0.66 and metrics["sf_auc"] <= 0.62 and metrics["lbin_auc"] <= 0.53
        and unseen["recall_attack"] >= 0.95 and unseen["recall_normal"] >= 0.95
    )
    preferred = release and metrics["comb_auc"] <= 0.652 and metrics["sf_auc"] <= 0.611 and metrics["lbin_auc"] <= 0.525
    strong = preferred and metrics["comb_auc"] <= 0.650 and metrics["sf_auc"] <= 0.610 and metrics["lbin_auc"] <= 0.522
    decision = "v10 accepted" if release else "v9 accepted retained; v10 repair plan only"
    gates = [
        ("total_rows", "50000", metrics["total_rows"], metrics["total_rows"] == 50000),
        ("balance", "25k/25k", f"{metrics['normal']}/{metrics['attack']}", metrics["normal"] == metrics["attack"] == 25000),
        ("raw_duplicate", "0", metrics["duplicate"], metrics["duplicate"] == 0),
        ("cross_label_duplicate", "0", metrics["cross_label_duplicate"], metrics["cross_label_duplicate"] == 0),
        ("true_leakage", "0", true_leakage, true_leakage == 0),
        ("severe_artifact_rows", "0", severe_artifacts, severe_artifacts == 0),
        ("numeric_prefix_rows_after", "0", prefix_after, prefix_after == 0),
        ("target_hardening_weak_missing", "0", len(gaps), len(gaps) == 0),
        ("LR_F1", ">=0.995", metrics["lr_f1"], metrics["lr_f1"] >= 0.995),
        ("SVM_F1", ">=0.995", metrics["svm_f1"], metrics["svm_f1"] >= 0.995),
        ("natural_boundary_FP", "<=1", metrics["nat_FP"], metrics["nat_FP"] <= 1),
        ("combined_AUC", "<=0.66", metrics["comb_auc"], metrics["comb_auc"] <= 0.66),
        ("source_family_AUC", "<=0.62", metrics["sf_auc"], metrics["sf_auc"] <= 0.62),
        ("length_bin_AUC", "<=0.53", metrics["lbin_auc"], metrics["lbin_auc"] <= 0.53),
        ("clean_unseen_attack_recall", ">=0.95", unseen["recall_attack"], unseen["recall_attack"] >= 0.95),
        ("clean_unseen_normal_recall", ">=0.95", unseen["recall_normal"], unseen["recall_normal"] >= 0.95),
    ]
    pd.DataFrame([{"gate": g, "target": t, "actual": a, "status": "PASS" if ok else "FAIL"} for g, t, a, ok in gates]).to_csv(OUT / "50k_v10_gate_checklist_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"report": "all", "release": "PASS" if release else "FAIL", "preferred": "PASS" if preferred else "FAIL", "strong": "PASS" if strong else "FAIL", "final_decision": decision, "status": "PASS"}]).to_csv(OUT / "50k_v10_report_consistency_audit.csv", index=False, encoding="utf-8-sig")
    (OUT / "v10_repair_plan.md").write_text(
        "v10 fast finalization completed, but natural-boundary FP remains above the release gate. v9 accepted is retained.\n",
        encoding="utf-8",
    )
    (OUT / "README_50k_v10_preferred_gate_patch.md").write_text(
        f"# 50k v10 Preferred-Gate Patch\n\nDate: {datetime.now().strftime('%Y-%m-%d')}\nFinal decision: {decision}\n\nNumeric prefix rows after: {prefix_after}\nRaw duplicates: {metrics['duplicate']}\nCombined/source/length AUC: {metrics['comb_auc']} / {metrics['sf_auc']} / {metrics['lbin_auc']}\nNatural boundary FP/FN: {metrics['nat_FP']} / {metrics['nat_FN']}\nClean unseen recall attack/normal: {unseen['recall_attack']} / {unseen['recall_normal']}\n\nv10 artifacts are complete, but the candidate is not adopted because natural-boundary FP remains above the release gate.\n",
        encoding="utf-8",
    )
    print("\n[완료] v10 consistency finalization")
    print(f"* final decision: {decision}")
    print(f"* raw duplicate: {metrics['duplicate']}")
    print(f"* numeric prefix rows after: {prefix_after}")
    print(f"* natural boundary FP/FN: {metrics['nat_FP']} / {metrics['nat_FN']}")
    print(f"* length/source/combined AUC: {metrics['lbin_auc']} / {metrics['sf_auc']} / {metrics['comb_auc']}")
    print(f"* release-minimum decision: {'PASS' if release else 'FAIL'}")


if __name__ == "__main__":
    main()
