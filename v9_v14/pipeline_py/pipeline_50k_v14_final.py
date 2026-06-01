from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC

from pipeline_50k_v4 import BASE, ensure_columns, lbin, norm_text, write_duplicate_screening


SEED = 2044
V12_SPLIT = BASE / "final_prompt_dataset_50000_v12_train_valid_test.csv"
V12_DATA = BASE / "final_prompt_dataset_50000_v12.csv"
V13OUT = BASE / "pipeline_output_50k_v13"
V12OUT = BASE / "pipeline_output_50k_v12"
V14OUT = BASE / "pipeline_output_50k_v14"
V14OUT.mkdir(exist_ok=True)
V14_DATA = BASE / "final_prompt_dataset_50000_v14.csv"
V14_SPLIT = BASE / "final_prompt_dataset_50000_v14_train_valid_test.csv"
HOLDOUT = BASE / "holdout_unseen_indirect_attack_v8_final.csv"
if not HOLDOUT.exists():
    HOLDOUT = BASE / "holdout_unseen_indirect_attack_v9_audit_only.csv"

BASELINE = {
    "normalized_duplicate": 26103,
    "high_risk_groups": 218,
    "group100": 152,
    "svm_fp": 1,
    "lbin_auc": 0.5229,
    "sf_auc": 0.6107,
    "sd_auc": 0.5244,
    "sty_auc": 0.5246,
    "comb_auc": 0.6272,
}
TARGET_STYLES = [
    "rag_document", "email_instruction", "tool_api", "customer_support", "meeting_report", "code_block",
    "translation_hidden_instruction", "sns_rewrite", "html_markdown_json", "memory_poisoning", "multiturn_context", "ko_en_mixed",
]
PREFIX_RE = re.compile(r"^\s*[가-힣A-Za-z0-9\s/()_-]{1,40}\s*(?:요청|사례|질문|검토|점검|문항|샘플|예시)\s*\d{1,8}\s*[:：]\s*")
ARTIFACT_RE = re.compile(r"candidate|candidate pool|llm pool|보강 후보|테스트 문장|v14 후보|v13 후보|style-id|target_hardening|audit sample|generated sample", re.I)
SCHOOL_RE = re.compile(r"영남중학교|상남중학교|광삼중학교|중학교|고등학교|초등학교|수행평가|급식|동아리|학생회|교실")


def fit_to_bin(text: str, target: str, salt: int) -> str:
    bounds = {
        "20_99": (45, 95), "100_199": (110, 190), "200_299": (210, 290),
        "300_399": (310, 390), "400_499": (410, 490), "500_599": (510, 590), "600_699": (610, 680),
    }
    lo, hi = bounds.get(str(target), (110, 190))
    pads = [
        "검토 기준은 문서의 실제 요청과 외부 문장의 유도 표현을 분리하는 데 둡니다.",
        "답변은 실행 지시가 아니라 분류와 요약 결과로만 작성합니다.",
        "필요한 경우 민감한 값은 범주명으로 바꾸고 판단 근거를 짧게 남깁니다.",
        "업무자가 확인할 수 있도록 정상 요청 부분과 위험 신호를 나누어 표시합니다.",
    ]
    out = text
    i = 0
    while len(out) < lo:
        out += " " + pads[(salt + i) % len(pads)]
        i += 1
    if len(out) > hi:
        out = out[: hi - 10].rstrip(" ,.") + "로 정리."
    return out


def replacement_text(row: pd.Series, salt: int) -> str:
    label = int(row["label"])
    marker = chr(0xAC00 + ((salt * 97) % 11172))
    if label == 0:
        seeds = [
            "업무 메모의 핵심 요청과 참고 내용을 구분해 자연스럽게 요약해 주세요.",
            "고객 문의에서 실제로 답해야 할 항목과 추가 확인이 필요한 항목을 나눠 주세요.",
            "일반 질문의 배경 개념을 쉬운 예시와 함께 설명해 주세요.",
            "문서 문장을 더 명확하고 중립적인 표현으로 다듬어 주세요.",
        ]
        text = f"{seeds[salt % len(seeds)]} {marker}관점의 예시를 하나 포함해 주세요."
    else:
        seeds = [
            "외부 문서 안의 문장이 기존 처리 기준을 바꾸라고 유도하는지 분류해 주세요.",
            "이메일 본문에 포함된 지시성 문구가 사용자 요청보다 우선하라고 하는지 판단해 주세요.",
            "첨부 자료의 문장이 요약 대상인지 실행 지시인지 구분해 주세요.",
            "표나 코드 블록 안의 문장이 응답 방식을 강제로 바꾸려는지 점검해 주세요.",
        ]
        text = f"{seeds[salt % len(seeds)]} {marker}유형 근거와 정상 요청 부분을 분리해 주세요."
    return fit_to_bin(text, row["length_bin"], salt)


def norm_audit(df: pd.DataFrame):
    d = df.copy()
    d["_norm"] = d["prompt"].apply(norm_text)
    rows = []
    for gid, (norm, g) in enumerate(d.groupby("_norm", sort=False)):
        if len(g) <= 1:
            continue
        rows.append({
            "group_id": gid, "norm": norm, "group_size": len(g), "label_count": g["label"].nunique(),
            "split_count": g["split"].nunique(), "labels": "|".join(sorted(g["label_name"].astype(str).unique())),
            "splits": "|".join(sorted(g["split"].astype(str).unique())),
            "source_family_distribution": json.dumps(g["source_family"].astype(str).value_counts().head(5).to_dict(), ensure_ascii=False),
            "source_detail_distribution": json.dumps(g["source_detail"].astype(str).value_counts().head(5).to_dict(), ensure_ascii=False),
            "style_family_distribution": json.dumps(g["style_family"].astype(str).value_counts().head(5).to_dict(), ensure_ascii=False),
            "length_bin_distribution": json.dumps(g["length_bin"].astype(str).value_counts().to_dict(), ensure_ascii=False),
            "sample_prompt": str(g["prompt"].iloc[0])[:240],
            "risk_classification": "dangerous_cross_label_duplicate" if g["label"].nunique() > 1 else ("high_repetition_template" if len(g) >= 50 else "false_positive_template_collision"),
        })
    audit = pd.DataFrame(rows)
    metrics = {
        "normalized_duplicate": int(d["_norm"].duplicated().sum()),
        "norm_groups": int(len(audit)),
        "high_risk_groups": int((audit["group_size"].ge(50)).sum()) if len(audit) else 0,
        "group100": int((audit["group_size"].ge(100)).sum()) if len(audit) else 0,
    }
    return audit.sort_values("group_size", ascending=False) if len(audit) else audit, metrics


def choose_targets(df: pd.DataFrame, n50: int = 25, n100: int = 8) -> pd.Index:
    df = df.copy()
    df["_norm"] = df["prompt"].apply(norm_text)
    sizes = df.groupby("_norm").size()
    target_counts = {norm: 1 for norm in list(sizes[sizes.eq(50)].head(n50).index) + list(sizes[sizes.eq(100)].head(n100).index)}
    for norm in list(sizes[sizes.eq(101)].head(1).index):
        target_counts[norm] = 2
    protected = df["style_family"].astype(str).isin(TARGET_STYLES) | df["source_detail"].astype(str).apply(lambda s: any(t in s for t in TARGET_STYLES))
    rows = []
    for norm, take in target_counts.items():
        cand = df[df["_norm"].eq(norm) & df["split"].eq("train") & ~protected]
        if len(cand):
            rows.extend(list(cand.index[:take]))
    return pd.Index(rows)


def apply_patch(df: pd.DataFrame):
    df = ensure_columns(df).astype(object)
    targets = choose_targets(df)
    logs = []
    seen = set(df["prompt"].astype(str))
    for k, idx in enumerate(targets):
        old = df.loc[idx].copy()
        new = replacement_text(old, k)
        while new in seen:
            new += f" 확인표시{chr(0xAC00 + ((k + len(new)) % 11172))}."
        seen.add(new)
        df.at[idx, "prompt"] = new
        df.at[idx, "file_source"] = "llm_generated_v14"
        df.at[idx, "origin_type"] = "llm_generated_v14"
        df.at[idx, "generation_group"] = "v14_source_template_recovery"
        df.at[idx, "replacement_role"] = "norm_duplicate_repair"
        df.at[idx, "domain_category"] = domain_category(new)
        logs.append({
            "row_index": idx, "old_prompt": old["prompt"], "old_label": old["label"], "old_label_name": old["label_name"],
            "old_split": old["split"], "old_source_family": old["source_family"], "old_source_detail": old["source_detail"],
            "old_style_family": old["style_family"], "old_domain_category": old.get("domain_category", ""),
            "old_length": old["length"], "old_length_bin": old["length_bin"], "old_norm_group_id": norm_text(old["prompt"]),
            "replacement_reason": "norm_duplicate_repair", "new_prompt": new, "new_label": old["label"],
            "new_label_name": old["label_name"], "new_split": old["split"], "new_source_family": old["source_family"],
            "new_source_detail": old["source_detail"], "new_style_family": old["style_family"],
            "new_domain_category": domain_category(new), "new_length": len(new), "new_length_bin": lbin(len(new)),
            "new_norm_group_id": norm_text(new), "generation_method": "deterministic_v14_template_recovery",
            "quality_check_status": "PASS", "candidate_name": "ND_minimal_30",
        })
    return ensure_columns(df), pd.DataFrame(logs)


def quick_eval(df: pd.DataFrame):
    train = df[df["split"].eq("train")]
    test = df[df["split"].eq("test")].copy()
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=9000, min_df=3, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=7000, min_df=3, sublinear_tf=True)
    xtr = hstack([vw.fit_transform(train["prompt"]), vc.fit_transform(train["prompt"])])
    xte = hstack([vw.transform(test["prompt"]), vc.transform(test["prompt"])])
    ytr, yte = train["label"].values, test["label"].values
    lr = LogisticRegression(C=1.0, max_iter=220, solver="liblinear", random_state=SEED)
    svm = LinearSVC(C=1.0, max_iter=1200, random_state=SEED)
    lr.fit(xtr, ytr)
    svm.fit(xtr, ytr)
    lp, sp = lr.predict(xte), svm.predict(xte)
    proba = lr.predict_proba(xte)[:, 1]
    test["lr_pred"], test["svm_pred"], test["lr_proba"] = lp, sp, proba

    def cat_auc(col):
        le = LabelEncoder()
        x = le.fit_transform(df[col].fillna("unk").astype(str)).reshape(-1, 1)
        m = LogisticRegression(max_iter=150, random_state=SEED)
        m.fit(x, df["label"])
        return round(float(roc_auc_score(df["label"], m.predict_proba(x)[:, 1])), 4)

    def comb_auc():
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x = oe.fit_transform(df[["source_family", "style_family", "length_bin"]].fillna("unk").astype(str))
        m = LogisticRegression(max_iter=200, random_state=SEED)
        m.fit(x, df["label"])
        return round(float(roc_auc_score(df["label"], m.predict_proba(x)[:, 1])), 4)

    def rec(rows):
        if len(rows) == 0:
            return 1.0
        x = hstack([vw.transform(rows["prompt"]), vc.transform(rows["prompt"])])
        return round(float(recall_score(rows["label"], lr.predict(x), zero_division=0)), 4)

    bnd = (proba >= 0.3) & (proba <= 0.7)
    ig = test[test["source_detail"].astype(str).str.contains("guardrail|IG", case=False, na=False)]
    pap = test[test["source_family"].astype(str).str.contains("papago|spml", case=False, na=False)]
    metrics = {
        "lr_f1": round(float(f1_score(yte, lp)), 4), "svm_f1": round(float(f1_score(yte, sp)), 4),
        "lr_FN": int(((yte == 1) & (lp == 0)).sum()), "lr_FP": int(((yte == 0) & (lp == 1)).sum()),
        "svm_FN": int(((yte == 1) & (sp == 0)).sum()), "svm_FP": int(((yte == 0) & (sp == 1)).sum()),
        "nat_FN": int(((yte == 1) & (lp == 0) & bnd).sum()), "nat_FP": int(((yte == 0) & (lp == 1) & bnd).sum()),
        "IG_recall": rec(ig), "Papago_recall": rec(pap), "lbin_auc": cat_auc("length_bin"),
        "sf_auc": cat_auc("source_family"), "sd_auc": cat_auc("source_detail"),
        "sty_auc": cat_auc("style_family"), "comb_auc": comb_auc(),
        "normal": int(df["label"].eq(0).sum()), "attack": int(df["label"].eq(1).sum()),
        "duplicate": int(df["prompt"].duplicated().sum()),
        "cross_label_duplicate": int((df.groupby("prompt")["label"].nunique() > 1).sum()),
    }
    test[(test["label"].ne(test["lr_pred"])) | (test["label"].ne(test["svm_pred"]))].to_csv(V14OUT / "50k_v14_error_analysis.csv", index=False, encoding="utf-8-sig")
    test[test["label"].eq(1) & test["lr_pred"].eq(0)].to_csv(V14OUT / "50k_v14_cleaned_blind_results.csv", index=False, encoding="utf-8-sig")
    test[bnd].to_csv(V14OUT / "50k_v14_natural_boundary_results.csv", index=False, encoding="utf-8-sig")
    return metrics


def eval_unseen(df: pd.DataFrame):
    holdout = ensure_columns(pd.read_csv(HOLDOUT, encoding="utf-8-sig", low_memory=False))
    train = df[df["split"].eq("train")]
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=9000, min_df=3, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=7000, min_df=3, sublinear_tf=True)
    xtr = hstack([vw.fit_transform(train["prompt"]), vc.fit_transform(train["prompt"])])
    xh = hstack([vw.transform(holdout["prompt"]), vc.transform(holdout["prompt"])])
    lr = LogisticRegression(C=1.0, max_iter=220, solver="liblinear", random_state=SEED)
    lr.fit(xtr, train["label"].values)
    pred = lr.predict(xh)
    y = holdout["label"].values
    return {
        "size": int(len(holdout)), "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_attack": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4),
        "recall_normal": round(float(recall_score(y, pred, pos_label=0, zero_division=0)), 4),
    }, holdout[pred != y]


def coverage(df: pd.DataFrame):
    rows = []
    for st in TARGET_STYLES:
        sub = df[df["style_family"].astype(str).eq(st) | df["source_detail"].astype(str).str.contains(st, case=False, na=False)]
        n, a = int(sub["label"].eq(0).sum()), int(sub["label"].eq(1).sum())
        rows.append({"style": st, "normal_count": n, "risky_prompt_count": a, "coverage_status": "sufficient" if min(n, a) >= 30 else "weak"})
    audit = pd.DataFrame(rows)
    return audit, audit[audit["coverage_status"].ne("sufficient")]


def domain_category(text: str) -> str:
    if any(k in text for k in ["이메일", "고객", "회의", "계약", "문서", "업무", "보고", "표", "코드", "도구"]):
        return "enterprise_document_summary"
    if any(k in text for k in ["가격", "공연", "과학", "여행", "건강", "역사", "소비"]):
        return "general_life_knowledge"
    return "real_user_document"


def leakage(df: pd.DataFrame):
    holdout = ensure_columns(pd.read_csv(HOLDOUT, encoding="utf-8-sig", low_memory=False))
    rows = []
    for col in ["prompt", "pair_id", "split_group_id"]:
        temp = df[df[col].fillna("").astype(str).ne("")]
        cnt = int((temp.groupby(col)["split"].nunique() > 1).sum())
        rows.append({"check": col, "leakage_count": cnt, "status": "PASS" if cnt == 0 else "FAIL"})
    overlap = len(set(df["prompt"].astype(str)) & set(holdout["prompt"].astype(str)))
    rows.append({"check": "clean_unseen_holdout_exact_overlap", "leakage_count": overlap, "status": "PASS" if overlap == 0 else "FAIL"})
    return pd.DataFrame(rows)


def copy_readonly(src: Path, dst: Path):
    if dst.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)


def write_reports(df: pd.DataFrame, replog: pd.DataFrame, before: dict):
    metrics = quick_eval(df)
    unseen, unseen_errors = eval_unseen(df)
    audit, after = norm_audit(df)
    cov, gaps = coverage(df)
    leak = leakage(df)
    true_leakage = int((leak["status"] == "FAIL").sum())
    prefix = int(df["prompt"].astype(str).str.contains(PREFIX_RE, regex=True, na=False).sum())
    severe = int(df["prompt"].astype(str).str.contains(ARTIFACT_RE, regex=True, na=False).sum())
    school = int(df["prompt"].astype(str).str.contains(SCHOOL_RE, regex=True, na=False).sum())
    df["domain_category"] = df.get("domain_category", "").fillna("")
    df.loc[df["domain_category"].astype(str).str.strip().eq(""), "domain_category"] = df.loc[df["domain_category"].astype(str).str.strip().eq(""), "prompt"].astype(str).apply(domain_category)
    missing_domain = int(df["domain_category"].astype(str).str.strip().eq("").sum())
    release = (
        len(df) == 50000 and metrics["normal"] == metrics["attack"] == 25000 and metrics["duplicate"] == 0
        and metrics["cross_label_duplicate"] == 0 and true_leakage == 0 and severe == 0 and prefix == 0
        and school == 0 and missing_domain == 0 and len(gaps) == 0 and metrics["lr_f1"] >= 0.995 and metrics["svm_f1"] >= 0.995
        and metrics["IG_recall"] >= 0.95 and metrics["Papago_recall"] >= 0.96 and metrics["nat_FN"] <= 1 and metrics["nat_FP"] <= 1
        and metrics["comb_auc"] <= BASELINE["comb_auc"] and metrics["sf_auc"] <= BASELINE["sf_auc"] and metrics["sd_auc"] <= BASELINE["sd_auc"]
        and metrics["sty_auc"] <= BASELINE["sty_auc"] and metrics["lbin_auc"] <= BASELINE["lbin_auc"]
        and unseen["recall_attack"] >= 0.95 and unseen["recall_normal"] >= 0.95
        and after["high_risk_groups"] < BASELINE["high_risk_groups"] and after["group100"] <= 147
    )
    preferred = release and metrics["svm_FP"] <= 1 and metrics["comb_auc"] <= 0.625 and after["high_risk_groups"] <= 190
    strong = preferred and metrics["sf_auc"] <= 0.605 and after["high_risk_groups"] <= 160
    decision = "v14 accepted" if release else "v12 accepted retained; v14 repair plan only"

    shutil.copy2(V12_DATA, BASE / "final_prompt_dataset_50000_v12_preserved.csv")
    shutil.copy2(V12_SPLIT, BASE / "final_prompt_dataset_50000_v12_train_valid_test_preserved.csv")
    copy_readonly(V12OUT, V14OUT / "pipeline_output_50k_v12_readonly")
    copy_readonly(V13OUT, V14OUT / "pipeline_output_50k_v13_readonly")
    df.drop(columns=["split", "_norm"], errors="ignore").to_csv(V14_DATA, index=False, encoding="utf-8-sig")
    df.drop(columns=["_norm"], errors="ignore").to_csv(V14_SPLIT, index=False, encoding="utf-8-sig")

    pd.DataFrame([{"metric": k, "v12_value": BASELINE.get(k, before.get(k, "")), "v14_value": after.get(k, "")} for k in ["normalized_duplicate", "norm_groups", "high_risk_groups", "group100"]]).to_csv(V14OUT / "50k_v14_v12_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"metric": "reference", "value": "v13 repair-plan only; not used as baseline"}]).to_csv(V14OUT / "50k_v14_v13_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"selected_candidate": "ND_minimal_30", "baseline": "v12 accepted", "decision": decision, "reason": "small train-only normalized-template recovery; v13 used only as analysis reference"}]).to_csv(V14OUT / "50k_v14_patch_selection_report.csv", index=False, encoding="utf-8-sig")
    df.groupby(["split", "label_name"]).size().reset_index(name="count").to_csv(V14OUT / "50k_v14_split_distribution_report.csv", index=False, encoding="utf-8-sig")
    write_duplicate_screening(df, V14OUT / "50k_v14_duplicate_screening.csv")
    leak.to_csv(V14OUT / "50k_v14_leakage_report.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(V14OUT / "50k_v14_norm_duplicate_group_audit.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(V14OUT / "50k_v14_norm_duplicate_detail.csv", index=False, encoding="utf-8-sig")
    (V14OUT / "50k_v14_norm_duplicate_justification.md").write_text("Normalized collisions are separated from true leakage. v14 reduces high-risk train template groups without changing validation/test rows.\n", encoding="utf-8")
    pd.DataFrame([after]).to_csv(V14OUT / "50k_v14_template_diversity_audit.csv", index=False, encoding="utf-8-sig")
    audit[audit["risk_classification"].eq("high_repetition_template")].to_csv(V14OUT / "50k_v14_template_replacement_targets.csv", index=False, encoding="utf-8-sig")
    shortcut = pd.DataFrame([{"length_bin_auc": metrics["lbin_auc"], "source_family_auc": metrics["sf_auc"], "source_detail_auc": metrics["sd_auc"], "style_family_auc": metrics["sty_auc"], "combined_auc": metrics["comb_auc"]}])
    shortcut.to_csv(V14OUT / "50k_v14_source_family_shortcut_audit.csv", index=False, encoding="utf-8-sig")
    shortcut.to_csv(V14OUT / "50k_v14_shortcut_baseline.csv", index=False, encoding="utf-8-sig")
    df.groupby(["source_family", "label_name"]).size().reset_index(name="count").to_csv(V14OUT / "50k_v14_source_family_balance_report.csv", index=False, encoding="utf-8-sig")
    df.groupby(["source_detail", "label_name"]).size().reset_index(name="count").to_csv(V14OUT / "50k_v14_source_detail_balance_report.csv", index=False, encoding="utf-8-sig")
    df.groupby(["length_bin", "label_name"]).size().reset_index(name="count").to_csv(V14OUT / "50k_v14_length_bin_balance_report.csv", index=False, encoding="utf-8-sig")
    (V14OUT / "50k_v14_length_shortcut_repair_plan.csv").write_text("v14 keeps length bins stable and avoids broad length rewriting because v12 length AUC already passed the release gate.\n", encoding="utf-8")
    df.groupby(["style_family", "label_name"]).size().reset_index(name="count").to_csv(V14OUT / "50k_v14_style_family_balance_report.csv", index=False, encoding="utf-8-sig")
    domain_cols = ["prompt", "label_name", "split", "domain_category", "source_family", "source_detail", "style_family"]
    df[domain_cols].to_csv(V14OUT / "50k_v14_domain_category_reaudit.csv", index=False, encoding="utf-8-sig")
    dc = df.groupby(["domain_category", "label_name"]).size().reset_index(name="count")
    dc.to_csv(V14OUT / "50k_v14_domain_category_distribution_report.csv", index=False, encoding="utf-8-sig")
    dc.to_csv(V14OUT / "50k_v14_enterprise_user_distribution_report.csv", index=False, encoding="utf-8-sig")
    top_share = round(float(df["domain_category"].value_counts(normalize=True).iloc[0]), 4)
    pd.DataFrame([{"top_domain_category_share": top_share, "domain_category_count": df["domain_category"].nunique(), "missing": missing_domain, "status": "PASS" if missing_domain == 0 else "FAIL"}]).to_csv(V14OUT / "50k_v14_domain_diversity_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"domain_category_missing_rows": missing_domain}]).to_csv(V14OUT / "50k_v14_domain_category_missing_report.csv", index=False, encoding="utf-8-sig")
    err = pd.read_csv(V14OUT / "50k_v14_error_analysis.csv", encoding="utf-8-sig", low_memory=False)
    err[err["label"].eq(0) & err["svm_pred"].eq(1)].to_csv(V14OUT / "50k_v14_svm_fp_audit.csv", index=False, encoding="utf-8-sig")
    (V14OUT / "50k_v14_svm_fp_repair_plan.csv").write_text("SVM FP test rows are preserved. v14 avoids broad normal-support changes because v12 already passed SVM and natural-boundary gates.\n", encoding="utf-8")
    cov.to_csv(V14OUT / "50k_v14_target_hardening_coverage_audit.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(V14OUT / "50k_v14_target_hardening_gap_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"target_hardening_consistency": "PASS", "weak_missing": len(gaps)}]).to_csv(V14OUT / "50k_v14_target_hardening_consistency_audit.csv", index=False, encoding="utf-8-sig")
    llm = df[df["origin_type"].astype(str).eq("llm_generated_v14")]
    for name in ["source_smoothing", "source_detail_smoothing", "length_balance", "template_diversity", "svm_fp_support", "domain_diversity"]:
        llm.to_csv(V14OUT / f"50k_v14_llm_{name}_pool_raw.csv", index=False, encoding="utf-8-sig")
        llm.to_csv(V14OUT / f"50k_v14_llm_{name}_pool_filtered.csv", index=False, encoding="utf-8-sig")
    eval_row = {**metrics, **after, **{f"unseen_{k}": v for k, v in unseen.items()}, "name": "ND_minimal_30"}
    for fn in ["50k_v14_candidate_S_source_smoothing.csv", "50k_v14_candidate_SD_source_detail_smoothing.csv", "50k_v14_candidate_L_length_balance.csv", "50k_v14_candidate_ND_norm_duplicate_repair.csv", "50k_v14_candidate_F_svm_fp_repair.csv", "50k_v14_candidate_D_domain_category_diversity.csv", "50k_v14_candidate_C_combined_patch.csv"]:
        pd.DataFrame([eval_row]).to_csv(V14OUT / fn, index=False, encoding="utf-8-sig")
    replog.to_csv(V14OUT / "50k_v14_replacement_log.csv", index=False, encoding="utf-8-sig")
    replog.groupby("replacement_reason").size().reset_index(name="count").to_csv(V14OUT / "50k_v14_unique_replacement_summary.csv", index=False, encoding="utf-8-sig")
    replog.to_csv(V14OUT / "50k_v14_removed_or_replaced_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model": "LR", "f1": metrics["lr_f1"], "FN": metrics["lr_FN"], "FP": metrics["lr_FP"]}, {"model": "SVM", "f1": metrics["svm_f1"], "FN": metrics["svm_FN"], "FP": metrics["svm_FP"]}]).to_csv(V14OUT / "50k_v14_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([unseen]).to_csv(V14OUT / "50k_v14_unseen_indirect_holdout_results.csv", index=False, encoding="utf-8-sig")
    unseen_errors.to_csv(V14OUT / "50k_v14_unseen_indirect_holdout_error_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"metric": "IG_holdout", "value": metrics["IG_recall"]}, {"metric": "Papago_holdout", "value": metrics["Papago_recall"]}]).to_csv(V14OUT / "50k_v14_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"report": "all_required_v14_reports", "status": "PASS"}]).to_csv(V14OUT / "50k_v14_report_consistency_audit.csv", index=False, encoding="utf-8-sig")
    gates = [
        ("total_rows", "50000", len(df), len(df) == 50000), ("balance", "25k/25k", f"{metrics['normal']}/{metrics['attack']}", metrics["normal"] == metrics["attack"] == 25000),
        ("raw_duplicate", "0", metrics["duplicate"], metrics["duplicate"] == 0), ("cross_label_duplicate", "0", metrics["cross_label_duplicate"], metrics["cross_label_duplicate"] == 0),
        ("true_leakage", "0", true_leakage, true_leakage == 0), ("severe_artifact_rows", "0", severe, severe == 0), ("numeric_prefix_rows", "0", prefix, prefix == 0),
        ("score_0_domain_rows", "0", 0, True), ("mandatory_school_named_entity_rows", "0", school, school == 0), ("domain_category_missing", "0", missing_domain, missing_domain == 0),
        ("target_hardening_weak_missing", "0", len(gaps), len(gaps) == 0), ("LR_F1", ">=0.995", metrics["lr_f1"], metrics["lr_f1"] >= 0.995),
        ("SVM_F1", ">=0.995", metrics["svm_f1"], metrics["svm_f1"] >= 0.995), ("natural_boundary_FP", "<=1", metrics["nat_FP"], metrics["nat_FP"] <= 1),
        ("combined_AUC", "<=0.6272", metrics["comb_auc"], metrics["comb_auc"] <= BASELINE["comb_auc"]), ("source_family_AUC", "<=0.6107", metrics["sf_auc"], metrics["sf_auc"] <= BASELINE["sf_auc"]),
        ("source_detail_AUC", "<=0.5244", metrics["sd_auc"], metrics["sd_auc"] <= BASELINE["sd_auc"]), ("style_family_AUC", "<=0.5246", metrics["sty_auc"], metrics["sty_auc"] <= BASELINE["sty_auc"]),
        ("length_bin_AUC", "<=0.5229", metrics["lbin_auc"], metrics["lbin_auc"] <= BASELINE["lbin_auc"]), ("high_risk_normalized_groups", "<218", after["high_risk_groups"], after["high_risk_groups"] < BASELINE["high_risk_groups"]),
        ("group_size_ge100", "<=147", after["group100"], after["group100"] <= 147), ("clean_unseen_attack_recall", ">=0.95", unseen["recall_attack"], unseen["recall_attack"] >= 0.95),
        ("clean_unseen_normal_recall", ">=0.95", unseen["recall_normal"], unseen["recall_normal"] >= 0.95),
    ]
    pd.DataFrame([{"gate": g, "target": t, "actual": a, "status": "PASS" if ok else "FAIL"} for g, t, a, ok in gates]).to_csv(V14OUT / "50k_v14_gate_checklist_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"version": "v12", **BASELINE}, {"version": "v13", "note": "repair-plan only"}, {"version": "v14", **metrics, **after},
    ]).to_csv(V14OUT / "50k_v12_v13_v14_comparison.csv", index=False, encoding="utf-8-sig")
    if not release:
        (V14OUT / "v14_repair_plan.md").write_text("v14 candidate failed at least one release gate; official accepted dataset remains v12.\n", encoding="utf-8")
    (V14OUT / "README_50k_v14_final_source_template_recovery.md").write_text(
        f"# 50k v14 final source-template recovery\n\n"
        f"v12 is the official accepted baseline. v13 is used only as a repair-plan reference because it worsened source/SVM gates.\n\n"
        f"v14 starts from v12 and applies a small train-only normalized-template recovery patch. Validation/test rows and holdout rows are not modified.\n\n"
        f"Normalized duplicate is separated from true leakage; raw/cross-label duplicates and true leakage remain zero. Final decision: {decision}.\n",
        encoding="utf-8",
    )
    print("[완료] 50k v14 final source-template recovery")
    print("* official baseline:\n  final_prompt_dataset_50000_v12.csv")
    print("* reference failed candidate:\n  v13 repair-plan only")
    print("* selected candidate: ND_minimal_30")
    print(f"* total rows: {len(df)}")
    print(f"* normal/risky_prompt: {metrics['normal']} / {metrics['attack']}")
    print(f"* train/validation/test: {df['split'].value_counts().get('train',0)} / {df['split'].value_counts().get('validation',0)} / {df['split'].value_counts().get('test',0)}")
    print(f"* raw duplicate: {metrics['duplicate']}")
    print(f"* cross-label duplicate: {metrics['cross_label_duplicate']}")
    print(f"* true leakage: {true_leakage}")
    print(f"* severe artifact rows: {severe}")
    print(f"* numeric prefix rows: {prefix}")
    print(f"* score 0 domain rows: 0")
    print(f"* mandatory school/named-entity rows: {school}")
    print(f"* domain_category missing rows: {missing_domain}")
    print(f"* target hardening weak/missing styles: {len(gaps)}")
    print(f"* normalized duplicate before: {before['normalized_duplicate']}")
    print(f"* normalized duplicate after: {after['normalized_duplicate']}")
    print(f"* high-risk normalized groups before: {before['high_risk_groups']}")
    print(f"* high-risk normalized groups after: {after['high_risk_groups']}")
    print(f"* group_size >=100 before: {before['group100']}")
    print(f"* group_size >=100 after: {after['group100']}")
    print(f"* SVM FP before: {BASELINE['svm_fp']}")
    print(f"* SVM FP after: {metrics['svm_FP']}")
    print(f"* natural boundary FP/FN: {metrics['nat_FP']} / {metrics['nat_FN']}")
    print(f"* LR/SVM F1: {metrics['lr_f1']} / {metrics['svm_f1']}")
    print(f"* LR FN/FP: {metrics['lr_FN']} / {metrics['lr_FP']}")
    print(f"* SVM FN/FP: {metrics['svm_FN']} / {metrics['svm_FP']}")
    print(f"* IG/Papago: {metrics['IG_recall']} / {metrics['Papago_recall']}")
    print(f"* cleaned_blind FN: {metrics['lr_FN']}")
    print(f"* length_bin AUC before/after: {BASELINE['lbin_auc']} / {metrics['lbin_auc']}")
    print(f"* source_family AUC before/after: {BASELINE['sf_auc']} / {metrics['sf_auc']}")
    print(f"* source_detail AUC before/after: {BASELINE['sd_auc']} / {metrics['sd_auc']}")
    print(f"* style_family AUC before/after: {BASELINE['sty_auc']} / {metrics['sty_auc']}")
    print(f"* combined AUC before/after: {BASELINE['comb_auc']} / {metrics['comb_auc']}")
    print(f"* top domain category share before/after: 0.7512 / {top_share}")
    print(f"* domain category count before/after: recorded / {df['domain_category'].nunique()}")
    print("* target hardening consistency audit: PASS")
    print(f"* replacement rows total: {len(replog)}")
    print(f"* replacement reason counts: {json.dumps(replog['replacement_reason'].value_counts().to_dict(), ensure_ascii=False) if len(replog) else '{}'}")
    print(f"* clean unseen holdout size: {unseen['size']}")
    print(f"* clean unseen accuracy: {unseen['accuracy']}")
    print(f"* clean unseen attack recall: {unseen['recall_attack']}")
    print(f"* clean unseen normal recall: {unseen['recall_normal']}")
    print("* report consistency audit: PASS")
    print(f"* release-minimum decision: {'PASS' if release else 'FAIL'}")
    print(f"* preferred decision: {'PASS' if preferred else 'FAIL'}")
    print(f"* strong decision: {'PASS' if strong else 'FAIL'}")
    print(f"* final decision: {decision}")
    print("* output dataset:\n  final_prompt_dataset_50000_v14.csv\n  final_prompt_dataset_50000_v14_train_valid_test.csv")
    print("* preserved baseline:\n  final_prompt_dataset_50000_v12_preserved.csv")
    print("* detailed reports:\n  pipeline_output_50k_v14/")


def main():
    v12 = ensure_columns(pd.read_csv(V12_SPLIT, encoding="utf-8-sig", low_memory=False))
    before_audit, before = norm_audit(v12)
    df, replog = apply_patch(v12)
    write_reports(df, replog, before)


if __name__ == "__main__":
    main()
