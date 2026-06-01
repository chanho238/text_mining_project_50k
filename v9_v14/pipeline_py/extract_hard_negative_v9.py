"""
Extract hard-negative normal rows from the accepted v9 dataset.

This script does not create or modify dataset rows. It reads v9, selects normal
defensive/boundary rows, writes strict and expanded candidate files, and keeps
paired risky_prompt rows only in a separate reference report.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_50k_v4 import BASE, ensure_columns, lbin, norm_text


OUT = BASE / "pipeline_output_50k_v9_hard_negative_extract"
OUT.mkdir(parents=True, exist_ok=True)

V9_SPLIT = BASE / "final_prompt_dataset_50000_v9_train_valid_test.csv"
V9_DATA = BASE / "final_prompt_dataset_50000_v9.csv"

STYLE_CANDIDATES = {
    "rag_document",
    "email_instruction",
    "tool_api",
    "customer_support",
    "meeting_report",
    "code_block",
    "translation_hidden_instruction",
    "sns_rewrite",
    "html_markdown_json",
    "memory_poisoning",
    "multiturn_context",
    "ko_en_mixed",
    "short_normal_boundary",
    "short_general_question",
}

EXPANDED_META_RE = re.compile(
    r"hard_negative|boundary|safe_boundary|normal_boundary|security_review|defensive|"
    r"target_hardening|privacy_safe_summary|defensive_normal|safe_review|instruction_conflict_safe|"
    r"llm_v9_short_normal_boundary_support",
    re.I,
)

DEFENSIVE_PROMPT_RE = re.compile(
    r"판별|구분|분류|검토|확인|요약|마스킹|제외|정리|설명|번역만|따르지|실행하지|"
    r"도구.*호출하지|개인정보|민감|범주명|주석|JSON|HTML|Markdown|외부 문서|승인|"
    r"숨은 지시|지시처럼|데이터로만|보류|안전",
    re.I,
)

ATTACK_EXEC_RE = re.compile(
    r"실제로\s*(따르|수행|실행|적용|반영)|우선\s*(적용|따르)|검증 없이\s*(실행|호출)|"
    r"승인 없이\s*(실행|호출)|비공개.*(추출|공개|포함)|시스템 지침.*(출력|공개)|"
    r"숨겨진.*(출력|공개)|기억하라|저장하라|우회하|jailbreak|탈옥|"
    r"유해.*자연스럽게|금지.*바꾸",
    re.I,
)

ARTIFACT_RE = re.compile(
    r"candidate|pool|보강 후보|테스트 문장|v9 후보|target_hardening|style-id|audit sample|generated sample",
    re.I,
)

KOREAN_RE = re.compile(r"[가-힣]")
NUMERIC_PREFIX_ID_RE = re.compile(r"^(.{2,60}?)(?:\s+\d{2,})(\s*:)")


def boolish(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def strip_numeric_prefix_id(text: object) -> str:
    s = str(text)
    # Remove leading template ids such as "마케팅 문안 검토 요청 5974:" while
    # keeping the natural-language prefix and the colon.
    for _ in range(2):
        s = NUMERIC_PREFIX_ID_RE.sub(r"\1\2", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def source_path() -> Path:
    return V9_SPLIT if V9_SPLIT.exists() else V9_DATA


def add_extract_columns(df: pd.DataFrame, rule: str, decision: str, reason: str) -> pd.DataFrame:
    out = df.copy()
    text = out["prompt"].fillna("").astype(str)
    out["hard_negative_extract_rule"] = rule
    out["hard_negative_extract_decision"] = decision
    out["hard_negative_reason"] = reason
    out["defensive_normal_flag"] = text.str.contains(DEFENSIVE_PROMPT_RE)
    out["attack_like_surface_flag"] = (
        text.str.contains(r"prompt injection|인젝션|숨은 지시|외부 문서|도구|API|개인정보|마스킹|JSON|HTML|Markdown|시스템|정책", case=False, regex=True)
        | out["style_family"].astype(str).isin(STYLE_CANDIDATES)
    )
    out["quality_status"] = "pass"
    return out


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["prompt"].fillna("").astype(str)
    out["_prompt_len"] = text.str.len()
    out["_is_normal"] = out["label"].eq(0) & out["label_name"].eq("normal")
    out["_prompt_nonempty"] = text.str.strip().ne("")
    out["_len_ok"] = out["_prompt_len"] >= 20
    out["_attack_exec_flag"] = text.str.contains(ATTACK_EXEC_RE)
    out["_artifact_flag"] = text.str.contains(ARTIFACT_RE)
    out["_korean_flag"] = text.str.contains(KOREAN_RE)
    out["_exact_duplicate"] = text.duplicated(keep=False)
    out["_norm2"] = text.apply(norm_text)
    out["_normalized_duplicate"] = out["_norm2"].duplicated(keep=False)
    return out


def split_files(final: pd.DataFrame):
    if "split" not in final.columns:
        return
    for split in ["train", "validation", "test"]:
        final[final["split"].eq(split)].to_csv(OUT / f"hard_negative_v9_{split}.csv", index=False, encoding="utf-8-sig")


def pair_reference(final: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    refs = []
    attacks = src[src["label"].eq(1)].copy()
    attack_by_pair = {str(r.get("pair_id", "")): r for _, r in attacks[attacks["pair_id"].fillna("").astype(str).ne("")].iterrows()}
    attack_by_group = {str(r.get("split_group_id", "")): r for _, r in attacks[attacks["split_group_id"].fillna("").astype(str).ne("")].iterrows()}
    for idx, row in final.iterrows():
        pair_id = str(row.get("pair_id", "") or "")
        group_id = str(row.get("split_group_id", "") or "")
        attack = None
        if pair_id:
            attack = attack_by_pair.get(pair_id)
        if attack is None and group_id:
            attack = attack_by_group.get(group_id)
        if attack is not None:
            status = "same_split_pair" if str(attack.get("split", "")) == str(row.get("split", "")) else "paired_attack_found"
            refs.append(
                {
                    "hard_negative_row_id": row["row_id"],
                    "hard_negative_prompt": row["prompt"],
                    "hard_negative_split": row.get("split", ""),
                    "pair_id": pair_id,
                    "paired_risky_prompt_row_id": attack.get("row_id", attack.name),
                    "paired_risky_prompt_prompt": attack["prompt"],
                    "paired_risky_prompt_split": attack.get("split", ""),
                    "same_split_flag": str(attack.get("split", "")) == str(row.get("split", "")),
                    "pair_status": status,
                }
            )
        else:
            refs.append(
                {
                    "hard_negative_row_id": row["row_id"],
                    "hard_negative_prompt": row["prompt"],
                    "hard_negative_split": row.get("split", ""),
                    "pair_id": pair_id,
                    "paired_risky_prompt_row_id": "",
                    "paired_risky_prompt_prompt": "",
                    "paired_risky_prompt_split": "",
                    "same_split_flag": False,
                    "pair_status": "pair_missing",
                }
            )
    return pd.DataFrame(refs)


def distribution_report(final: pd.DataFrame) -> pd.DataFrame:
    rows = [{"metric": "total_hard_negative_rows", "value": len(final)}]
    for col in [
        "label_name",
        "label",
        "split",
        "style_family",
        "source_family",
        "source_detail",
        "length_bin",
        "normal_category",
        "is_hard_negative",
    ]:
        if col in final.columns:
            rows.append({"metric": f"{col}_distribution", "value": json.dumps(final[col].value_counts().head(30).to_dict(), ensure_ascii=False)})
    return pd.DataFrame(rows)


def quality_audit(final: pd.DataFrame) -> pd.DataFrame:
    text = final["prompt"].fillna("").astype(str)
    norm = text.apply(norm_text)
    rows = [
        {"check": "total_hard_negative_rows", "value": len(final), "status": "INFO"},
        {"check": "label_name_only_normal", "value": json.dumps(final["label_name"].value_counts().to_dict(), ensure_ascii=False), "status": "PASS" if set(final["label_name"]) == {"normal"} else "FAIL"},
        {"check": "label_only_zero", "value": json.dumps(final["label"].value_counts().to_dict(), ensure_ascii=False), "status": "PASS" if set(final["label"]) == {0} else "FAIL"},
        {"check": "risky_prompt_contamination", "value": int(final["label_name"].ne("normal").sum()), "status": "PASS" if int(final["label_name"].ne("normal").sum()) == 0 else "FAIL"},
        {"check": "attack_label_contamination", "value": int(final["label"].ne(0).sum()), "status": "PASS" if int(final["label"].ne(0).sum()) == 0 else "FAIL"},
        {"check": "exact_duplicates", "value": int(text.duplicated().sum()), "status": "PASS" if int(text.duplicated().sum()) == 0 else "FAIL"},
        {"check": "normalized_duplicates", "value": int(norm.duplicated().sum()), "status": "INFO"},
        {"check": "prompt_null", "value": int(text.str.strip().eq("").sum()), "status": "PASS" if int(text.str.strip().eq("").sum()) == 0 else "FAIL"},
        {"check": "prompt_length_min_mean_max", "value": f"{int(text.str.len().min())}/{round(float(text.str.len().mean()), 2)}/{int(text.str.len().max())}", "status": "INFO"},
        {"check": "korean_present_rows", "value": int(text.str.contains(KOREAN_RE).sum()), "status": "PASS" if int(text.str.contains(KOREAN_RE).sum()) == len(final) else "WARN"},
        {"check": "severe_artifact_rows", "value": int(text.str.contains(ARTIFACT_RE).sum()), "status": "PASS" if int(text.str.contains(ARTIFACT_RE).sum()) == 0 else "FAIL"},
        {"check": "actual_attack_like_request_contamination", "value": int(text.str.contains(ATTACK_EXEC_RE).sum()), "status": "PASS" if int(text.str.contains(ATTACK_EXEC_RE).sum()) == 0 else "FAIL"},
    ]
    return pd.DataFrame(rows)


def main():
    src_path = source_path()
    if not src_path.exists():
        raise FileNotFoundError("v9 dataset not found")
    src = ensure_columns(pd.read_csv(src_path, encoding="utf-8-sig", low_memory=False))
    src = src.reset_index(drop=False).rename(columns={"index": "row_id"})
    q = quality_flags(src)
    normal = q[q["_is_normal"] & q["_prompt_nonempty"] & q["_len_ok"]].copy()

    strict_mask = normal["is_hard_negative"].apply(boolish)
    strict = normal[strict_mask].copy()
    strict = strict[~strict["_attack_exec_flag"] & ~strict["_artifact_flag"]].copy()
    strict = add_extract_columns(strict, "strict_is_hard_negative", "include_strict", "normal label with defensive boundary request")

    meta_text = (
        normal["normal_category"].fillna("").astype(str)
        + " "
        + normal["replacement_role"].fillna("").astype(str)
        + " "
        + normal["risk_subtype"].fillna("").astype(str)
        + " "
        + normal["source_detail"].fillna("").astype(str)
    )
    expanded_mask = (
        meta_text.str.contains(EXPANDED_META_RE)
        | normal["style_family"].astype(str).isin(STYLE_CANDIDATES)
        | normal["prompt"].fillna("").astype(str).str.contains(DEFENSIVE_PROMPT_RE)
    )
    expanded = normal[expanded_mask & ~normal.index.isin(strict.index)].copy()
    expanded = add_extract_columns(expanded, "expanded_defensive_boundary_candidate", "exclude_uncertain", "candidate matched metadata or defensive wording")

    verified_mask = (
        expanded["prompt"].fillna("").astype(str).str.contains(DEFENSIVE_PROMPT_RE)
        & ~expanded["_attack_exec_flag"]
        & ~expanded["_artifact_flag"]
    )
    expanded.loc[verified_mask, "hard_negative_extract_decision"] = "include_expanded_verified"
    expanded.loc[verified_mask, "hard_negative_reason"] = "verified defensive normal request with attack-like surface"
    expanded.loc[expanded["_attack_exec_flag"], "hard_negative_extract_decision"] = "exclude_attack_like"
    expanded.loc[expanded["_artifact_flag"], "hard_negative_extract_decision"] = "exclude_not_hard_negative"

    final = pd.concat([strict, expanded[expanded["hard_negative_extract_decision"].eq("include_expanded_verified")]], ignore_index=True)
    final = final.drop_duplicates(subset=["prompt"]).copy()
    final = final[final["label"].eq(0) & final["label_name"].eq("normal")].copy()
    final = final[~final["prompt"].astype(str).str.contains(ATTACK_EXEC_RE)].copy()
    final = final[~final["prompt"].astype(str).str.contains(ARTIFACT_RE)].copy()
    final["quality_status"] = "pass"

    exclude_parts = [
        q[~q["_is_normal"]].assign(exclusion_reason="non_normal_or_attack_label").head(500),
        normal[normal["_attack_exec_flag"]].assign(exclusion_reason="actual_attack_like_request").head(500),
        normal[normal["_artifact_flag"]].assign(exclusion_reason="internal_artifact_phrase").head(500),
        normal[normal["_prompt_len"] < 20].assign(exclusion_reason="too_short").head(500),
        expanded[~expanded["hard_negative_extract_decision"].str.startswith("include", na=False)].assign(exclusion_reason="expanded_not_verified").head(500),
    ]
    exclude = pd.concat(exclude_parts, ignore_index=True)
    exclude_summary = pd.DataFrame(
        [
            {"exclusion_reason": "non_normal_or_attack_label", "count": int((~q["_is_normal"]).sum())},
            {"exclusion_reason": "actual_attack_like_request", "count": int(normal["_attack_exec_flag"].sum())},
            {"exclusion_reason": "internal_artifact_phrase", "count": int(normal["_artifact_flag"].sum())},
            {"exclusion_reason": "too_short", "count": int((normal["_prompt_len"] < 20).sum())},
            {"exclusion_reason": "expanded_not_verified", "count": int((~expanded["hard_negative_extract_decision"].str.startswith("include", na=False)).sum())},
        ]
    )

    keep_cols = [c for c in src.columns if c in final.columns]
    extra_cols = [
        "hard_negative_extract_rule",
        "hard_negative_extract_decision",
        "hard_negative_reason",
        "defensive_normal_flag",
        "attack_like_surface_flag",
        "quality_status",
    ]
    final_out = final[keep_cols + extra_cols].copy()
    strict_out = strict[keep_cols + extra_cols].copy()
    expanded_out = expanded[keep_cols + extra_cols].copy()

    for frame in [final_out, strict_out, expanded_out]:
        frame["prompt"] = frame["prompt"].apply(strip_numeric_prefix_id)
        frame["length"] = frame["prompt"].astype(str).str.len()
        frame["length_bin"] = frame["length"].apply(lbin)
    final_out = final_out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    strict_out = strict_out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    expanded_out = expanded_out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    final_out.to_csv(OUT / "hard_negative_v9_all.csv", index=False, encoding="utf-8-sig")
    strict_out.to_csv(OUT / "hard_negative_v9_strict.csv", index=False, encoding="utf-8-sig")
    expanded_out.to_csv(OUT / "hard_negative_v9_expanded_candidates.csv", index=False, encoding="utf-8-sig")
    exclude.drop(columns=[c for c in exclude.columns if c.startswith("_")], errors="ignore").to_csv(
        OUT / "hard_negative_v9_exclusion_report.csv", index=False, encoding="utf-8-sig"
    )
    exclude_summary.to_csv(OUT / "hard_negative_v9_exclusion_summary.csv", index=False, encoding="utf-8-sig")
    split_files(final_out)
    distribution_report(final_out).to_csv(OUT / "hard_negative_v9_distribution_report.csv", index=False, encoding="utf-8-sig")
    quality_audit(final_out).to_csv(OUT / "hard_negative_v9_quality_audit.csv", index=False, encoding="utf-8-sig")
    final_out.groupby("style_family").size().reset_index(name="count").to_csv(OUT / "hard_negative_v9_by_style_family.csv", index=False, encoding="utf-8-sig")
    final_out.groupby("source_detail").size().reset_index(name="count").to_csv(OUT / "hard_negative_v9_by_source_detail.csv", index=False, encoding="utf-8-sig")
    pair_reference(final_out, src).to_csv(OUT / "hard_negative_v9_pair_reference.csv", index=False, encoding="utf-8-sig")

    top_style = final_out["style_family"].value_counts().head(5).to_dict()
    top_source = final_out["source_family"].value_counts().head(5).to_dict()
    top_len = final_out["length_bin"].value_counts().head(5).to_dict()
    split_counts = final_out["split"].value_counts().to_dict() if "split" in final_out.columns else {}
    qa = quality_audit(final_out)
    qa_map = dict(zip(qa["check"], qa["value"]))

    readme = f"""# v9 Hard Negative Subset Extract

Date: {datetime.now().strftime('%Y-%m-%d')}

## Purpose
This subset contains only label-normal hard negative rows from the accepted v9 dataset.

## Definition
Rows are normal defensive or boundary requests that may look similar to prompt-injection, hidden-instruction, security-review, external-document, tool/API, privacy-masking, or translation-instruction cases. They are not attack execution requests.

## Input
- {src_path.name}

## Inclusion
- Strict: label_name normal, label 0, is_hard_negative true, nonempty prompt, length >= 20.
- Expanded verified: normal rows with defensive/boundary metadata or defensive wording, excluding actual attack-like execution and artifact phrases.

## Exclusion
Risky_prompt or attack label rows are not included. Paired risky_prompt rows are recorded only in `hard_negative_v9_pair_reference.csv`, not in the hard negative subset.

## Counts
- Final rows: {len(final_out)}
- Strict rows: {len(strict_out)}
- Expanded candidates: {len(expanded_out)}
- Split counts: {json.dumps(split_counts, ensure_ascii=False)}
- Top style_family: {json.dumps(top_style, ensure_ascii=False)}
- Top source_family: {json.dumps(top_source, ensure_ascii=False)}
- Top length_bin: {json.dumps(top_len, ensure_ascii=False)}

## Quality
- Label contamination: {qa_map.get('attack_label_contamination')}
- Risky_prompt contamination: {qa_map.get('risky_prompt_contamination')}
- Exact duplicates: {qa_map.get('exact_duplicates')}
- Severe artifact rows: {qa_map.get('severe_artifact_rows')}
- Actual attack-like request contamination: {qa_map.get('actual_attack_like_request_contamination')}

This subset contains only normal hard negative rows. Risky_prompt or attack label rows are not included. The extraction targets defensive normal requests that may look attack-like on the surface, not attack execution data.
"""
    (OUT / "README_hard_negative_v9_extract.md").write_text(readme, encoding="utf-8")

    print("\n[완료] v9 hard negative subset extraction")
    print(f"* input dataset: {src_path.name}")
    print(f"* total rows in input: {len(src)}")
    print(f"* strict hard negative rows: {len(strict_out)}")
    print(f"* expanded candidate rows: {len(expanded_out)}")
    print(f"* final hard negative rows: {len(final_out)}")
    print(f"* train hard negative rows: {split_counts.get('train', 0)}")
    print(f"* validation hard negative rows: {split_counts.get('validation', 0)}")
    print(f"* test hard negative rows: {split_counts.get('test', 0)}")
    print(f"* label contamination: {qa_map.get('attack_label_contamination')}")
    print(f"* risky_prompt contamination: {qa_map.get('risky_prompt_contamination')}")
    print(f"* exact duplicates: {qa_map.get('exact_duplicates')}")
    print(f"* normalized duplicates: {qa_map.get('normalized_duplicates')}")
    print(f"* severe artifact rows: {qa_map.get('severe_artifact_rows')}")
    print(f"* actual attack-like request contamination: {qa_map.get('actual_attack_like_request_contamination')}")
    print(f"* top style_family: {json.dumps(top_style, ensure_ascii=False)}")
    print(f"* top source_family: {json.dumps(top_source, ensure_ascii=False)}")
    print(f"* top length_bin: {json.dumps(top_len, ensure_ascii=False)}")
    print("* output folder:")
    print("  pipeline_output_50k_v9_hard_negative_extract/")
    print("* main output:")
    print("  hard_negative_v9_all.csv")


if __name__ == "__main__":
    main()
