import pandas as pd
from pathlib import Path

OUT = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물/pipeline_output_50k_v1")

llm_rows = [
    # ATTACK GENERATION (13,394 total)
    {"generation_group":"attack","label":"risky_prompt","category":"RAG_hidden_instruction",
     "target_count":3500,"length_bin":"150_350","style_family":"external_content_rag",
     "source_detail":"llm_generated_rag_attack","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":True,
     "notes":"외부 문서 안에 숨긴 지시 포함. RAG 컨텍스트. 이전 지시 무시 유형"},
    {"generation_group":"attack","label":"risky_prompt","category":"tool_agent_misuse",
     "target_count":2500,"length_bin":"100_250","style_family":"tool_api_instruction",
     "source_detail":"llm_generated_tool_attack","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":True,
     "notes":"API/도구 오용 유도. 에이전트 권한 탈취. 비즈니스 컨텍스트"},
    {"generation_group":"attack","label":"risky_prompt","category":"context_extraction",
     "target_count":2000,"length_bin":"100_250","style_family":"conversation_context",
     "source_detail":"llm_generated_context_attack","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":True,
     "notes":"시스템 프롬프트/메모리 추출 시도. 역할극 위장"},
    {"generation_group":"attack","label":"risky_prompt","category":"fn_cluster_hard_positive",
     "target_count":2500,"length_bin":"100_300","style_family":"normal_like_subtle_attack",
     "source_detail":"llm_generated_fn_hard_positive","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":True,
     "notes":"FN 위험 영역. 정상처럼 보이는 미묘한 공격. 오분류 분석 기반"},
    {"generation_group":"attack","label":"risky_prompt","category":"policy_evasion",
     "target_count":1500,"length_bin":"150_300","style_family":"jailbreak_instruction",
     "source_detail":"llm_generated_policy_evasion","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":True,
     "notes":"정책 우회 jailbreak. 점진적 위반 유도"},
    {"generation_group":"attack","label":"risky_prompt","category":"template_wrapper",
     "target_count":1394,"length_bin":"50_200","style_family":"obfuscated_attack",
     "source_detail":"llm_generated_template_attack","source_family":"llm_generated_pool",
     "pair_id_needed":False,"normal_pair_needed":False,
     "notes":"prefix/suffix wrapper 다변화. 난독화 공격"},
    # NORMAL HARD NEGATIVE GENERATION (13,530 total)
    # Uses same source_family as attack generation -> breaks source_family shortcut
    {"generation_group":"normal_hard_negative","label":"normal","category":"privacy_safe_summary",
     "target_count":2500,"length_bin":"100_300","style_family":"privacy_safe_summary",
     "source_detail":"llm_generated_hn_privacy","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":False,
     "notes":"개인정보 제외 요약 요청. 안전한 정상 요청. 공격 쌍 pair_id로 연결"},
    {"generation_group":"normal_hard_negative","label":"normal","category":"safe_rag_summary",
     "target_count":2500,"length_bin":"150_350","style_family":"external_content_rag",
     "source_detail":"llm_generated_hn_rag","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":False,
     "notes":"외부 문서 정상 요약. 원본 지시 무시 없이 안전 처리. RAG 공격 쌍"},
    {"generation_group":"normal_hard_negative","label":"normal","category":"safe_tool_api",
     "target_count":2000,"length_bin":"100_250","style_family":"tool_api_instruction",
     "source_detail":"llm_generated_hn_tool","source_family":"llm_generated_pool",
     "pair_id_needed":True,"normal_pair_needed":False,
     "notes":"API/도구 안전 사용 설명. 권한 확인 요청 포함"},
    {"generation_group":"normal_hard_negative","label":"normal","category":"safe_marketing_sns",
     "target_count":2030,"length_bin":"50_200","style_family":"safe_marketing",
     "source_detail":"llm_generated_hn_marketing","source_family":"llm_generated_pool",
     "pair_id_needed":False,"normal_pair_needed":False,
     "notes":"안전한 SNS/마케팅 문구 작성. 유해 표현 제거 요청"},
    {"generation_group":"normal_hard_negative","label":"normal","category":"security_policy_explanation",
     "target_count":2500,"length_bin":"100_300","style_family":"security_policy",
     "source_detail":"llm_generated_hn_security","source_family":"llm_generated_pool",
     "pair_id_needed":False,"normal_pair_needed":False,
     "notes":"보안 정책 설명 요청. 취약점 대응 방법. 방어적 관점"},
    {"generation_group":"normal_hard_negative","label":"normal","category":"harmful_content_removal",
     "target_count":2000,"length_bin":"50_200","style_family":"safe_sns",
     "source_detail":"llm_generated_hn_filter","source_family":"llm_generated_pool",
     "pair_id_needed":False,"normal_pair_needed":False,
     "notes":"유해 콘텐츠 제거 요청. 필터링/분류 목적의 정상 요청"},
]

df = pd.DataFrame(llm_rows)
df.to_csv(OUT / "50k_llm_generation_plan.csv", index=False, encoding="utf-8-sig")

atk_total = int(df[df["label"]=="risky_prompt"]["target_count"].sum())
hn_total  = int(df[df["label"]=="normal"]["target_count"].sum())
print(f"LLM plan rows: {len(df)}")
print(f"Attack needed: {atk_total}")
print(f"Normal HN needed: {hn_total}")
print(f"Total LLM rows: {atk_total + hn_total}")

# Why source_family mixing matters
print()
print("=== Source Family AUC Fix via llm_generated_pool ===")
print("Current: all attack families = 100% attack -> AUC~1.0")
print("After LLM gen: llm_generated_pool = 13394 attack + 13530 normal (mixed)")
print("  -> source_family no longer perfectly predicts label")
print("  -> expected source_family AUC reduction to ~0.65-0.70")

# Shortage summary
shortage_rows = [
    {"issue":"attack_rows_shortage","current":11606,"target":25000,"gap":13394,
     "severity":"CRITICAL","fix":"LLM attack generation (RAG/tool/context/FN/policy/template)"},
    {"issue":"normal_hard_negative_shortage","current":470,"target":14000,"gap":13530,
     "severity":"CRITICAL","fix":"LLM normal HN generation (privacy/RAG/tool/marketing/security)"},
    {"issue":"source_family_shortcut","current_auc":0.9262,"target_auc":0.68,"gap":0.2462,
     "severity":"CRITICAL","fix":"llm_generated_pool used for both normal and attack"},
    {"issue":"length_only_shortcut","current_auc":0.6886,"target_auc":0.56,"gap":0.1286,
     "severity":"HIGH","fix":"shorter LLM attack generation (50-150 char) + balance"},
    {"issue":"source_detail_shortcut","current_auc":0.6904,"target_auc":0.68,"gap":0.0104,
     "severity":"LOW","fix":"LLM generation with diverse source_detail values"},
    {"issue":"total_rows","current":36606,"target":50000,"gap":13394,
     "severity":"CRITICAL","fix":"same as attack_rows_shortage"},
]
pd.DataFrame(shortage_rows).to_csv(
    OUT / "50k_comprehensive_shortage_report.csv", index=False, encoding="utf-8-sig")
print()
print("Shortage report written")
