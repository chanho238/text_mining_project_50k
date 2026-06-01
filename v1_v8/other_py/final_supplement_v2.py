"""Generate 500 more rows to cover S15k gap + include full v1 LLM pool in pipeline."""
import random, re, pandas as pd
from pathlib import Path
random.seed(99999)
BASE = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물")
OUT  = BASE / "pipeline_output_50k_v2"
V1OUT= BASE / "pipeline_output_50k_v1"

KO_RE = re.compile(r"[가-힣]")
def has_ko(s): return bool(KO_RE.search(str(s)))
def lbin(n):
    if n<100: return "20_99"
    if n<200: return "100_199"
    if n<300: return "200_299"
    if n<400: return "300_399"
    return "400_499"
def norm_t(s):
    s=str(s).lower().strip()
    s=re.sub(r"\d{3,}","NUM",s)
    s=re.sub(r"[^\w\s가-힣]"," ",s)
    return re.sub(r"\s+"," ",s).strip()[:180]

# Current pool
pool = pd.read_csv(OUT/"50k_v2_llm_candidate_pool_raw.csv", encoding="utf-8-sig", low_memory=False)
# Full v1 LLM pool (use rows NOT in v1 dataset)
v1_final = pd.read_csv(BASE/"final_prompt_dataset_50000_v1_train_valid_test.csv",
                        encoding="utf-8-sig", low_memory=False, usecols=["prompt"])
v1_llm_full = pd.read_csv(V1OUT/"50k_llm_generated_pool.csv", encoding="utf-8-sig", low_memory=False)

v1_set = set(v1_final["prompt"].apply(norm_t))
pool_set= set(pool["prompt"].apply(norm_t))
all_seen = v1_set | pool_set

# v1 LLM pool rows NOT in final dataset
v1_llm_full["_norm"] = v1_llm_full["prompt"].apply(norm_t)
v1_llm_unused = v1_llm_full[~v1_llm_full["_norm"].isin(v1_set)].copy()
print(f"v1 LLM unused rows: {len(v1_llm_unused)} "
      f"(atk={int((v1_llm_unused['label']==1).sum())}, "
      f"nrm={int((v1_llm_unused['label']==0).sum())})")

# Add unused v1 LLM rows to pool (they're already diverse)
v1_llm_unused_clean = v1_llm_unused.drop(columns=["_norm"], errors="ignore")
v1_llm_unused_clean["generation_group"] = "v1_llm_unused"
v1_llm_unused_clean["file_source"] = "llm_generated_v1_unused"

combined_pool = pd.concat([pool, v1_llm_unused_clean], ignore_index=True)
combined_pool = combined_pool.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

print(f"Combined pool: {len(combined_pool)} "
      f"(atk={int((combined_pool['label']==1).sum())}, "
      f"nrm={int((combined_pool['label']==0).sum())})")
combined_pool.to_csv(OUT/"50k_v2_llm_candidate_pool_raw.csv", index=False, encoding="utf-8-sig")
print("Saved combined pool")

# Check S15k feasibility
v1_ds = pd.read_csv(BASE/"final_prompt_dataset_50000_v1_train_valid_test.csv",
                    encoding="utf-8-sig", low_memory=False)
v1_ds["label"] = v1_ds["label"].astype(int)
v1_pure_atk = v1_ds[(v1_ds["source_family"]!="llm_generated_pool") & (v1_ds["label"]==1)]
v1_pure_nrm = v1_ds[(v1_ds["source_family"]!="llm_generated_pool") & (v1_ds["label"]==0)]
v1_llm_atk  = v1_ds[(v1_ds["source_family"]=="llm_generated_pool") & (v1_ds["label"]==1)]
v1_llm_nrm  = v1_ds[(v1_ds["source_family"]=="llm_generated_pool") & (v1_ds["label"]==0)]

new_pool_atk = combined_pool[combined_pool["label"]==1]
new_pool_nrm = combined_pool[combined_pool["label"]==0]

avail_atk = int(len(v1_llm_atk) + len(new_pool_atk))
avail_nrm = int(len(v1_llm_nrm) + len(new_pool_nrm))
print(f"\nAvailability check:")
print(f"  avail_atk={avail_atk}, need for S15k={25000-7500}={17500} -> {'OK' if avail_atk>=17500 else 'FAIL'}")
print(f"  avail_nrm={avail_nrm}, need for S15k={25000-7500}={17500} -> {'OK' if avail_nrm>=17500 else 'FAIL'}")
print(f"  avail_atk={avail_atk}, need for S10k={25000-5000}={20000} -> {'OK' if avail_atk>=20000 else 'FAIL'}")
print(f"  avail_nrm={avail_nrm}, need for S10k={25000-5000}={20000} -> {'OK' if avail_nrm>=20000 else 'FAIL'}")
