import pandas as pd
from pathlib import Path
OUT = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물/pipeline_output_50k_v1")
pool = pd.read_csv(OUT / "_styled_pool.csv", encoding="utf-8-sig", low_memory=False)
llm_atk = pool[(pool["source_family"]=="llm_generated_pool") & (pool["label"]==1)]
llm_norm = pool[(pool["source_family"]=="llm_generated_pool") & (pool["label"]==0)]
print(f"LLM attack in styled pool: {len(llm_atk)}")
print(f"LLM normal in styled pool: {len(llm_norm)}")
total_atk = int((pool["label"]==1).sum())
total_norm = int((pool["label"]==0).sum())
print(f"Total attack: {total_atk}")
print(f"Total normal: {total_norm}")
print(f"LLM attack shortfall from target 13394: {13394 - len(llm_atk)}")
