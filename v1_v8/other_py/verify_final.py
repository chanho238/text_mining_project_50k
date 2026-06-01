import pandas as pd
from pathlib import Path

OUT  = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물/pipeline_output_50k_v1")
BASE = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물")

df = pd.read_csv(BASE / "final_prompt_dataset_50000_v1_train_valid_test.csv",
                 encoding="utf-8-sig", low_memory=False)

print("=== Final Dataset ===")
print(f"Total rows : {len(df):,}")
print(f"Columns    : {len(df.columns)} ({list(df.columns)[:6]}...)")
print()
print("Split distribution:")
print(df.groupby(["split","label_name"]).size().to_string())
print()
print("Source family:")
print(df.groupby("source_family").size().sort_values(ascending=False).to_string())
print()
print("Length by label:")
print(df.groupby("label_name")["length"].describe()[["mean","min","max"]].round(1).to_string())

auc = pd.read_csv(OUT / "50k_shortcut_baseline.csv", encoding="utf-8-sig")
print()
print("=== Shortcut AUC ===")
print(auc[["baseline","auc","hard_limit","status"]].to_string(index=False))

met = pd.read_csv(OUT / "50k_model_metrics.csv", encoding="utf-8-sig")
print()
print("=== Baseline Model ===")
print(met.to_string(index=False))

dup = pd.read_csv(OUT / "50k_duplicate_report.csv", encoding="utf-8-sig")
print()
print("=== Dedup ===")
print(dup.to_string(index=False))

leak = pd.read_csv(OUT / "50k_leakage_report.csv", encoding="utf-8-sig")
print()
print("=== Leakage ===")
print(leak.to_string(index=False))
