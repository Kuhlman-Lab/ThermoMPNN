
import pandas as pd
import subprocess

df = pd.read_csv("data/v2_dataset_12072022/fireprotDB_cleaned_allproteins_allmutations.csv")
seen_uniprots = set()
for i, row in df.iterrows():
    if row.uniprot_id in seen_uniprots: continue
    seen_uniprots.add(row.uniprot_id)
    subprocess.run("/proj/kuhl_lab/alphafold/run/get_msa.py", "--sequence", row.sequence, "--output-dir", f"data/{row.uniprot_id}")
    break