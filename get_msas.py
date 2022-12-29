
import pandas as pd
import subprocess

# fireprot
# df = pd.read_csv("data/v2_dataset_12072022/fireprotDB_cleaned_allproteins_allmutations.csv")
# seen_uniprots = set()
# for i, row in df.iterrows():
#     if row.uniprot_id in seen_uniprots: continue
#     seen_uniprots.add(row.uniprot_id)
#     subprocess.run("python", "/proj/kuhl_lab/alphafold/run/get_msa.py", "--sequence", row.sequence, "--output-dir", f"data/{row.uniprot_id}")
#     break

# rocklin

fname = "data/mega_scale/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv"
df = pd.read_csv(fname)
seen_names = set()
for i, row in df.iterrows():
    if row.WT_name in seen_names: continue
    seen_names.add(row.WT_name)
    subprocess.run("python", "/proj/kuhl_lab/alphafold/run/get_msa.py", "--sequence", row.aa_seq_full, "--output-dir", f"data/{row.WT_name}")