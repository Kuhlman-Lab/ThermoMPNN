import sys, os
sys.path.append("/proj/kuhl_lab/alphafold/alphafold")
sys.path.append("/proj/kuhl_lab/alphafold/run/")
import pandas as pd
from features import getRawInputs


def get_msa(sequence, uniprot):
    # Form query object.
    query = ('_INPUT_', [sequence])

    # Get MSAs
    raw_inputs = getRawInputs(
        queries=[query],
        msa_mode="MMseqs2-U+E",
        output_dir="./",
        design_run=True)
    
    # Write MSA file
    with open(f"data/msas/{uniprot}.a3m", 'w') as f:
        f.writelines(raw_inputs[sequence][0])


if __name__ == "__main__":
    # fireprot data
    # seq = "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
    # name = "kaggle"
    # get_msa(seq, name)
    # df = pd.read_csv("data/v2_dataset_12072022/fireprotDB_cleaned_allproteins_allmutations.csv")
    # seen_uniprots = set()
    # for i, row in df.iterrows():
    #     if row.uniprot_id in seen_uniprots: continue
    #     print(i, row.uniprot_id)
    #     seen_uniprots.add(row.uniprot_id)
    #     get_msa(row.sequence, row.uniprot_id)
    # rocklin data
    fname = "data/mega_scale/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv"
    df = pd.read_csv(fname)
    seen_names = set()
    for i, row in df.iterrows():
        if row.WT_name in seen_names: continue
        seen_names.add(row.WT_name)
        get_msa(row.aa_seq_full, row.WT_name)
