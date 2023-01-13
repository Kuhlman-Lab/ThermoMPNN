from collections import defaultdict
import torch
import pandas as pd
from Bio import pairwise2
from fireprot_dataset import Mutation, parse_pdb_cached, alphabet, get_msa_hist, seq1_index_to_seq2_index

class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, df = None):

        self.cfg = cfg
        self.split = split

        if df is None:
            fname = "data/mega_scale/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv"
            df = pd.read_csv(fname)

        self.df = df

        # for now -- yeet the names without a pdb file
        clusters = defaultdict(list)
        for name in df.WT_name.unique():
            pdb_name = name.split(".pdb")[0].replace("|",":")
            clusters[pdb_name].append(name)

        split_fracs = {
            "val": (0.0, 0.1),
            "test": (0.1, 0.2),
            "train": (0.2, 1.0)
        }
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "all": list(df.WT_name.unique()),
        }

        cluster_keys = list(clusters.keys())
        self.name_to_cluster = {}
        for cluster, names in clusters.items():
            for name in names:
                self.name_to_cluster[name] = cluster

        to_skip = [ "2HBB_con", "2J6K_con" ]
        for key, (start, stop) in split_fracs.items():
            split_clusters = cluster_keys[int(start*len(cluster_keys)):int(stop*len(cluster_keys))]
            self.split_wt_names[key] = []
            for cluster_key in split_clusters:
                should_add = True
                if cluster_key in to_skip:
                    print(f"Skipping {cluster_key}")
                    should_add = False
                    for wt_name in clusters[cluster_key]:
                        self.split_wt_names["all"].remove(wt_name)
                    continue
                if key in ("val", "test"):
                    for pdb_id in ("1UZC", "1UBQ", "1MJC", "1PGA", "1YU5"):
                        if pdb_id in cluster_key:
                            print("Skipping", key, pdb_id)
                            should_add = False
                            break
                if should_add:
                    self.split_wt_names[key] += clusters[cluster_key]

        self.wt_seqs = {}
        self.wt_dGs = {}
        self.mut_rows = {}
        self.wt_names = self.split_wt_names[self.split]

        for wt_name in self.wt_names:
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
            self.wt_dGs[wt_name] = wt_rows.deltaG.median()

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        mut_data = self.mut_rows[wt_name]
        wt_seq = self.wt_seqs[wt_name]
        wt_dG = self.wt_dGs[wt_name]
        cluster = self.name_to_cluster[wt_name]

        pdb_file = f"data/mega_scale/AlphaFold_model_PDBs/{cluster}.pdb"
        pdb = parse_pdb_cached(self.cfg, pdb_file)
        assert len(pdb[0]["seq"]) == len(wt_seq)
        pdb[0]["seq"] = wt_seq

        msa_align = None
        msa_seq = None
        try:
            all_msa_hist, msa_seq = get_msa_hist(self.cfg, f"data/msas/{wt_name}.a3m")
            msa_align, *rest = pairwise2.align.globalxx(wt_seq, msa_seq)
        except FileNotFoundError:
            pass
            # print(f"No msa for {wt_name} ({len(mut_data)}), skipping")

        

        mutations = []
        for i, row in mut_data.iterrows():
            # no insertions, deletions, or double mutants
            if "ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type:
                continue
            assert len(row.aa_seq) == len(wt_seq)
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1
            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut

            msa_hist = torch.zeros((len(alphabet,)))
            if row.ddG_ML == '-':
                continue # unreliable data

            ddG = float(row.ddG_ML) # row.deltaG - wt_dG
            ddG = -torch.tensor([ddG], dtype=torch.float32)

            msa_idx = None
            if msa_align is not None:
                msa_idx = seq1_index_to_seq2_index(msa_align, idx)

            msa_hist = torch.zeros((len(alphabet,)))
            if msa_idx is not None:
                assert msa_seq[msa_idx] == wt
                msa_hist = all_msa_hist[msa_idx]

            mutations.append(Mutation(idx, wt, mut, msa_hist, ddG, None))

        return pdb, mutations