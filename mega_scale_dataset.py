import torch
import pandas as pd
from fireprot_dataset import Mutation, parse_pdb_cached, alphabet

class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        fname = "data/mega_scale/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv"
        df = pd.read_csv(fname)
        # df["name_corrected"] = df.name.str.split("_").str[0]

        self.df = df

        # for now -- yeet the names without a pdb file
        wt_names = [ name for name in df.WT_name.unique() if "_" not in name ]

        split_fracs = {
            "val": (0.0, 0.1),
            "test": (0.1, 0.2),
            "train": (0.2, 1.0)
        }
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": []
        }

        for key, (start, stop) in split_fracs.items():
            self.split_wt_names[key] = wt_names[int(start*len(wt_names)):int(stop*len(wt_names))]

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

        pdb_file = f"data/mega_scale/AlphaFold_model_PDBs/{wt_name}"
        pdb = parse_pdb_cached(self.cfg, pdb_file)
        assert pdb[0]["seq"] == wt_seq

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
            ddG = row.deltaG - wt_dG
            ddG = torch.tensor([ddG], dtype=torch.float32)

            mutations.append(Mutation(idx, wt, mut, msa_hist, ddG, None))

        return pdb, mutations