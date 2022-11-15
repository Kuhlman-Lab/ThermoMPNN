import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import os.path
import subprocess
import pickle
from tqdm import tqdm
from glob import glob

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

from cache import cache

@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    return parse_PDB(pdb_file)

class KaggleTrainDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):
        source = {
            "train": "jin_train.csv",
            "val": "kaggle.csv",
        }[split]

        self.cfg = cfg
        self.split = split
        self.df = pd.read_csv("data/all_train_data_v17.csv").query("source == @source").reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()


        folder = f"{self.cfg.platform.cache_dir}/kaggle_getitem/"
        os.makedirs(folder, exist_ok=True)
        cache_file = f"{folder}/{self.split}_{index}"
        try:
            with open(cache_file, "rb") as fh:
                ret = pickle.load(fh)
                return ret
        except (FileNotFoundError, EOFError):
            pass


        if self.split == "train":
            pdb_file = f"data/downloaded_pdb/{self.df.PDB[index]}.pdb"
        else:
            pdb_file = f"data/cif/{self.df.CIF[index]}-model_v3.pdb"
        
        pdb_wt = None
        if os.path.exists(pdb_file):
            pdb_wt = parse_pdb_cached(self.cfg, pdb_file)
            pdb_mut = deepcopy(pdb_wt)

        if pdb_wt is None or pdb_wt[0]['seq'] != self.df.sequence[index]:
            print(f"! Error with pdb {pdb_file} !")
            ret = None
        else:
            pdb_mut[0]['seq'] = self.df.mutant_seq[index]

            device = 'cpu'
            wt_feat = tied_featurize([pdb_wt[0]], device, None, None, None, None, None, None, ca_only=False)
            mut_feat = tied_featurize([pdb_mut[0]], device, None, None, None, None, None, None, ca_only=False)
            position = torch.tensor(self.df.position[index], device=device, dtype=torch.long)

            if self.split == "train":
                out = self.df.ddG[index]
            else:
                out = self.df.dTm[index]
            out = torch.tensor(self.df.dTm[index], device=device, dtype=torch.float32)

            ret = wt_feat, mut_feat, out, position

        with open(cache_file, "wb") as fh:
            pickle.dump(ret, fh)

        return ret

if __name__ == "__main__":
    with open("data/cif/convert.pml", "w") as f:
        for fname in glob("data/cif/*.cif"):
            cif = fname.split("/")[-1].split(".")[0]
            f.write(f"load {cif}.cif\n")
            f.write(f"save {cif}.pdb, {cif}\n")