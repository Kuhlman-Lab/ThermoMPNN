from dataclasses import dataclass
import os
import pickle
from typing import Optional
import torch
import pandas as pd

from cache import cache
from protein_mpnn_utils import parse_PDB

@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    return parse_PDB(pdb_file)

@dataclass
class Mutation:
    position: int
    mutation: str
    ddG: Optional[float] = None
    dTm: Optional[float] = None

class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        df = pd.read_csv("data/v1_dataset_11152022/3_fireprotDB_curated_v1_bestpH_avgDupes.csv")
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}

        # group all the data by wildtype seq
        for wt_seq in df.sequence_corrected.unique():
            self.seq_to_data[wt_seq] = df.query("sequence_corrected == @wt_seq").reset_index(drop=True)
        
        sorted_seqs = list(sorted(self.seq_to_data.keys(), key=lambda seq: len(self.seq_to_data[seq])))
        
        # divide all the sequences into train/val/test
        split_fracs = {
            "val": 0.1,
            "test": 0.1,
            "train": 0.8
        }
        splits = {
            "val": [],
            "test": [],
            "train": []
        }
        split_iter = iter(split_fracs.keys())
        cur_split = next(split_iter)
        tot_data = len(df)
        cur_data_len = 0
        for seq in sorted_seqs:
            data = self.seq_to_data[seq]
            cur_data_len += len(data)
            splits[cur_split].append(seq)
            if cur_data_len >= split_fracs[cur_split]*tot_data:
                cur_split = next(split_iter)
                cur_data_len = 0

        # now we have our sequences
        self.wt_sequences = splits[self.split]

    def __len__(self):
        return len(self.wt_sequences)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()

        seq = self.wt_sequences[index]
        data = self.seq_to_data[seq]

        pdb_file = f"data/v1_dataset_11152022/pdbs/{data.pdb_id_corrected[0]}.pdb1"
        pdb = parse_pdb_cached(self.cfg, pdb_file)

        print(pdb[0]['seq'])
        print(seq)

        assert len(pdb[0]['seq']) == len(seq)

        mutations = []
        for i, row in data.iterrows():
            mut = Mutation(row.position_corrected, row.mutation, row.ddG, row.dTm)
            mutations.append(mut)

        return pdb, mutations
    