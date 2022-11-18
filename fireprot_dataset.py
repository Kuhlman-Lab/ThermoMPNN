from dataclasses import dataclass
import os
import pickle
from typing import Optional, Sequence
import torch
import pandas as pd
from Bio import pairwise2
from terrace.batch import Batchable, make_batch

from cache import cache
from protein_mpnn_utils import parse_PDB

@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    return parse_PDB(pdb_file)

@dataclass
class Mutation(Batchable):
    position: int
    mutation: str
    ddG: Optional[float] = None
    dTm: Optional[float] = None

def seq1_index_to_seq2_index(align, index):
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != '-':
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break
    
    # now the index in seq 2 cooresponding to aligned index
    if align.seqB[aln_idx] == '-':
        return None

    # print(pairwise2.format_alignment(*align))

    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == '-':
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx

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
        # if index >= len(self):
        #    raise StopIteration()

        seq = self.wt_sequences[index]
        data = self.seq_to_data[seq]

        pdb_file = f"data/v1_dataset_11152022/pdbs/{data.pdb_id_corrected[0]}.pdb1"
        pdb = parse_pdb_cached(self.cfg, pdb_file)

        # assert len(pdb[0]['seq']) == len(seq)
        align, *rest = pairwise2.align.globalxx(seq, pdb[0]['seq'].replace("-", "X"))

        mutations = []
        for i, row in data.iterrows():
            pdb_idx = seq1_index_to_seq2_index(align, row.position_corrected)
            # we don't have crystal data for this mutation alas
            if pdb_idx is None:
                continue
            assert seq[row.position_corrected] == row.wild_type
            # print(seq)
            # print(pdb[0]['seq'])
            # print(row.position_corrected, pdb_idx)
            # print(row.wild_type, pdb[0]['seq'][pdb_idx])

            assert pdb[0]['seq'][pdb_idx] == row.wild_type
            ddG = None if row.ddG is None else torch.tensor([row.ddG], dtype=torch.float32)
            dTm = None if row.dTm is None else torch.tensor([row.dTm], dtype=torch.float32)
            mut = Mutation(pdb_idx, row.mutation, ddG, dTm)
            mutations.append(mut)

        return pdb, mutations
    