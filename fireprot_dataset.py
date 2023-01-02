from dataclasses import dataclass
from collections import defaultdict
import os
import pickle
from typing import Optional, Sequence
import torch
import pandas as pd
from Bio import pairwise2

from cache import cache
from protein_mpnn_utils import parse_PDB

@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    return parse_PDB(pdb_file)

@dataclass
class Mutation:
    position: int
    wildtype: str
    mutation: str
    msa_hist: torch.Tensor
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

alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
@cache(lambda cfg, msa_file: msa_file, disable=False, version=3.0)
def get_msa_hist(cfg, msa_file):
    first_seq = None
    counts = None
    with open(msa_file, "r") as f:
        for line in f.readlines():
            if line.startswith(">"): continue
            if counts is None:
                first_seq = line.strip()
                counts = [ defaultdict(int) for c in line ]
            i = 0
            for c in line.upper():
                if c.islower():
                    continue
                if i > len(counts) - 1:
                    break
                counts[i][c] += 1
                i += 1
    ret = torch.zeros((len(counts), len(alphabet)), dtype=torch.float32)
    for i, count_dict in enumerate(counts):
        for j, c in enumerate(alphabet):
            ret[i][j] = count_dict[c]
    # print("!")
    return ret/ret.sum(-1).unsqueeze(-1), first_seq

class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, version=2):

        self.cfg = cfg
        self.split = split
        self.version = version

        if version == 1:
            filename = "data/v1_dataset_11152022/3_fireprotDB_curated_v1_bestpH_avgDupes.csv"
        elif version == 2:
            # filename = "data/v2_dataset_12072022/fireprotDB_cleaned_monomers_allmutations.csv"
            filename = "data/v2_monomers_allmutations_oldAlignment_wmetadata_bestpH.csv"
        else:
            raise AssertionError()
        df = pd.read_csv(filename)
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}

        seq_key = "sequence_corrected"
        # seq_key = "sequence" if version != 1 else "sequence_corrected"
        # group all the data by wildtype seq
        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)
        
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

        if self.version == 1:
            pdb_file = f"data/v1_dataset_11152022/pdbs/{data.pdb_id_corrected[0]}.pdb1"
        else:
            pdb_file = f"data/v2_dataset_12072022/pdbs/monomers/{data.pdb_id_corrected[0]}.pdb"
        pdb = parse_pdb_cached(self.cfg, pdb_file)

        # if self.version == 1:
        align, *rest = pairwise2.align.globalxx(seq, pdb[0]['seq'].replace("-", "X"))

        msa_align = None
        msa_seq = None
        try:
            all_msa_hist, msa_seq = get_msa_hist(self.cfg, f"data/msas/{data.uniprot_id[0]}.a3m")
            msa_align, *rest = pairwise2.align.globalxx(seq, msa_seq)
        except FileNotFoundError:
            pass
            # print(f"No msa for {data.uniprot_id[0]} ({len(data)}), skipping")
            # return pdb, []

        mutations = []
        for i, row in data.iterrows():
            # print(i)
            if self.version == 1:
                pdb_idx = seq1_index_to_seq2_index(align, row.position_corrected)
                # we don't have crystal data for this mutation alas
                if pdb_idx is None:
                    continue
            else:
                pdb_idx = seq1_index_to_seq2_index(align, row.position_corrected)
                # we don't have crystal data for this mutation alas
                if pdb_idx is None:
                    continue

            msa_idx = None
            if msa_align is not None:
                msa_idx = seq1_index_to_seq2_index(msa_align, row.position_corrected)

            msa_hist = torch.zeros((len(alphabet,)))
            if msa_idx is not None:
                assert msa_seq[msa_idx] == row.wild_type
                msa_hist = all_msa_hist[msa_idx]

            assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.sequence_corrected[row.position_corrected]

            ddG = None if row.ddG is None else torch.tensor([row.ddG], dtype=torch.float32)
            dTm = None if row.dTm is None else torch.tensor([row.dTm], dtype=torch.float32)
            mut = Mutation(pdb_idx, pdb[0]['seq'][pdb_idx], row.mutation, msa_hist, ddG, dTm)
            mutations.append(mut)

            # if self.split == "train":
            #     zero = torch.tensor([0.0], dtype=torch.float32)
            #     wt = Mutation(pdb_idx, row.wild_type, row.wild_type, zero, zero)
            #     mutations.append(wt)

        return pdb, mutations
    