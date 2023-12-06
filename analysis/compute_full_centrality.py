import pandas as pd
from tqdm import tqdm 
import os
import torch

import sys
sys.path.append('../')
from omegaconf import OmegaConf
from datasets import SSMDataset, MegaScaleDataset, FireProtDataset, ddgBenchDataset, Mutation
from itertools import combinations_with_replacement

def full_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20,
                            surface_threshold: int = 15, backup_atom: str = "C", chains: list = ['A', 'B']):
    coords = {}
    for chain in chains:
        coord_chain = [c for c in xyz.keys() if f"coords_chain_{chain}" in c][0]
        coord_chain = xyz[coord_chain][basis_atom + f"_chain_{chain}"]
        coords[chain] = torch.tensor(coord_chain)

    pairwise_combs = ["".join(i) for i in combinations_with_replacement(chains, 2)]

    num_neighbors = {}

    for pairwise_comb in pairwise_combs:
        coords_ch1 = coords[pairwise_comb[0]]
        coords_ch2 = coords[pairwise_comb[1]]

        pairwise_dists = torch.cdist(coords_ch1, coords_ch2)
        pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)

        # subtract 1 from n_neighbors if computing within same chain
        if pairwise_comb[0] == pairwise_comb[1]:
            num_neighbors[pairwise_comb] = torch.sum(pairwise_dists < radius, dim=-1) - 1
        else:
            num_neighbors[pairwise_comb] = torch.sum(pairwise_dists < radius, dim=-1)

    return num_neighbors

    
def main(cfg):
    datasets = {
        "SSM": SSMDataset(cfg, "all")
    }
    
    # Get all chains in the dataset
    chainlist = []
    for dataset_name, dataset in datasets.items():
        # Compute the number of neighbors
        print("Computing neighbors for all chains...")
        
        cols = ['dataset', 'pdb', 'seq', 'chain', 'position', 'residue', 'comparison_chain', 'n_neighbors']
        raw_df = pd.DataFrame(columns=cols)
        
        for i, batch in enumerate(tqdm(dataset)):
            xyz, mutations = batch
            xyz = xyz[0]
            chains = [c[-1] for c in xyz.keys() if f"coords_chain_" in c]
    
            neighbors = full_centrality(xyz, chains=chains)
            
            for comb in neighbors:
                for j, neigbor_calc in enumerate(neighbors[comb]):
                    outrow = [dataset_name, xyz["name"],xyz[f"seq_chain_{comb[-2]}"], comb[-2], 
                              j+1, xyz[f"seq_chain_{comb[-2]}"][j], comb[-1], neigbor_calc.cpu().item()]
                    raw_df.loc[len(raw_df.index)] = outrow
                    
        raw_df.to_csv(f"{dataset_name}_full_centrality.csv")
                    

if __name__ == "__main__":
    cfg = OmegaConf.load("../local.yaml")
    main(cfg)