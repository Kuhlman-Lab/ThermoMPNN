import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
from itertools import combinations_with_replacement

import sys

sys.path.append('../')
from datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset, SSMDataset
from transfer_model import get_protein_mpnn
from train_thermompnn import TransferModelPL
from protein_mpnn_utils import tied_featurize

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20,
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

        num_neighbors[pairwise_comb] = torch.sum(pairwise_dists < radius, dim=-1) - 1

    return num_neighbors


class ProteinMPNNBaseline(nn.Module):
    """Class for running ProteinMPNN as a ddG proxy predictor"""

    def __init__(self, cfg, version='v_48_020.pt'):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn(cfg, version=version)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
            [pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        *_, log_probs = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        out = []
        for mut in mutations:
            if mut is None:
                out.append(None)
                continue

            aa_index = ALPHABET.index(mut.mutation)
            pred = log_probs[0, mut.position, aa_index]

            out.append({
                "ddG": torch.unsqueeze(pred, 0),
                "dTm": torch.unsqueeze(pred, 0)
            })
        return out, log_probs


def get_metrics():
    return {
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson": PearsonCorrCoef().to('cuda'),
    }


def get_trained_model(model_name, config, checkpt_dir='models/', override_custom=False):
    if override_custom:
        return TransferModelPL.load_from_checkpoint(model_name, cfg=config).model
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
        return TransferModelPL.load_from_checkpoint(model_loc, cfg=config).model


def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics(),
    }
    print('Testing Model %s on dataset %s' % (name, dataset_name))

    for i, batch in enumerate(tqdm(dataset)):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality_chains, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    cols = ['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
            'best_AA', 'pdb']

    if centrality:
        if centrality_chains is None:
            # Find all the chains in the dataset
            centrality_chains = []
            for i, batch in enumerate(dataset):
                mut_pdb, mutations = batch
                chains = [c[-1] for c in mut_pdb[0].keys() if f"coords_chain_" in c]
                centrality_chains.extend([chain for chain in chains if chain not in centrality_chains])

        else:
            chains = centrality_chains

        neighbor_cols = ["neighbors_" + "".join(i) for i in combinations_with_replacement(chains, 2)]
        cols.extend(neighbor_cols)

    raw_pred_df = pd.DataFrame(columns=cols)

    metrics = {
        "ddG": get_metrics(),
    }
    print('Running model %s on dataset %s' % (name, dataset_name))

    # append model and dataset columns
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    results.append(column)

    # Init csv file
    csv_fh = f"{name}_{dataset_name}_raw_preds.csv"
    raw_pred_df.to_csv(csv_fh, mode="w")

    for i, batch in enumerate(tqdm(dataset)):
        mut_pdb, mutations = batch
        pred, _ = model(mut_pdb, mutations)

        if centrality:
            neighbors = compute_centrality(mut_pdb[0], basis_atom='CA', radius=10., backup_atom='C', chains=chains)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

                # assign raw preds and useful details to df
                col_list = ['ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                val_list = [mut.ddG.cpu().item(), out["ddG"].cpu().item(), mut.position, mut.wildtype,
                            mut.mutation, mut.pdb.strip('.pdb')]
                for col, val in zip(col_list, val_list):
                    raw_pred_df.loc[row, col] = val

                if centrality:
                    combs = [comb for comb in neighbors.keys()]
                    for i, comb in enumerate(combs):
                        raw_pred_df.loc[row, neighbor_cols[i]] = neighbors[comb][mut.position].cpu().item()

            raw_pred_df.loc[row, 'Model'] = name
            raw_pred_df.loc[row, 'Dataset'] = dataset_name
            if 'Megascale' not in dataset_name:  # different pdb column formatting
                key = mut.pdb
            else:
                key = mut.pdb + '.pdb'
            if 'S669' not in dataset_name:  # S669 is missing WT seq info - omit to prevent error
                raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        raw_pred_df.to_csv(csv_fh, mode="a", header=False)
        raw_pred_df = raw_pred_df[0:0]

        if max_batches is not None and i >= max_batches:
            break

    for dtype in ["ddG"]:  # , "dTm"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                pass

    del raw_pred_df

    return results


def main(cfg, args):
    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    models = {
        'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        "ThermoMPNN": get_trained_model(model_name='thermoMPNN_default.pt',
                                        config=cfg)

    }

    misc_data_loc = '/nas/longleaf/home/dieckhau/protein-stability/enzyme-stability/data'
    datasets = {
        # "Megascale-test": MegaScaleDataset(cfg, "test"),
        # "Fireprot-test": FireProtDataset(cfg, "test"),
        "SSM": SSMDataset(cfg, "val")
        # "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free"),
        # "P53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv')),
        # "MYOGLOBIN": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')),
        #
        # "SSYM_dir": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')),
        # "SSYM_inv": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')),
        # "S669": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'S669/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'S669/s669_clean_dir.csv')),
    }

    results = []

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                if args.centrality_chains is not None:
                    centrality_chains = args.centrality_chains.split(",")
                else:
                    centrality_chains = args.centrality_chains
                results = run_prediction_keep_preds(name, model, dataset_name, dataset, results,
                                                    centrality_chains, centrality=args.centrality)
            else:
                results = run_prediction_default(name, model, dataset_name, dataset, results)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("ThermoMPNN_metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_preds', action='store_true', default=False, help='Save raw model predictions as csv')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Use with --centrality_chains to specify specific chains to use. '
                             'If --centrality_chains is not specified, '
                             'script will calculate centrality between and within all chains. '
                             'Only used if --keep_preds is enabled.')
    parser.add_argument('--centrality_chains',
                        help="Use with --centrality to specify chains to calculate centrality for. "
                             'Input is a comma-delimited string, like so: --centrality_chains "A,B" [other arguments]')

    args = parser.parse_args()
    cfg = OmegaConf.load("../local.yaml")
    with torch.no_grad():
        main(cfg, args)
