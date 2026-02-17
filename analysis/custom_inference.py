import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from Bio.PDB import PDBParser

import sys
sys.path.append('../')
from datasets import Mutation
from train_thermompnn import TransferModelPL
from protein_mpnn_utils import tied_featurize, alt_parse_PDB
from thermompnn_benchmarking import get_trained_model
from SSM import get_ssm_mutations


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

ABPATH = os.path.dirname(os.path.abspath(__file__))

def get_chains(pdb):
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb)
  chains = [c.id for c in structure.get_chains()]
  return chains


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

    cfg = OmegaConf.load(os.path.join(ABPATH, "..", "local.yaml"))
    
    # Convert config dict to OmegaConf object and merge with local.yaml
    config_omega = OmegaConf.create(config)
    merged_config = OmegaConf.merge(cfg, config_omega)

    # load the chosen model and dataset
    models = {
        "ThermoMPNN": get_trained_model(model_name=args.model_path,
                                        config=merged_config, override_custom=True)
    }

    input_pdb = args.pdb
    pdb_id = os.path.splitext(os.path.basename(input_pdb))[0]

    datasets = {
        pdb_id: args.pdb
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation',])
    row = 0
    for name, model in models.items():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(device)
        for dataset_name, dataset in datasets.items():
            if len(args.chain) < 1:  # if unspecified, take first chain
                chain = get_chains(input_pdb)[0]
            else:
                chain = args.chain
            mut_pdb = alt_parse_PDB(input_pdb, chain)
            mutation_list = get_ssm_mutations(mut_pdb[0])
            final_mutation_list = []

            # build into list of Mutation objects
            for n, m in enumerate(mutation_list):
                if m is None:
                    final_mutation_list.append(None)
                    continue
                m = m.strip()  # clear whitespace
                wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])

                assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA,
                                        ddG=None, pdb=mut_pdb[0]['name'])
                final_mutation_list.append(mutation_obj)

            pred, _ = model(mut_pdb, final_mutation_list)

            for mut, out in zip(final_mutation_list, pred):
                if mut is not None:
                    col_list = ['ddG_pred', 'position', 'wildtype', 'mutation', 'pdb', 'chain']
                    val_list = [out["ddG"].cpu().item(), mut.position, mut.wildtype,
                                mut.mutation, mut.pdb.strip('.pdb'), chain]
                    for col, val in zip(col_list, val_list):
                        raw_pred_df.loc[row, col] = val

                    raw_pred_df.loc[row, 'Model'] = name
                    raw_pred_df.loc[row, 'Dataset'] = dataset_name
                    row += 1

    print(raw_pred_df)
    if args.out_dir == './':
        args.out_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    csv_file = os.path.join(args.out_dir, "ThermoMPNN_inference_%s.csv" % pdb_id)
    raw_pred_df.to_csv(csv_file)
    print(f'Saved ThermoMPNN output to {csv_file}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='', help='Input PDB to use for custom inference')
    parser.add_argument('--chain', type=str, default='A', help='Chain in input PDB to use.')
    parser.add_argument('--model_path', type=str, default='', help='filepath to model to use for inference')
    parser.add_argument('--out_dir', type=str, default='./', help='Output directory in which to save predictions.')

    args = parser.parse_args()
    if args.model_path in ["", None]:
        candidates = [
            os.path.join(ABPATH, "..", "models", "thermoMPNN_default.ckpt"),
            os.path.join(ABPATH, "..", "models", "thermoMPNN_default.pt"),
        ]
        # also scan common folders for any ckpt
        for folder in ["models", "vanilla_model_weights"]:
            folder_path = os.path.join(ABPATH, "..", folder)
            if os.path.isdir(folder_path):
                for fn in os.listdir(folder_path):
                    if fn.endswith(".ckpt"):
                        candidates.insert(0, os.path.join(folder_path, fn))

        for c in candidates:
            if os.path.isfile(c):
                args.model_path = c
                break

        assert args.model_path not in ["", None], "No checkpoint found. Please pass --model_path to a .ckpt file."

    # load config relative to this script location
    cfg = OmegaConf.load(os.path.join(ABPATH, "..", "local.yaml"))

    with torch.no_grad():
        main(cfg, args)
