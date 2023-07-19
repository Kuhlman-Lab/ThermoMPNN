import os
import pandas as pd
from tqdm import tqdm
import torch
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf
from Bio import pairwise2

import sys
sys.path.append('../')
from datasets import seq1_index_to_seq2_index, FireProtDataset, MegaScaleDataset, ddgBenchDataset

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
        "pearson": PearsonCorrCoef(),
    }


def get_rosetta_data(data_loc, dataset_name):
    if 'Megascale' in dataset_name:

        df1 = pd.read_csv(os.path.join(data_loc, 'ROCKLIN_ROSETTA_RESULTS_ALL.csv'))
        df2 = pd.read_csv(os.path.join(data_loc, 'ROCKLIN_ROSETTA_RESULTS_EXTRA.csv'))
        dftot = pd.concat([df1, df2])
    else:
        df1 = pd.read_csv(os.path.join(data_loc, 'FIREPROT_ROSETTA_RESULTS_EXTRA.csv'))
        dftot = df1

    return dftot


def match_rosetta_pred(mut_pdb, mutations, dftot, dataset):
    # match first by PDB ID (Rocklin or Fireprot)
    if mutations[0].pdb == '1BZ6':
        checker = '1BVC'
    else:
        checker = mutations[0].pdb
    m_slice = dftot.loc[dftot['PDB_ID'] == checker, :]

    variants = m_slice['Mutation Name'].unique()
    wt_res = [v[2:-2] for v in variants]

    unique_res = []
    for wtr in wt_res:
        if wtr not in unique_res:
            unique_res.append(wtr)
    unique_res = [u[0] for u in unique_res]
    seq = ''.join(unique_res)
    try:
        first_res = int(wt_res[0][1:])
    except IndexError:
        print('Missing data for protein %s, skipping for now' % mutations[0].pdb)
        quit()
        return [None] * len(mutations)

    preds = []
    for m in mutations:
        try:
            pdb_idx = m.position
            # check to make sure WT residue is correct (alignment matches)
            assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['Position'] == pdb_idx + 1, :]['WT Residue'].unique()[0]
            spec = m_slice.loc[m_slice['Position'] == pdb_idx + 1, :]

        except (IndexError, AssertionError):
            align, *rest = pairwise2.align.globalxx(mut_pdb[0]['seq'].replace("-", "X"), seq)
            new_pdb_idx = seq1_index_to_seq2_index(align, m.position)
            if new_pdb_idx is None:
                print('Alignment failed - skipping mutation')
                preds.append(None)
                continue

            m_slice['Position'] = m_slice['Position'].astype(str).replace('A', '', regex=True)
            try:
                assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx, :]['WT Residue'].unique()[0]
                spec = m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx, :]

            except (AssertionError, IndexError):
                try:
                    assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res, :]['WT Residue'].unique()[0]
                    spec = m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res, :]

                except (AssertionError, IndexError):
                    try:
                        assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res - 1, :]['WT Residue'].unique()[0]
                        spec = m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res + 1, :]

                    except (AssertionError, IndexError):
                        gaps = [a for a in mut_pdb[0]['seq'][0:pdb_idx ] if a == '-']
                        gaps = len(gaps)
                        if m.pdb == '1HGU':
                            new_pdb_idx += 3
                        new_pdb_idx += gaps
                        assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res, :]['WT Residue'].unique()[0]
                        spec = m_slice.loc[m_slice['Position'].astype(int) == new_pdb_idx + first_res, :]

        # grab specific data point for the mutant of interest
        spec = spec.loc[spec['Mutant Residue'] == m.mutation, :]
        if spec.shape[0] == 0:
            print('Residue matching failed - skipping mutation')
            quit()
        preds.append(spec)
    return preds


def get_RaSP_data(data_loc, dataset):

    if 'Megascale' in dataset:
        rasp_dir = os.path.join(data_loc, 'megascale')
    else:
        rasp_dir = os.path.join(data_loc, 'fireprot')

    all_csv = sorted(os.listdir(rasp_dir))
    df_list = []

    for csv in all_csv:
        df = pd.read_csv(os.path.join(rasp_dir, csv))
        df['PDB_ID'] = '_'.join(csv.split('.')[0].split('_')[1:-1])
        df_list.append(df)

    dftot = pd.concat(df_list)
    return dftot


def match_RaSP_pred(mut_pdb, mutations, dftot):

    # match first by PDB ID
    if mutations[0].pdb == '1BZ6':  # same protein, different PDBs
        checker = '1BVC'
    else:
        checker = mutations[0].pdb
    m_slice = dftot.loc[dftot['PDB_ID'] == checker, :]
    variants = m_slice.variant.unique()
    wt_res = [v[0:-1] for v in variants]
    unique_res = []
    for wtr in wt_res:
        if wtr not in unique_res:
            unique_res.append(wtr)

    unique_res = [u[0] for u in unique_res]
    seq = ''.join(unique_res)
    preds = []

    for m in mutations:
        try:
            pdb_idx = m.position
            assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['pos'] == pdb_idx + 1, :].wt_AA.unique()[0]
            spec = m_slice.loc[m_slice['pos'] == pdb_idx + 1, :]

        except (AssertionError, IndexError):

                align, *rest = pairwise2.align.globalxx(mut_pdb[0]['seq'].replace("-", "X"), seq)
                new_pdb_idx = seq1_index_to_seq2_index(align, m.position)
                if new_pdb_idx is None:
                    preds.append(None)
                    continue
                # print('NEW IDX:', pdb_idx, new_pdb_idx)
                # print(mut_pdb[0]['seq'][pdb_idx], m.wildtype, m_slice.loc[m_slice['pos'] == new_pdb_idx + 1, :].wt_AA.unique()[0])
                try:
                    assert mut_pdb[0]['seq'][pdb_idx] == m.wildtype == m_slice.loc[m_slice['pos'] == new_pdb_idx + 1, :].wt_AA.unique()[0]
                    spec = m_slice.loc[m_slice['pos'] == new_pdb_idx + 1, :]
                    # print(spec['wt_AA'].unique(), pdb_idx, m.position)
                except IndexError:
                    preds.append(None)
                    continue

        # check to make sure WT residue is correct (alignment matches)
        assert spec['wt_AA'].unique()[0] == m.wildtype

        # grab specific data point for the mutant of interest
        spec = spec.loc[spec['mt_AA'] == m.mutation, :]
        # print('SPEC:', spec)
        preds.append(spec)

    return preds


def main(cfg):

    # NOTE: include either RaSP or Rosetta for a single run - not both
    # rosetta_data_loc = '../../scores/Rosetta'
    # rosetta_dataset = 'Megascale'
    # rosetta_dataset = 'Fireprot'
    # models = {"Rosetta": get_rosetta_data(rosetta_data_loc, rosetta_dataset)}

    rasp_data_loc = '../../scores/RaSP'
    # rasp_dataset = 'Megascale'
    rasp_dataset = 'Fireprot'
    models = {"RaSP": get_RaSP_data(rasp_data_loc, rasp_dataset)}

    # datasets = {"Megascale-test": MegaScaleDataset(cfg, "test")}
    datasets = {"Fireprot-HF": FireProtDataset(cfg, "homologue-free")}

    max_batches = None
    results = []
    for name, model in models.items():
        for dataset_name, dataset in datasets.items():
            metrics = {
                "ddG": get_metrics(),
            }
            for i, batch in enumerate(tqdm(dataset)):
                mut_pdb, mutations = batch
                if 'Rosetta' in name:
                    pred = match_rosetta_pred(mut_pdb, mutations, model, rosetta_dataset)
                else:
                    pred = match_RaSP_pred(mut_pdb, mutations, model)

                for mut, out in zip(mutations, pred):
                    if mut.ddG is not None and out is not None:
                        for metric in metrics["ddG"].values():
                            if 'Rosetta' in name:
                                metric.update(torch.tensor(out["Delta Energy (mutant-WT)"].values), mut.ddG)
                            else:
                                # Fermi transformed scores performed better in most cases
                                metric.update(torch.tensor(out["score_ml_fermi"].values), mut.ddG)
                                # metric.update(torch.tensor(out["score_ml"].values), mut.ddG)

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
                    except ValueError:
                        pass
            results.append(column)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("metrics.csv")


if __name__ == "__main__":
    cfg = OmegaConf.load("../local.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    main(cfg)
