from glob import glob
import os
import wandb
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef
import torch.nn.functional as F
from omegaconf import OmegaConf

import torch.nn as nn
from torch.utils.data import DataLoader
from cache import cache
from combo_dataset import ComboDataset
from mega_scale_dataset import MegaScaleDataset
from protein_mpnn_utils import loss_smoothed, tied_featurize
from train import TransferModelPL
from training.model_utils import featurize

from transfer_model import TransferModel, get_protein_mpnn
from fireprot_dataset import FireProtDataset

def get_weight_artifact(run, tag="latest"):
    api = wandb.Api()
    return api.artifact(f"{run.project}/model-{run.id}:latest", type='model')

def get_old_model(cfg, run, tag="latest"):
    artifact = get_weight_artifact(run, tag)
    artifact_dir = f"artifacts/model-{run.id}:{artifact.version}"
    if not os.path.exists(artifact_dir):
        assert os.path.normpath(artifact_dir) == os.path.normpath(artifact.download())
    checkpoint_file = artifact_dir + "/model.ckpt"
    return TransferModelPL.load_from_checkpoint(checkpoint_file, cfg=cfg).model

def get_old_model_by_name(cfg, name):
    api = wandb.Api()
    runs = api.runs(path=cfg.project, filters={"display_name": name})
    assert len(runs) == 1
    return get_old_model(cfg, runs[0])

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
class ProteinMPNNBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn(cfg)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device
        if tied_feat:
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
        else:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize([pdb], device)
        # all_mpnn_hid, mpnn_embed = self.prot_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None)
        *_, log_probs = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        out = []
        for mut in mutations:
            # hacky fix to account for deletions (which we don't support atm)
            if mut is None:
                out.append(None)
                continue

            aa_index = alphabet.index(mut.mutation)
            pred = log_probs[0, mut.position, aa_index]

            out.append({
                "ddG": -torch.unsqueeze(pred, 0),
                "dTm": torch.unsqueeze(pred, 0)
            })
        return out, log_probs

def get_metrics():
    return {
        "r2": R2Score(),
        # "mse": MeanSquaredError(),
        "spearman": SpearmanCorrCoef(),
    }

def main(cfg):
    models = {
        "Rocklin (ML)": get_old_model_by_name(cfg, "rocklin_ml"),
        "Combo 0.9 - 0.1": get_old_model_by_name(cfg, "combo_0.9_0.1"),
        "Combo 0.5 - 0.5": get_old_model_by_name(cfg, "combo_0.5_0.5"),
        "Combo 0.1 - 0.9": get_old_model_by_name(cfg, "combo_0.1_0.9"),
        "FireProt": TransferModelPL.load_from_checkpoint("data/fireprot.ckpt", cfg=cfg).model,
        "Rocklin": TransferModelPL.load_from_checkpoint("data/rocklin.ckpt", cfg=cfg).model,
        'ProteinMPNN': ProteinMPNNBaseline()
    }
    datasets = {
        "Rocklin (val)": MegaScaleDataset(cfg, "val"),
        "FireProt (val)": FireProtDataset(cfg, "val"),
        # "Rocklin (train)": MegaScaleDataset(cfg, "train"),
        # "FireProt (train)": FireProtDataset(cfg, "train")
    }

    max_batches = None
    results = []
    for name, model in models.items():
        model = model.eval()
        for dataset_name, dataset in datasets.items():
            metrics = {
                "ddG": get_metrics(),
                "dTm": get_metrics()
            }
            for i, batch in enumerate(tqdm(dataset)):
                (mut_pdb, mutations) = batch
                pred, _ = model(mut_pdb, mutations)
                for mut, out in zip(mutations, pred):
                    if mut.ddG is not None:
                        for metric in metrics["ddG"].values():
                            metric.update(out["ddG"], mut.ddG)
                    if mut.dTm is not None:
                        for metric in metrics["dTm"].values():
                            metric.update(out["dTm"], mut.dTm)
                if max_batches is not None and i >= max_batches:
                    break
            column = {
                "Model": name,
                "Dataset": dataset_name,
            }
            for dtype in [ "ddG", "dTm" ]:
                for met_name, metric in metrics[dtype].items():
                    try:
                        column[f"{dtype} {met_name}"] =  metric.compute().cpu().item()
                    except ValueError:
                        pass
            results.append(column)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("data/dataset_metrics.csv")
    # df.to_clipboard()

    rows = []
    for model in ["FireProt", "Rocklin", "Rocklin (ML)", "ProteinMPNN", "Combo 0.9 - 0.1", "Combo 0.5 - 0.5", "Combo 0.1 - 0.9"]:
        cur = { "model": model }
        for metric in [ "ddG spearman", "dTm spearman" ]:
            for dataset in [ "Rocklin (val)", "FireProt (val)" ]:
                cur_rows = df.query("Model == @model and Dataset == @dataset").reset_index(drop=True)
                assert len(cur_rows) == 1
                cur[dataset + " " + metric] = cur_rows[metric][0]
        rows.append(cur)        
    out_df = pd.DataFrame(rows)
    out_df.to_csv("data/dataset_metrics_out.csv")

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    with torch.no_grad():
        main(cfg)