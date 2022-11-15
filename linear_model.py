import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import xgboost as xgb

from cache import cache
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
from kaggle_dataset import KaggleTrainDataset

device = 'cuda:0'

def get_model():

    hidden_dim = 128
    num_layers = 3 

    checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # freeze these weights for transfer learning
    for param in model.parameters():
        param.requires_grad = False

    return model

def move_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)([ move_to(item, device) for item in obj])
    else:
        return obj

def dataset_key(cfg, model, dataset):
    return dataset.split

@cache(dataset_key, disable=False)
def get_preds(cfg, model, dataset):
    
    wt_preds = []
    mut_preds = []

    for i, x in enumerate(tqdm(dataset)):
        if x is None:
            wt_preds.append(None)
            mut_preds.append(None)
            continue
        wt_feat, mut_feat, dTm, position = x
        wt_feat = move_to(wt_feat, 'cuda:0')
        mut_feat = move_to(mut_feat, 'cuda:0')
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = wt_feat
        wt_preds.append(model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None))

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = mut_feat
        mut_preds.append(model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None))

        # if i > 10:
        #    return wt_preds, mut_preds

    return wt_preds, mut_preds

def encode_aa(aa):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    idx = alphabet.index(aa)
    one_hot = torch.zeros((len(alphabet)), device='cuda:0')
    one_hot[idx] = 1
    return one_hot

@cache(dataset_key, disable=False)
def get_x_and_y(cfg, model, dataset):
    df = dataset.df
    wt_preds, mut_preds = get_preds(cfg, model, dataset)

    x = []
    y = []

    for wt_pred, mut_pred, wt, mut, dTm, ddG, pos in zip(tqdm(wt_preds), mut_preds, df.wildtype, df.mutation, df.dTm, df.ddG, df.position):
        if wt_pred is None: continue
        wt_aa = encode_aa(wt)
        mut_aa = encode_aa(mut)
        wt_data = wt_pred[0][int(pos)-1]
        mut_data = mut_pred[0][int(pos)-1]
        x_data = torch.cat((wt_data, mut_data, wt_aa, mut_aa))
        x.append(x_data)
        dTm = torch.tensor([dTm], dtype=float, device='cuda:0')
        ddG = torch.tensor([ddG], dtype=float, device='cuda:0')
        if torch.isnan(dTm)[0]:
            assert not torch.isnan(ddG)[0]
            y.append(ddG)
        else:
            assert not torch.isnan(dTm)[0]
            assert torch.isnan(ddG)[0]
            y.append(dTm)

    return torch.stack(x).cpu().numpy(), torch.stack(y).cpu().numpy()

@cache(lambda c, x, y: "", disable=True)
def get_linear_model(cfg, x, y):
    linear_model = RandomForestRegressor(500)
    linear_model.fit(x, np.ravel(y))
    # linear_model = Ridge()
    # linear_model.fit(x, y)
    return linear_model

@cache(lambda c, x, y: "", disable=False)
def get_xg_model(cfg, x, y):
    # XGB MODEL PARAMETERS
    xgb_parms = { 
        'max_depth':4, 
        'learning_rate':0.001, 
        'subsample':0.6,
        'colsample_bytree':0.2, 
        'eval_metric':'rmse',
        'objective':'reg:squarederror',
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'random_state': 50
    }
    dtrain = xgb.DMatrix(data=x, label=y)
    model = xgb.train(xgb_parms, 
            dtrain=dtrain,
            evals=[(dtrain,'train')],
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=100)

    return model

def train(cfg):

    model = get_model()
    dataset = KaggleTrainDataset(cfg, "train")
    x, y = get_x_and_y(cfg, model, dataset)
    linear_model = get_xg_model(cfg, x, y)

    for split in ("train", "val"):
        dataset = KaggleTrainDataset(cfg, split)
        x, y = get_x_and_y(cfg, model, dataset)
        y_pred = linear_model.predict(xgb.DMatrix(x))
        r2 = r2_score(y, y_pred)
        spearman, pval = spearmanr(y, y_pred)
        print(f"{split} {r2=:.2f}, {spearman=:.2f}")

    return linear_model

@cache(lambda c: "", disable=False)
def get_test_x(cfg):

    prot_mpnn = get_model()

    df = pd.read_csv("data/test.csv")
    pdb = parse_PDB("data/wildtype_structure_prediction_af2.pdb")
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
    wt_pred = prot_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None)

    wt = pdb[0]
    wt_seq = wt['seq']
    x = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if len(row.protein_sequence) < len(wt_seq):
            continue # ignore deletions for now
        eq = [ c1 != c2 for c1, c2 in zip(row.protein_sequence, wt_seq)]
        if sum(eq) == 0:
            continue # we found the wt sequence
        assert sum(eq) == 1

        idx = eq.index(True)
        aa_idx = alphabet.index(row.protein_sequence[idx])

        wt['seq'] = row.protein_sequence
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
        mut_pred = prot_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None)
        
        wt_aa = encode_aa(wt_seq[idx])
        mut_aa = encode_aa(row.protein_sequence[idx])
        mut_data = mut_pred[0][idx]
        wt_data = wt_pred[0][idx]
        x.append(torch.cat((wt_data, mut_data, wt_aa, mut_aa)))

    return torch.stack(x).cpu().numpy()

def predict(cfg, lin_model):

    df = pd.read_csv("data/test.csv")
    xs = get_test_x(cfg)

    pdb = parse_PDB("data/wildtype_structure_prediction_af2.pdb")
    wt = pdb[0]
    wt_seq = wt['seq']

    scores = []
    idx = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if len(row.protein_sequence) < len(wt_seq):
            scores.append(-100)
            continue # ignore deletions for now
        eq = [ c1 != c2 for c1, c2 in zip(row.protein_sequence, wt_seq)]
        if sum(eq) == 0:
            scores.append(0)
            continue # we found the wt sequence
        assert sum(eq) == 1

        x = xs[idx:idx+1]
        score = lin_model.predict(xgb.DMatrix(x))[0]# -lin_model.predict(x)[0]
        scores.append(score)
        idx += 1

    df['score'] = scores
    new_df = pd.DataFrame({'seq_id': df.seq_id, 'tm': df.score })
    new_df.to_csv("data/submission.csv", index=False)

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    with torch.no_grad():
        lin_model = train(cfg)
        predict(cfg, lin_model)