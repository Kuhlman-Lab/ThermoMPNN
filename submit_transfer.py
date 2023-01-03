from glob import glob
import torch
from omegaconf import OmegaConf
import pandas as pd
from train import TransferModelPL
from fireprot_dataset import Mutation, get_msa_hist
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB

alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
def submit(cfg):
    model = TransferModelPL.load_from_checkpoint(glob("checkpoints/*.ckpt")[-1], cfg=cfg).model

    pdb = parse_PDB("data/wildtype_structure_prediction_af2.pdb")
    wt_seq = pdb[0]['seq']
    mutations = []
    df = pd.read_csv("data/test.csv")

    msa_file = "data/msas/kaggle.a3m"
    all_msa_hist, msa_seq = get_msa_hist(cfg, msa_file)
    assert msa_seq == wt_seq

    for i, row in df.iterrows():
        if len(row.protein_sequence) < len(wt_seq):
            mutations.append(None)
            continue # ignore deletions for now
        eq = [ c1 != c2 for c1, c2 in zip(row.protein_sequence, wt_seq)]
        if sum(eq) == 0:
            mutations.append(None)
            continue # we found the wt sequence
        assert sum(eq) == 1
        idx = eq.index(True)
        wt_aa = wt_seq[idx]
        mut_aa = row.protein_sequence[idx]
        msa_hist = all_msa_hist[idx]
        mutation = Mutation(position=idx, wildtype=wt_aa, mutation=mut_aa, msa_hist=msa_hist)
        mutations.append(mutation)

    # preds = []
    # for mut in mutations:
    #     if mut is None:
    #         preds.append(0.0)
    #         continue
    #     mut_aa_idx = alphabet.index(mut.mutation)
    #     prob = float(mut.msa_hist[mut_aa_idx])
    #     print(prob)
    #     preds.append(prob)

    with torch.no_grad():
        preds = [ out['ddG'].cpu().item() if out is not None else -100 for out in model(pdb, mutations)[0] ]

    df['dTm'] = preds
    new_df = pd.DataFrame({'seq_id': df.seq_id, 'tm': df.dTm })
    new_df.to_csv("data/submission.csv", index=False)

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    submit(cfg)