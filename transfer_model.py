import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_mpnn_utils import ProteinMPNN, tied_featurize
from training.model_utils import featurize

def get_protein_mpnn(cfg):

    hidden_dim = 128
    num_layers = 3 

    checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if cfg.model.freeze_weights:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model

HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True

class TransferModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.use_msa = cfg.model.use_msa
        self.prot_mpnn = get_protein_mpnn(cfg)

        extra_size = len(alphabet) if self.use_msa else 0
        hid_sizes = [ HIDDEN_DIM*self.num_final_layers + EMBED_DIM  + extra_size]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        self.both_out = nn.Sequential()
        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)
        self.dtm_out = nn.Linear(1, 1)

        self.seq_out = nn.Linear(HIDDEN_DIM*self.num_final_layers, len(alphabet))

    def forward(self, pdb, mutations, tied_feat=True):
        
        device = next(self.parameters()).device
        if tied_feat:
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
        else:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize([pdb], device)
        # all_mpnn_hid, mpnn_embed = self.prot_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None)
        all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
        seq = self.seq_out(mpnn_hid)

        out = []
        for mut in mutations:
            # hacky fix to account for deletions (which we don't support atm)
            if mut is None:
                out.append(None)
                continue

            aa_index = alphabet.index(mut.mutation)
            wt_aa_index = alphabet.index(mut.wildtype)

            hid = mpnn_hid[0][mut.position]
            embed = mpnn_embed[0][mut.position]
            inputs = [hid, embed]
            if self.use_msa:
                inputs.append(mut.msa_hist)
            lin_input = torch.cat(inputs, -1)
            both_input = torch.unsqueeze(self.both_out(lin_input), -1)

            ddg_out = self.ddg_out(both_input)
            dTm_out = self.dtm_out(both_input)

            if self.subtract_mut:
                ddg = ddg_out[aa_index][0] - ddg_out[wt_aa_index][0]
                dtm = dTm_out[aa_index][0] - dTm_out[wt_aa_index][0]
            else:
                ddg = ddg_out[aa_index][0]
                dtm = dTm_out[aa_index][0]

            out.append({
                "ddG": torch.unsqueeze(ddg, 0),
                "dTm": torch.unsqueeze(dtm, 0)
            })

        return out, F.log_softmax(seq, dim=-1)