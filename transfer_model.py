import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, tied_featurize

def get_protein_mpnn():

    hidden_dim = 128
    num_layers = 3 

    checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # freeze these weights for transfer learning
    for param in model.parameters():
        param.requires_grad = False

    return model

HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
class TransferModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn()
        self.ddg_out = nn.Linear(HIDDEN_DIM+EMBED_DIM, VOCAB_DIM)
        self.dtm_out = nn.Linear(HIDDEN_DIM+EMBED_DIM, VOCAB_DIM)

    def forward(self, pdb, mutations):
        device = next(self.parameters()).device
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], device, None, None, None, None, None, None, ca_only=False)
        mpnn_hid, mpnn_embed = self.prot_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, None)
        out = []
        for mut in mutations:
            aa_index = alphabet.index(mut.mutation)
            hid = mpnn_hid[0][mut.position]
            embed = mpnn_embed[0][mut.position]
            lin_input = torch.cat([hid, embed], -1)
            out.append({
                "ddG": torch.unsqueeze(self.ddg_out(lin_input)[aa_index], 0),
                "dTm": torch.unsqueeze(self.dtm_out(lin_input)[aa_index], 0)
            })

        return out