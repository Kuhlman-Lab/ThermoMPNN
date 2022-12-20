import random
import torch
from cache import cache
from fireprot_dataset import FireProtDataset
from training.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader

@cache(lambda cfg, p: "")
def train_clusters_cached(cfg, params):
    return build_training_clusters(params, False)

@cache(lambda cfg, x, max_len, units: (max_len, units), disable=False)
def get_train_pdbs(cfg, train_set, max_len, units):
    loader = torch.utils.data.DataLoader(train_set)
    return get_pdbs(loader, max_length=max_len, num_units=units)
    
@cache(lambda cfg, x, max_len, units: (max_len, units), disable=False)
def get_valid_pdbs(cfg, valid_set, max_len, units):
    loader = torch.utils.data.DataLoader(valid_set)
    return get_pdbs(loader, max_length=max_len, num_units=units)

class ComboDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.fireprot = FireProtDataset(cfg, split)

        data_path = cfg.platform.pdb_dir
        params = {
            "LIST"    : f"{data_path}/list.csv", 
            "VAL"     : f"{data_path}/valid_clusters.txt",
            "TEST"    : f"{data_path}/test_clusters.txt",
            "DIR"     : f"{data_path}",
            "DATCUT"  : "2030-Jan-01",
            "RESCUT"  : 3.5, #resolution cutoff for PDBs (3.5)
            "HOMO"    : 0.70 #min seq.id. to detect homo chains
        }

        train, valid, test = train_clusters_cached(cfg, params)
        train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
        valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)

        max_protein_length = cfg.data.max_protein_length # 10000
        num_units = cfg.data.num_pdb_datapoints

        train_pdbs = get_train_pdbs(cfg, train_set, max_protein_length, num_units)
        valid_pdbs = get_valid_pdbs(cfg, valid_set, max_protein_length, num_units)

        split_pdbs = {
            "val": valid_pdbs,
            "train": train_pdbs
        }[split]

        self.pdb_dataset = StructureDataset(split_pdbs, truncate=None, max_length=max_protein_length)

    def __len__(self):
        return len(self.fireprot)

    def __getitem__(self, index):
        # get the ith fireprot item and randomly sample from the pdb dataset
        return self.fireprot[index], random.choice(self.pdb_dataset)