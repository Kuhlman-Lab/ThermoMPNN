import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef
import torch.nn.functional as F
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from transfer_model import TransferModel
from fireprot_dataset import FireProtDataset

def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(),
        "spearman": SpearmanCorrCoef(),
    }

class TransferModelPL(pl.LightningModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModel()
        
        self.ddg_lambda = cfg.loss.ddG_lambda
        self.dtm_lambda = cfg.loss.dTm_lambda
        self.learn_rate = cfg.learn_rate

        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            for out in ("ddG", "dTm"):
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        assert len(batch) == 1
        pdb, mutations = batch[0]
        pred = self(pdb, mutations)
        ddg_mses = []
        dtm_mses = []
        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                ddg_mses.append(F.mse_loss(out["ddG"], mut.ddG))
                for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)
            elif mut.dTm is not None:
                dtm_mses.append(F.mse_loss(out["dTm"], mut.dTm))
                for metric in self.metrics[f"{prefix}_metrics"]["dTm"].values():
                    metric.update(out["dTm"], mut.dTm)
        
        ddg_loss = 0.0 if len(ddg_mses) == 0 else torch.stack(ddg_mses).mean()
        dtm_loss = 0.0 if len(dtm_mses) == 0 else torch.stack(dtm_mses).mean()
        loss = self.dtm_lambda*dtm_loss + self.ddg_lambda*ddg_loss

        on_step = prefix == "train"
        on_epoch = not on_step
        for output in ("ddG", "dTm"):
            for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
                try:
                    self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))
                except ValueError:
                    # can't log metrics that haven't been updated
                    pass
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learn_rate)

def train(cfg):
    train_dataset = FireProtDataset(cfg, "train")
    val_dataset = FireProtDataset(cfg, "val")
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x)
    model_pl = TransferModelPL(cfg)
    trainer = pl.Trainer(num_sanity_val_steps=0)
    trainer.fit(model_pl, train_loader, val_loader)

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    train(cfg)