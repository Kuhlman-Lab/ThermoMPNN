import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from transfer_model import TransferModel
from datasets import FireProtDataset, MegaScaleDataset, ComboDataset


def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


class TransferModelPL(pl.LightningModule):
    """Class managing training loop with pytorch lightning"""
    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModel(cfg)

        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False

        # set up metrics dictionary
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        assert len(batch) == 1
        mut_pdb, mutations = batch[0]
        pred, _ = self(mut_pdb, mutations)

        ddg_mses = []
        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                ddg_mses.append(F.mse_loss(out["ddG"], mut.ddG))
                for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        loss = 0.0 if len(ddg_mses) == 0 else torch.stack(ddg_mses).mean()
        on_step = False
        on_epoch = not on_step

        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        if loss == 0.0:
            return None
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        if self.stage == 2: # for second stage, drop LR by factor of 10
            self.learn_rate /= 10.
            print('New second-stage learning rate: ', self.learn_rate)

        if not cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

        if self.model.lightattn:  # adding light attention parameters
            if self.stage == 2:
                param_list.append({"params": self.model.light_attention.parameters(), "lr": 0.})
            else:
                param_list.append({"params": self.model.light_attention.parameters()})


        mlp_params = [
            {"params": self.model.both_out.parameters()},
            {"params": self.model.ddg_out.parameters()}
            ]

        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_ddG_mse'
            }
        else:
            return opt


def train(cfg):
    print('Configuration:\n', cfg)

    if 'project' in cfg:
        wandb.init(project=cfg.project, name=cfg.name)
    else:
        cfg.name = 'test'

    # load the specified dataset
    if len(cfg.datasets) == 1: # one dataset training
        dataset = cfg.datasets[0]
        if dataset == 'fireprot':
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = FireProtDataset(cfg, "val")
        elif dataset == 'megascale_s669':
            train_dataset = MegaScaleDataset(cfg, "train_s669")
            val_dataset = MegaScaleDataset(cfg, "val")
        elif dataset.startswith('megascale_cv'):
                cv = dataset[-1]
                train_dataset = MegaScaleDataset(cfg, f"cv_train_{cv}")
                val_dataset = MegaScaleDataset(cfg, f"cv_val_{cv}")
        elif dataset == 'megascale':
                train_dataset = MegaScaleDataset(cfg, "train")
                val_dataset = MegaScaleDataset(cfg, "val")
        else:
            raise ValueError("Invalid dataset specified!")
    else:
        train_dataset = ComboDataset(cfg, "train")
        val_dataset = ComboDataset(cfg, "val")

    if 'num_workers' in cfg.training:
        train_workers, val_workers = int(cfg.training.num_workers * 0.75), int(cfg.training.num_workers * 0.25)
    else:
        train_workers, val_workers = 0, 0

    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

    model_pl = TransferModelPL(cfg)
    model_pl.stage = 1

    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model="all") if 'project' in cfg else None
    max_ep = cfg.training.epochs if 'epochs' in cfg.training else 100

    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep,
                         accelerator=cfg.platform.accel, devices=1)
    trainer.fit(model_pl, train_loader, val_loader)

    if 'two_stage' in cfg.training:  # sequential combo training
        if cfg.training.two_stage:
            print('Two-stage Training Enabled')
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = MegaScaleDataset(cfg, "val")
            train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
            val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

            model_pl.stage = 2
            # re-start training with a new trainer
            trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep * 2,
                                accelerator=cfg.platform.accel, devices=1)
            trainer.fit(model_pl, train_loader, val_loader, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) == 1:
        yaml = "config.yaml"
    else:
        yaml = sys.argv[1]

    cfg = OmegaConf.load(yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    train(cfg)
