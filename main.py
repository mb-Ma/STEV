import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from loguru import logger as log
from omegaconf import open_dict


def init_dataset(cfg):
    import importlib

    # Dynamic import based on the chosen module
    loader_module = importlib.import_module(f"dataset.{cfg.dataset_name}")
    get_dataloaders = getattr(loader_module, "get_dataloaders")

    return get_dataloaders(cfg)


def init_base(cfg):
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(log_path, "main.log"))

    with open_dict(cfg):
        cfg.log_path = log_path

    log.info(cfg)

    return cfg


def seed_everything(seed=42):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(cfg):
    if cfg.model_name == "mlp":
        from model.mlp import MLPBase
        return MLPBase(cfg).to(cfg.device)
    elif cfg.model_name == "dyngraphwave":
        from model.dyngraphwave import DynGraphWave
        model = DynGraphWave(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "dynagcrn":
        from model.dyngraphwave import DynGraphWave
        model = DynGraphWave(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "adpdyngraphwave":
        from model.Adpgwnet import DynGraphWave
        model = DynGraphWave(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "clusteradpdyngraphwave":
        from model.ClusterAdpgwnet import DynGraphWave
        model = DynGraphWave(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "adpconsdyngraphwave":
        from model.AdpConsGwnet import DynGraphWave
        model = DynGraphWave(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "SepNet":
        from model.SepNet import SepNet
        model = SepNet(cfg).to(cfg.device)
        return model
    elif cfg.model_name == "agcrn":
        from model.agcrn import AGCRN
        model = AGCRN(cfg).to(cfg.device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        return model
    elif cfg.model_name == "megacrn":
        from model.MegaCRN import MegaCRN
        model = MegaCRN(cfg).to(cfg.device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        return model
    elif cfg.model_name == "gwnet":
        from model.gwnet import gwnet
        return gwnet(cfg).to(cfg.device)
    elif cfg.model_name == "sgp":
        from model.SGP import SGPModel
        return SGPModel(cfg).to(cfg.device)
    elif cfg.model_name == "msgnet":
        from model.msgnet import Model
        return Model(cfg).to(cfg.device)
    elif cfg.model_name == "ginar":
        from model.GINAR import GinAR
        model = GinAR(cfg).to(cfg.device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        return model
    elif cfg.model_name == 'FCSTGNN':
        from model.FCSTGNN import FC_STGNN
        return FC_STGNN(cfg).to(cfg.device)
    elif cfg.model_name == "Linear":
        from model.linear import Dlinear
        return Dlinear(cfg).to(cfg.device)
    elif cfg.model_name == "gru":
        from model.gru import GRUNet
        return GRUNet(cfg).to(cfg.device)
    elif cfg.model_name == "iTransformer":
        from model.iTransformer import Model
        return Model(cfg).to(cfg.device)
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = init_base(cfg.common)
    seed_everything(cfg.seed)

    train_loader, valid_loader, test_loader, scaler = init_dataset(cfg)
    model = init_model(cfg)
    # log model arch
    log.info(f"Model Arch:\n{model}")

    if cfg.name.startswith("pyg"):
        # processing pyg training mode
        if cfg.is_cons_loss:
            from pygTrainer import CosTrainer
            trainer = CosTrainer(model, scaler, cfg)
        else:
            from pygTrainer import Trainer
            trainer = Trainer(model, scaler, cfg)
    elif cfg.name.startswith("mask"):
        # processing mask training mode
        from Trainer_mask import Trainer
        trainer = Trainer(model, scaler, cfg)
    else:
        from Trainer import Trainer
        trainer = Trainer(model, scaler, cfg)
        
    model = trainer.train(train_loader, valid_loader)
    torch.save(model.state_dict(), os.path.join(cfg.log_path, "model.pt"))
    metrics, real_y, pred_y = trainer.test(test_loader)
    if True:
        np.savez(
            os.path.join(cfg.log_path, "result.npz"),
            real_y=real_y,
            pred_y=pred_y,
        )


if __name__ == "__main__":
    main()
