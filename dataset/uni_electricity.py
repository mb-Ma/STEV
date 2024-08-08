import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from utils.augmente import TimeSeriesAugmentation
from utils.scaler import init_scaler


class uniElectricity(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        scaler=None,
    ):
        self.mode = mode
        input_size = cfg.input_size
        output_size = cfg.output_size

        data = np.load(cfg.data_path, allow_pickle=True)

        if scaler is None:
            self.scaler = init_scaler(cfg, data["stats"].item())
        else:
            self.scaler = scaler

        if mode == "train":
            self.processTraindata(cfg, input_size, output_size, data)
        else:
            self.X, self.y, _ = data[mode]
            self.X, self.y, self.idx = self._process_xy(
                self.X, self.y, input_size, output_size
            )
        logger.info(f"Dataset '{mode}' shape: {self.X.shape}")
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).float()

    def processTraindata(self, cfg, input_size, output_size, data):
        keys = list(data.keys())
        self.X, self.y, self.idx = [], [], []
        for k in keys:
            if k.startswith("train"):
                
                if cfg.is_first_period_only and k.endswith("2"):
                    continue
                
                x, y, _ = data[k]

                augement_cond = cfg.augementation_rate != 0 and k.endswith("2")
                x, y, idx = self._process_xy(
                    x,
                    y,
                    input_size,
                    output_size,
                    cfg.augementation_rate,
                    0 if cfg.augement_all else cfg.first_feats,
                    augement_cond,
                )
                self.X.append(x)
                self.y.append(y)
                self.idx.append(idx)
                
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.idx = np.concatenate(self.idx, axis=0)

    def _process_xy(
        self,
        x,
        y,
        input_size,
        output_size,
        auge_rate=0,
        auge_feats=0,
        auge_cond=False,
    ):
        x = x.transpose(0, 2, 1)
        y = y.transpose(0, 2, 1)
        n_feats = x.shape[1]
        idx = np.repeat(np.arange(n_feats)[np.newaxis, :], x.shape[0], axis=0)

        x = self.scaler.transform(x)
        y = self.scaler.transform(y)

        if auge_cond:
            a_x, a_y, a_idx = TimeSeriesAugmentation(auge_rate)(
                x, y, auge_feats
            )
            a_x = a_x.reshape(-1, input_size)
            a_y = a_y.reshape(-1, output_size)
            a_idx = a_idx.reshape(-1, 1)

        x = x.reshape(-1, input_size)
        y = y.reshape(-1, output_size)
        idx = idx.reshape(-1, 1)

        if auge_cond:
            x = np.concatenate([x, a_x], axis=0)
            y = np.concatenate([y, a_y], axis=0)
            idx = np.concatenate([idx, a_idx], axis=0)

        return x, y, idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.idx[idx]


def get_dataloaders(cfg):
    train = uniElectricity(cfg, "train")
    scaler = train.scaler

    valid = uniElectricity(cfg, "valid", scaler)
    test = uniElectricity(cfg, "test", scaler)

    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader, test_loader, scaler


if __name__ == "__main__":
    config = {
        "data_path": "./data/processed/elc-928_165-7_321-0-12_12.npz",
        "horizon": 12,
        "batch_size": 64,
    }
    config = type("config", (), config)
    # dataset = Uni_elc(config, "train")
    loaders = get_dataloaders(config)
