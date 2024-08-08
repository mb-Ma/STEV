from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.scaler import MinMaxScaler, StandardScaler


class Mask_STDataset(Dataset):
    """
    only for the incrementaly train dataset using mask mechanism.
    """
    def __init__(self, path, normalizer=None, incre_num=1, is_col=False) -> None:
        self.path = path
        self.is_col = is_col
        self.normalizer = normalizer
        self.incre_num = incre_num

        X_list, Y_list, ts_list, gs_list = [], [], [], []
        for i in range(1, incre_num + 1):
            # get the sequential data
            data = np.load(path, allow_pickle=True)["train_" + str(i)]
            
            x, y = data
            x, y, xts= x[..., :-1].astype(np.float32), y[..., :-1].astype(np.float32), x[..., -1]
            x = x[..., np.newaxis] # # x [samples, time, nodes, feature], gs [nodes, nodes]
            y = y[..., np.newaxis]
            X_list.append(x)
            Y_list.append(y)
            ts_list.append(xts)
        
        # maks previous data with zero and generator mask matrix
        self.num_node = X_list[-1].shape[2]

        # first normalize, then do padding.
        # 1. get the mean-std or max-min
        stat = np.load(path, allow_pickle=True)["stats"].item()
        if normalizer == "standard":
            self.scaler = self.get_MeanStd(stat["mean"], stat["std"])
        elif normalizer == "maxmin":
            self.scaler = self.get_MaxMin(stat["max"], stat["min"])
        else:
            self.scaler = None
        # 3. do mask
        # 3.1 fill then concatenate them
        _x, _y, mask = [], [], []
        for i in range(len(X_list) - 1):
            # import pdb; pdb.set_trace()
            _x.append(
                np.pad(
                    X_list[i],
                    (
                        (0, 0),
                        (0, 0),
                        (0, self.num_node - X_list[i].shape[2]),
                        (0, 0),
                    ),
                    mode="constant",
                )
            )
            _y.append(
                np.pad(
                    Y_list[i],
                    (
                        (0, 0),
                        (0, 0),
                        (0, self.num_node - Y_list[i].shape[2]),
                        (0, 0),
                    ),
                    mode="constant",
                )
            )
            mask_matrix = np.concatenate(
                (
                    np.ones((X_list[i].shape[0], X_list[i].shape[2])),
                    np.zeros(
                        (
                            X_list[i].shape[0],
                            self.num_node - X_list[i].shape[2],
                        )
                    ),
                ),
                axis=1,
            )
            mask.append(mask_matrix)
        # 3.2 add the last one
        _x.append(X_list[-1])
        _y.append(Y_list[-1])
        mask.append(np.ones((X_list[-1].shape[0], X_list[-1].shape[2])))
        # 3. concatenate all elements
        self.x = np.vstack(_x)
        self.y = np.vstack(_y)
        self.mask = np.vstack(mask)

        # 2. do transformation
        self.x = self.transform(self.x, self.scaler)
        self.y = self.transform(self.y, self.scaler)

    def get_MaxMin(self, max, min):
        scaler = MinMaxScaler(min=min, max=max)
        return scaler

    def get_MeanStd(self, mean, std):
        scaler = StandardScaler(mean, std, is_col=self.is_col)
        return scaler

    def transform(self, data, normalizer):
        data = normalizer.transform(data)
        return data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            x: (N, T, S, F)
            y: (N, T, S, F)
        """
        return self.x[index], self.y[index], self.mask[index]


class STDataset(Dataset):
    """
    STDataset is a dataset for normal spatio-temporal data.

    """

    def __init__(self, path, types, normalizer=None, norma_name=None, is_col=False) -> None:
        self.is_col = is_col
        data = np.load(path, allow_pickle=True)[types]
        if types == "train":
            stat = np.load(path, allow_pickle=True)["stats"].item()
            if norma_name == "standard":
                self.scaler = self.get_MeanStd(stat["mean"], stat["std"])
            elif norma_name == "maxmin":
                self.scaler = self.get_MaxMin(stat["max"], stat["min"])
            else:
                self.scaler = None
        self.x, self.y = data
        self.x, self.y, self.ts = self.x[..., :-1].astype(np.float32), self.y[..., :-1].astype(np.float32), self.x[..., 0]
        if self.ts is None:
            print(
                f"{types}: X shape: {self.x.shape}, Y shape: {self.y.shape}"
            )
        else:
            print(
                f"{types}: X shape: {self.x.shape}, Y shape: {self.y.shape},"
                + f" TS shape: {self.ts.shape}"
            )
        self.x = np.expand_dims(self.x, -1)
        self.y = np.expand_dims(self.y, -1)

        if normalizer and types != "train":
            self.transform(normalizer)
        elif types == "train":
            self.transform(self.scaler)
        else:
            raise Exception("Specify a normalizer!")

    def transform(self, normalizer):
        self.x = normalizer.transform(self.x)
        self.y = normalizer.transform(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            x: (N, T, S, F)
            y: (N, T, S, F)
        """
        return self.x[index], self.y[index]

    def get_MaxMin(self, max, min):
        scaler = MinMaxScaler(min=min, max=max)
        return scaler

    def get_MeanStd(self, mean, std):
        scaler = StandardScaler(mean, std, is_col=self.is_col)
        return scaler


def get_dataloaders(cfg):
    """
    incre_num: 1: only the first period data
    is_last: if only use the expanding data
    is_oracle: if using oracle data
    """
    if cfg.is_oracle:
        train = STDataset(cfg.data_path, "train", norma_name=cfg.scaler, is_col=cfg.is_col)
    elif cfg.incre_num == 1:
        train = STDataset(cfg.data_path, "train"+ str(cfg.incre_num), norma_name=cfg.scaler, is_col=cfg.is_col)
    elif cfg.is_last:
        train = STDataset(cfg.data_path, "train" + str(cfg.incre_num), norma_name=cfg.scaler, is_col=cfg.is_col)
    else:
        train = Mask_STDataset(
            cfg.data_path, normalizer=cfg.scaler, incre_num=cfg.incre_num, is_col=cfg.is_col
        )
    scaler = train.scaler
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    valid = STDataset(cfg.data_path, "valid", normalizer=scaler, is_col=cfg.is_col)
    valid_loader = DataLoader(
        valid, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    test = STDataset(cfg.data_path, "test", normalizer=scaler, is_col=cfg.is_col)
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader, scaler


if __name__ == "__main__":
    # path config
    config = {
        "data_path": Path(
            "./data/processed/elc-928_261-4_321-3_321-12_12.npz"
        ),
        "batch_size": 32,
        "incre_num": 2,
        "is_last": False,
        "normalizer": "standard",
        "is_default_graph": False,
    }
    config = type("config", (), config)
    loaders = get_dataloaders(config)
