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
        self.normalizer = normalizer
        self.incre_num = incre_num

        X_list, Y_list, ts_list, gs_list = [], [], [], []
        for i in range(1, incre_num + 1):
            # get the sequential data
            data = np.load(path, allow_pickle=True)["train_" + str(i)]
            # x [samples, time, nodes, feature], gs [nodes, nodes]
            x, y = data
            x = x[..., np.newaxis]
            y = y[..., np.newaxis]
            X_list.append(x)
            Y_list.append(y)

        # maks previous data with zero and generator mask matrix
        self.num_node = X_list[-1].shape[2]

        # first normalize, then do padding.
        # 1. get the mean-std or max-min
        stat = np.load(path, allow_pickle=True)["stats"].item()
        if normalizer == "standard":
            self.scaler = self.get_MeanStd(stat["mean"], stat["std"], is_col=is_col)
        elif normalizer == "maxmin":
            self.scaler = self.get_MaxMin(stat["max"], stat["min"], is_col=is_col)
        else:
            self.scaler = None
        # 2. do mask
        # 2.1 fill then concatenate them
        _x, _y, mask = [], [], []
        for i in range(len(X_list) - 1):
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
        # 2.2 add the last one
        _x.append(X_list[-1])
        _y.append(Y_list[-1])
        mask.append(np.ones((X_list[-1].shape[0], X_list[-1].shape[2])))
        # 4. concatenate all elements
        self.x = np.vstack(_x)
        self.y = np.vstack(_y)
        self.mask = np.vstack(mask)

        self.to_CudaTensor()

        # 5. do transformation
        self.scaler.to_cuda()
        self.x = self.transform(self.x, self.scaler)
        self.y = self.transform(self.y, self.scaler)

    def get_MaxMin(self, max, min, is_col=False):
        scaler = MinMaxScaler(min=min, max=max, is_col=is_col)
        return scaler

    def get_MeanStd(self, mean, std, is_col=False):
        scaler = StandardScaler(mean, std, is_col=is_col)
        return scaler

    def transform(self, data, normalizer):
        data = normalizer.transform(data)
        return data

    def to_CudaTensor(self):
        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.x = TensorFloat(self.x)
        self.y = TensorFloat(self.y)
        self.mask = TensorFloat(self.mask)

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

    def __init__(self, path, types, normalizer=None) -> None:
        data = np.load(path, allow_pickle=True)[types]

        self.x, self.y = data
        print(
            f"{types}: X shape: {self.x.shape}, Y shape: {self.y.shape},"
        )
        self.x = np.expand_dims(self.x, -1)
        self.y = np.expand_dims(self.y, -1)

        if normalizer:
            self.transform(normalizer)
        else:
            raise Exception("Specify a normalizer!")

    def transform(self, normalizer):
        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.x, self.y = TensorFloat(self.x), TensorFloat(self.y)
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


def get_dataloaders(cfg):
    """
    incre_num: the number of incremental task

    is_last: if use the last incremental data
    """
    if cfg.incre_num == 1:
        train = STDataset(cfg.data_path, "train")
    elif cfg.is_last:
        train = STDataset(cfg.data_path, "train" + str(cfg.incre_num))
    else:
        train = Mask_STDataset(
            cfg.data_path, normalizer=cfg.scaler, incre_num=cfg.incre_num, is_col=cfg.is_col
        )
    scaler = train.scaler
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    valid = STDataset(cfg.data_path, "valid", scaler)
    valid_loader = DataLoader(
        valid, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    test = STDataset(cfg.data_path, "test", scaler)
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
