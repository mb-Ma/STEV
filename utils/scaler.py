import numpy as np
import torch


def init_scaler(cfg, stats):
    if cfg.scaler == "standard":
        scaler = StandardScaler(stats["mean"], stats["std"], is_col=cfg.is_col)
    elif cfg.scaler == "minmax":
        scaler = MinMaxScaler(stats["min"], stats["max"], is_col=cfg.is_col)
    else:
        raise ValueError(f"Unknown scaler: {cfg.scaler}")

    return scaler


class StandardScaler:
    def __init__(self, mean=None, std=None, is_col=False):
        self.is_col = is_col
        if not is_col:
            mean = np.array([mean])
            std = np.array([std])
        self.mean = mean
        self.std = std

    def fit(self, X, mask=None):
        if mask is not None:
            X = np.ma.masked_array(X, ~mask.astype(bool))
        if self.is_col:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        else:
            self.mean = np.mean(X)
            self.std = np.std(X)

    def transform(self, X):
        '''
        masking scheme X [B, T, N, C]
        pyg scheme X [B, N, T]
        '''
        if self.is_col:
            if len(X.shape) == 3:
                n_feats = X.shape[1]
                mean = self.mean[np.newaxis, :n_feats, np.newaxis]
                std = self.std[np.newaxis, :n_feats, np.newaxis]
            elif len(X.shape) == 4:
                n_feats = X.shape[2]
                mean = self.mean[np.newaxis, np.newaxis, :n_feats, np.newaxis]
                std = self.std[np.newaxis, np.newaxis, :n_feats, np.newaxis]
            else:
                raise ValueError("input dimension error!!!")
            return (X - mean) / std
        else:
            return (X - self.mean) / self.std

    def inverse_transform_col(self, X, ptr):
        # X [B*N, T]
        # 需要根据ptr将均值方差对等排列，然后再进行计算。
        mean, std = [], []
        for i in range(1, len(ptr)):
            n_nodes = ptr[i] - ptr[i-1]
            mean.append(self.mean[:n_nodes, np.newaxis])
            std.append(self.std[:n_nodes, np.newaxis])
        
        mean = np.concatenate(mean, axis=0)
        std = np.concatenate(std, axis=0)
        
        if torch.is_tensor(X) and not torch.is_tensor(std):
            std = torch.from_numpy(std).to(X.device)
            mean = torch.from_numpy(mean).to(X.device)

        return X * std + mean

    def inverse_transform(self, X, idx=None):
        std = self.std
        mean = self.mean
        if torch.is_tensor(X) and not torch.is_tensor(std):
            std = torch.from_numpy(std).to(X.device)
            mean = torch.from_numpy(mean).to(X.device)
        if self.is_col:
            n_feats = X.shape[2]
            if torch.is_tensor(std):
                mean = mean[:n_feats].view((1, 1, n_feats, 1))
                std = std[:n_feats].view((1, 1, n_feats, 1))
            else:
                mean = mean[np.newaxis, :n_feats, np.newaxis]
                std = std[np.newaxis, :n_feats, np.newaxis]        

        return X * std + mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_cuda(self):
        self.mean = torch.Tensor(self.mean).cuda()
        self.std = torch.Tensor(self.std).cuda()


class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def fit(self, X, mask=None):
        if mask is not None:
            X = np.ma.masked_array(X, ~mask.astype(bool))
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_cuda(self):
        self.min = torch.Tensor(self.min).cuda()
        self.max = torch.Tensor(self.max).cuda()