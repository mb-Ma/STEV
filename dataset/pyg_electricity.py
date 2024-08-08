import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch_geometric.utils as pyg_utils
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.augmente import TimeSeriesAugmentation, jitter, mixup_sample
from utils.scaler import init_scaler


def time_encoder(ts):
    """
    ts: [N, L] N samples, L timesteps
    output: [N, L, 2] contains elements that belong to which workday and time of day
    """
    
    df = pd.DataFrame(ts)
    df_datetime = df.apply(pd.to_datetime)
    hours = df_datetime.apply(lambda col: col.dt.hour).to_numpy()
    weekday = df_datetime.apply(lambda col: col.dt.dayofweek).to_numpy()
    
    return np.stack([hours, weekday], axis=-1)


def process_one(
                data,
                scaler,
                augment_data=False,
                contrastive_aug=False,
                adj=None,
                is_pregraph=False,
                period=1,
    ):
    x, y = data
    x, y, xts= x[..., :-1].astype(np.float32), y[..., :-1].astype(np.float32), x[..., -1]

    logger.info(f"X shape: {x.shape}, y shape: {y.shape}")
    x = x.transpose(0, 2, 1)  # [B, N, T]
    y = y.transpose(0, 2, 1)  # [B, N, T]

    x = scaler.transform(x)
    y = scaler.transform(y)
    ts_feat = time_encoder(xts)[:, 0, :]
    
    if augment_data:
        # using mixup augmentation
        if period == 2:
            a_x, a_y = mixup_sample(x, y, 0.3, len(x) // 2)
            x = np.concatenate([x, a_x], axis=0)
            y = np.concatenate(
                [y, a_y], axis=0
            )
    if contrastive_aug:
        # aug_x, aug_y = mixup_sample(x, y, 0.3, len(x) // 2)
        augmenter = TimeSeriesAugmentation(rate=1)
        aug_x, _ = augmenter(x)
        aug_x = torch.from_numpy(aug_x).float()
        # aug_y = torch.from_numpy(aug_y).float()
    

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    ts_feat = torch.from_numpy(ts_feat).long()

    if is_pregraph:
        # add pre-defined graph, denoted as adj matrix.
        sparse_adj = pyg_utils.dense_to_sparse(
            torch.from_numpy(adj[period - 1]).float()
        )
        edge_index, edge_attr = sparse_adj[0], sparse_adj[1]
        if contrastive_aug:
            dataset = [
                Data(x=x_i, edge_index=edge_index, edge_attr=edge_attr, y=y_i, aug_x=aug_x_i, aug_y = y_i)
                for x_i, y_i, aug_x_i in zip(x, y, aug_x)
            ]
        else:
            dataset = [
                Data(x=x_i, edge_index=edge_index, edge_attr=edge_attr, y=y_i)
                for x_i, y_i in zip(x, y)
            ]
    else:
        # encapsulation each sample [num nodes, time_features] into a graph.
        # will do it before neural network.
        if contrastive_aug:
            dataset = [
                Data(x=x_i, y=y_i, n_id=torch.arange(x.shape[1]), aug_x=aug_x_i, time=ts_i)
                for x_i, y_i, aug_x_i, ts_i in zip(x, y, aug_x, ts_feat)
            ]  # each sample is a Data object
        else:
            dataset = [
                Data(x=x_i, y=y_i, n_id=torch.arange(x.shape[1]), time=ts_i)
                for x_i, y_i, ts_i in zip(x, y, ts_feat)
            ]
    return dataset


def get_dataloaders(cfg):
    # load processed data
    data = np.load(cfg.data_path, allow_pickle=True)
    # add Gaussian kernel function
    if cfg.is_pregraph:
        # refer to https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py
        def process_adj(adj, normalized_k):
            distances = adj[~np.isinf(adj)].flatten()
            std = distances.std()
            adj_mx = np.exp(-np.square(adj / std))
            # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
            adj_mx[adj_mx < normalized_k] = 0

        data["graph"][0] = process_adj(data["graph"][0], 0.1)
        data["graph"][1] = process_adj(data["graph"][1], 0.1)

    # get statistical info, i.e., mean, std, max, min.
    stats = data["stats"].item()
    scaler = init_scaler(cfg, stats)
    train = []
    for k in data.keys():
        if k.startswith("train"):
            # weather using data augmentation and weather the expading period.
            train.append(
                process_one(
                    data[k],
                    scaler,
                    cfg.augment_data,
                    cfg.is_cons_loss,
                    adj=None,
                    is_pregraph=cfg.is_pregraph,
                    period=int(k[-1]),
                )
            )

    # if over-sampling
    # indices = list(range(len(train[0]), len(train[0])+len(train[1])))
    if cfg.is_over_sampling:
        train = train[0] + train[1] + train[1]
    else:
        train = train[0] + train[1]
    # train = MyDataset(train, oversample_indices=indices, oversample_factor=1)

    logger.info(f"Train Lenght: {len(train)}")
    train = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    valid = DataLoader(
        process_one(
            data["valid"],
            scaler,
            adj=None,
            is_pregraph=cfg.is_pregraph,
            period=2,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test = DataLoader(
        process_one(
            data["test"],
            scaler,
            adj=None,
            is_pregraph=cfg.is_pregraph,
            period=2,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return train, valid, test, scaler


if __name__ == "__main__":
    config = {
        "data_path": "./data/processed/elc-928_261-5_321-2_321-12_12.npz",
        "horizon": 12,
        "batch_size": 64,
    }
    config = type("config", (), config)
    # dataset = Uni_elc(config, "train")
    loaders = get_dataloaders(config)
