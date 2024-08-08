import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import json


class MaskStat:
    """
    get the statistics of data, including mean, std, min and max

    is_col means whether to normalize by column.
    """

    def __init__(self, is_col=False) -> None:
        self.datas = []
        self.n_feats = []  # store the number of all sensors during each period.
        self.is_col = is_col

    def update(self, data):
        self.datas.append(data)
        self.n_feats.append(data.shape[1])

    def compute(self):
        feats = [(0, self.n_feats[0])] + [
            (self.n_feats[i], self.n_feats[i + 1]) for i in range(len(self.n_feats) - 1)
        ]
        stat = {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }

        if self.is_col:
            # statistics for
            for s, e in feats:
                # enumerate training periods
                _data = []
                for d in self.datas:
                    if d[:, s:e].shape[1] != 0:
                        _data.append(d[:, s:e])
                _data = np.concatenate(_data, axis=0)
                stat["mean"].append(np.mean(_data, axis=0))  # (num_feature)
                stat["std"].append(np.std(_data, axis=0))
                stat["min"].append(np.min(_data, axis=0))
                stat["max"].append(np.max(_data, axis=0))
            stat = {k: np.concatenate(v) for k, v in stat.items()}
        else:
            _data = [_d.flatten() for _d in self.datas]
            _data = np.concatenate(_data)
            stat["mean"] = [np.mean(_data)]
            stat["std"] = [np.std(_data)]
            stat["min"] = [np.min(_data)]
            stat["max"] = [np.max(_data)]

        return stat


def slice_window(data, window_size):
    """
    Slice the data into windows.
    [n, ...] -> [n - window_size + 1, window_size, ...]
    """
    windows = np.array(
        [data[i : i + window_size] for i in range(data.shape[0] - window_size + 1)]
    )
    return windows


def generate_all_data(data, ts_data, len_x, len_y, rules, is_col=False):
    xs, ys, ts = [], [], []
    cur_idx = 0
    data_stat = MaskStat(is_col)
    for i, (n_feats, n_day) in enumerate(rules):
        n_idx = n_day * 24 * 12
        fin_idx = cur_idx + n_idx
        # expanding sensors can be descided by sequence or spatial range
        if isinstance(n_feats, list):
            _data = data[cur_idx:fin_idx][:, n_feats]
        else:
            _data = data[cur_idx:fin_idx, :n_feats]
        _s_data = slice_window(_data, len_x + len_y)

        if _s_data.shape[0] < 1:
            # for empty valid set
            _xs, _ys, _ts = [], [], []
        else:
            _xs, _ys = _s_data[:, :len_x], _s_data[:, -len_y:]
            # 时间划分有bug，因为_xs用的是滑窗的样本大小，_ts还是计算的全部
            # _ts = ts_data[cur_idx:fin_idx]
            _ts = ts_data[cur_idx : cur_idx + _xs.shape[0]]
        if i < len(rules) - 2:
            data_stat.update(_data)

        xs.append(_xs)
        ys.append(_ys)
        ts.append(_ts)
        cur_idx = fin_idx
        # print(cur_idx)

    stats = data_stat.compute()

    return xs, ys, ts, stats


def main(data_path, dis_matrix_path, path_out, len_x, len_y, expanding_rules, is_col):
    # p_pems = Path("../raw/largestGLA/")

    # Loading data
    data = pd.read_csv(data_path).values
    ts_feat = data[:, 0]
    data = data[:, 1:].astype(np.float32)  # [N, num_feature]
    distance_matrix = np.load(dis_matrix_path)

    # rearange data and distance
    continual_list = list(np.load("./random_expansion_continual.npy"))
    expanding_list = list(np.load("./random_expansion_expanding.npy"))
    data = data[:, continual_list+expanding_list]
    distance_matrix = distance_matrix[continual_list+expanding_list,:][:, continual_list+expanding_list]

    adj1 = distance_matrix[:296, :][:, :296]
    xs, ys, ts, stats = generate_all_data(
        data, ts_feat, len_x, len_y, expanding_rules, is_col=is_col
    )

    assert len(xs) >= 4, "without expanding setting"
    for i in range(len(xs) - 2):
        print("shape of train_{} :".format(i))
        print(xs[i].shape, ys[i].shape, ts[i].shape)
    if xs[-2] != []:
        print("shape of valid :")
        print(xs[-2].shape, ys[-2].shape, ts[-2].shape)
    else:
        print("empty valid set")
    print("shape of test:")
    print(xs[-1].shape, ys[-1].shape, ts[-1].shape)
    # import pdb; pdb.set_trace()
    train_1 = (xs[0], ys[0], None)
    train_2 = (xs[1], ys[1], None)
    valid = (xs[2], ys[2], None)
    test = (xs[3], ys[3], None)

    train_1 = np.array(train_1, dtype=object)
    train_2 = np.array(train_2, dtype=object)
    valid = np.array(valid, dtype=object)
    test = np.array(test, dtype=object)
    graph = np.array((adj1, distance_matrix), dtype=object)
    np.savez_compressed(
        path_out,
        train_1=train_1,
        train_2=train_2,
        valid=valid,
        test=test,
        stats=stats,
        graph=graph,
    )  # about 4.2G / compressed 2.3G


if __name__ == "__main__":
    path = Path("../../../Baseline-Incremental/")
    data_path = path / "gla_pemsd7.csv"
    dis_matrix_path = path / "gla_dis.npy"
    len_x, len_y = 12, 12
    path_out = Path("../processed/random_pems-63_296-3_447-2-12_12_C.npz")
    expanding_rules = [(296, 63), (447, 3), (447, 2), (447, 22)]
    is_col = True # if column-wise normalize
    main(data_path, dis_matrix_path, path_out, len_x, len_y, expanding_rules, is_col)
