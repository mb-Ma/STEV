import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class MaskStat:
    def __init__(self, is_col=False) -> None:
        self.datas = []
        self.n_feats = (
            []
        )  # store the number of all sensors during each period.
        self.is_col = is_col

    def update(self, data):
        self.datas.append(data)
        self.n_feats.append(data.shape[1])

    def compute(self):
        feats = [(0, self.n_feats[0])]
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
        [
            data[i : i + window_size]
            for i in range(data.shape[0] - window_size + 1)
        ]
    )
    return windows


def generate_all_data(data, ts_data, len_x, len_y, rules, is_col=False):
    xs, ys, xts, yts = [], [], [], []
    cur_idx = 0
    data_stat = MaskStat(is_col)
    for i, (n_feats, n_day) in enumerate(rules):
        n_idx = n_day * 24
        fin_idx = cur_idx + n_idx
        # expanding sensors can be descided by sequence or spatial range
        if isinstance(n_feats, list):
            _data = data[cur_idx:fin_idx][:, n_feats]
        else:
            _data = data[cur_idx:fin_idx, :n_feats]
        _s_data = slice_window(_data, len_x + len_y)
        _s_ts = slice_window(ts_data[cur_idx:fin_idx], len_x + len_y)

        if _s_data.shape[0] < 1:
            # for empty valid set
            _xs, _ys, _xts, _yts = [], [], [], []
        else:
            _xs, _ys = _s_data[:, :len_x], _s_data[:, -len_y:]
            # 时间划分有bug，因为_xs用的是滑窗的样本大小，_ts还是计算的全部
            # _ts = ts_data[cur_idx:fin_idx]
            if ts_data is not None:
                _xts = _s_ts[:, :len_x]
                _yts = _s_ts[:, -len_y:]
        if i < len(rules) - 2:
            data_stat.update(_data)

        xs.append(_xs)
        ys.append(_ys)
        if ts_data is not None:
            xts.append(_xts)
            yts.append(_yts)
        cur_idx = fin_idx

    stats = data_stat.compute()

    return xs, ys, xts, yts, stats

def main(data_path, path_out, len_x, len_y, expanding_rules, is_col):
    # Loading data
    data = pd.read_csv(data_path).values
    ts_feat = data[:, 0]
    data = data[:, 1:].astype(np.float32)  # [N, num_feature]

    xs, ys, xts, yts, stats = generate_all_data(
        data, ts_feat, len_x, len_y, expanding_rules, is_col=is_col
    )

    assert len(xs) >= 4, "without expanding setting"
    for i in range(len(xs) - 2):
        print("shape of train_{} :".format(i))
        print(xs[i].shape, ys[i].shape)
    if xs[-2] != []:
        print("shape of valid :")
        print(xs[-2].shape, ys[-2].shape)
    else:
        print("empty valid set")
    print("shape of test:")
    print(xs[-1].shape, ys[-1].shape)
    xs[0] = np.concatenate([xs[0], xts[0][:, :, np.newaxis]], axis=-1)
    xs[1] = np.concatenate([xs[1], xts[1][:, :, np.newaxis]], axis=-1)
    xs[2] = np.concatenate([xs[2], xts[2][:, :, np.newaxis]], axis=-1)
    xs[3] = np.concatenate([xs[3], xts[3][:, :, np.newaxis]], axis=-1)
    ys[0] = np.concatenate([ys[0], yts[0][:, :, np.newaxis]], axis=-1)
    ys[1] = np.concatenate([ys[1], yts[1][:, :, np.newaxis]], axis=-1)
    ys[2] = np.concatenate([ys[2], yts[2][:, :, np.newaxis]], axis=-1)
    ys[3] = np.concatenate([ys[3], yts[3][:, :, np.newaxis]], axis=-1)
    
    train = (np.concatenate([xs[0], xs[1]], axis=0), np.concatenate([ys[0], ys[1]],axis=0))
    valid = (xs[2], ys[2])
    test = (xs[3], ys[3])
    train = np.array(train, dtype=object)
    valid = np.array(valid, dtype=object)
    test = np.array(test, dtype=object)
    np.savez_compressed(
        path_out,
        train= train,
        valid=valid,
        test=test,
        stats=stats,
    ) 


if __name__ == "__main__":
    path = Path("../raw/")
    data_path = path / "electricity.csv"
    len_x, len_y = 12, 12
    path_out = Path("../processed/elc-925_321-7_321-3_321-12_12_C_Oracle.npz")
    expanding_rules = [(321, 925), (321, 7), (321, 3), (321, 161)]
    is_col = True # if column-wise normalize
    main(data_path, path_out, len_x, len_y, expanding_rules, is_col)
