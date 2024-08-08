import numpy as np
import torch
from loguru import logger
from tsaug import Crop, Drift, Quantize, Reverse, TimeWarp


class TimeSeriesAugmentation:
    """
    cite "https://tsaug.readthedocs.io/en/stable/quickstart.html"
    TimeWarp() * 5  # random time warping 5 times in parallel
        + Crop(size=300)  # random crop subsequences with length 300
        + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        + Reverse() @ 0.5  # with 50% probability, reverse the sequence
    """

    def __init__(self, rate: float = 1) -> None:
        self.rate = rate
        self.augmenter = (
            TimeWarp() * rate
            + Drift(max_drift=(0.1, 0.5)) @ 0.8
            + Quantize(n_levels=[10, 20, 30])
        )

    def __call__(self, x, feats=0):
        """_summary_

        Args:
            x (np.array): [B, F, T]
            y (np.array): [B, F, T]
            feats (int): denotes the starting index of feature to be augmented

        Returns:
            _type_: _description_
        """

        idx = np.repeat(
            np.arange(x.shape[1])[np.newaxis, feats:], x.shape[0], axis=0
        )
        x = x[:, feats:]
        x = x.transpose(0, 2, 1)
        logger.info(f"Augmenting {x.shape} with {self.rate}")
        x = self.augmenter.augment(x)
        x = x.transpose(0, 2, 1)

        logger.info(f"Augmented {x.shape}")

        return x, idx


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(
        permutation(sample, max_segments=config.augmentation.max_seg),
        config.augmentation.jitter_ratio,
    )

    return weak_aug, strong_aug


# You can choose any augmentation methods listed as follows:


def jitter(x, sigma=0.8):
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    factor = np.random.normal(
        loc=2.0, scale=sigma, size=(x.shape[0], x.shape[2])
    )
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[2] - 2, num_segs[i] - 1, replace=False
                )
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1))
        * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):

        li = []
        for dim in range(x.shape[2]):
            li.append(
                CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(
                    orig_steps
                )
            )
        warper = np.array(li).T

        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1))
        * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(
                warp_steps[:, dim],
                warp_steps[:, dim] * random_warps[i, :, dim],
            )(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(
                orig_steps,
                np.clip(scale * time_warp, 0, x.shape[1] - 1),
                pat[:, dim],
            ).T
    return ret


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1] - target_len, size=(x.shape[0])
    ).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                pat[starts[i] : ends[i], dim],
            ).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.0]):
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
    ).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[: window_starts[i], dim]
            window_seg = np.interp(
                np.linspace(
                    0, warp_size - 1, num=int(warp_size * warp_scales[i])
                ),
                window_steps,
                pat[window_starts[i] : window_ends[i], dim],
            )
            end_seg = pat[window_ends[i] :, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(
                np.arange(x.shape[1]),
                np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                warped,
            ).T
    return ret


def cutout(ts, perc=0.1):
    seq_len = ts.shape[0]
    new_ts = ts.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len - win_len - 1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    new_ts[start:end, ...] = 0
    return new_ts


def mixup_time(X, Y, alpha):
    # 生成混合系数 lambda
    lam = np.random.beta(alpha, alpha)
    # 遍历每个样本
    mixed_X = []
    mixed_Y = []
    for i in range(len(X)):
        # 对每个变量的相邻时间步进行线性插值
        mixed_x = lam * X[i, :, :-1] + (1 - lam) * X[i, :, 1:]
        mixed_y = lam * Y[i, :, :-1] + (1 - lam) * Y[i, :, 1:]
        mixed_X.append(mixed_x)
        mixed_Y.append(mixed_y)
    return np.array(mixed_X), np.array(mixed_Y)


def mixup_sample(X, Y, alpha, num_samples):
    mixed_X = []
    mixed_Y = []
    for _ in range(num_samples):
        # 生成混合系数 lambda
        lam = np.random.beta(alpha, alpha)
        # 随机选择两个不同的样本索引
        idx1 = np.random.randint(0, len(X))
        idx2 = np.random.randint(0, len(X))
        # 对每个样本进行混合增强
        mixed_x = lam * X[idx1] + (1 - lam) * X[idx2]
        mixed_y = lam * Y[idx1] + (1 - lam) * Y[idx2]
        mixed_X.append(mixed_x)
        mixed_Y.append(mixed_y)
    return np.array(mixed_X), np.array(mixed_Y)


def fourier(query):
    """
    Function for the fourier transform
      input:
         - query: historical speed (B, N, T)
      output
         - fourier_coeff: fast fourier transform results (B, N, T * 2)
            - Note: convert fft results from complex number into real number
    """
    n_bins = query.shape[-1]
    f_trans = torch.fft.fft(query)
    fourier_coeff = torch.stack([2 * f_trans.real, -2 * f_trans.imag], dim=-1)
    return fourier_coeff / n_bins


if __name__ == "__main__":
    X = np.array(
        [
            [[1, 2, 3], [3, 4, 5], [4, 5, 6]],
            [[2, 3, 4], [5, 6, 7], [7, 8, 9]],
            [[2, 3, 4], [5, 6, 7], [7, 8, 9]],
        ]
    )
    Y = np.array(
        [
            [[1, 2, 3], [3, 4, 5], [4, 5, 6]],
            [[2, 3, 4], [5, 6, 7], [7, 8, 9]],
            [[2, 3, 4], [5, 6, 7], [7, 8, 9]],
        ]
    )

    AUG_x, AUG_y = mixup_sample(X, Y, 0.3, 2)
    import pdb

    pdb.set_trace()
