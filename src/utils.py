import torch
import numpy as np


def random_binary_mask(N, D, zero_ratio=0.9):
    masks = torch.ones(N, D)

    n_zeros = int(D * zero_ratio)

    perm = np.arange(D)
    idx = np.array([perm] * N)
    np.apply_along_axis(np.random.shuffle, 1, idx)

    indices = torch.tensor(idx[:, :n_zeros])
    masks.scatter_(1, indices, 0)

    return masks
