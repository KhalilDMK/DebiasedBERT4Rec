import numpy as np
import torch
import sys


def position_bias_in_data(position_distributions):
    position_distributions = position_distributions[1::, :]
    u = torch.Tensor([1 / position_distributions.shape[1]] * position_distributions.shape[1])
    return np.mean([js_divergence(x, u) for x in position_distributions])


def kl_divergence(p, q):
    return (p * torch.log2(p / q + sys.float_info.epsilon)).sum()


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return (0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)).cpu().item()