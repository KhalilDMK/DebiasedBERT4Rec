import numpy as np
import torch
import sys


def position_bias_in_data(position_distributions):
    position_distributions = position_distributions[1::, :]
    u = torch.Tensor([1 / position_distributions.shape[1]] * position_distributions.shape[1]).to(position_distributions.device)
    return np.mean([js_divergence(x, u) for x in position_distributions])


def propensity_bias_in_data(popularity_vector):
    popularity_vector = popularity_vector[1::]
    u = torch.Tensor([1 / len(popularity_vector)] * len(popularity_vector)).to(popularity_vector.device)
    return js_divergence(popularity_vector, u)


def propensity_bias_in_data_kl_p_u(popularity_vector):
    popularity_vector = popularity_vector[1::]
    u = torch.Tensor([1 / len(popularity_vector)] * len(popularity_vector)).to(popularity_vector.device)
    return kl_divergence(popularity_vector, u)


def propensity_bias_in_data_kl_u_p(popularity_vector):
    popularity_vector = popularity_vector[1::]
    u = torch.Tensor([1 / len(popularity_vector)] * len(popularity_vector)).to(popularity_vector.device)
    return kl_divergence(u, popularity_vector)


def propensity_bias_in_data_mse(popularity_vector):
    popularity_vector = popularity_vector[1::]
    u = torch.Tensor([1 / len(popularity_vector)] * len(popularity_vector)).to(popularity_vector.device)
    return mse(u, popularity_vector)


def propensity_bias_in_data_mae(popularity_vector):
    popularity_vector = popularity_vector[1::]
    u = torch.Tensor([1 / len(popularity_vector)] * len(popularity_vector)).to(popularity_vector.device)
    return mae(u, popularity_vector)


def temporal_exposure_bias_in_data(position_distributions):
    position_distributions = position_distributions[1::, :].flatten()
    u = torch.Tensor([1 / len(position_distributions)] * len(position_distributions)).to(position_distributions.device)
    return js_divergence(position_distributions, u)


def temporal_exposure_bias_in_data_kl_p_u(position_distributions):
    position_distributions = position_distributions[1::, :].flatten()
    u = torch.Tensor([1 / len(position_distributions)] * len(position_distributions)).to(position_distributions.device)
    return kl_divergence(position_distributions, u)


def temporal_exposure_bias_in_data_kl_u_p(position_distributions):
    position_distributions = position_distributions[1::, :].flatten()
    u = torch.Tensor([1 / len(position_distributions)] * len(position_distributions)).to(position_distributions.device)
    return kl_divergence(u, position_distributions)


def temporal_exposure_bias_in_data_mse(position_distributions):
    position_distributions = position_distributions[1::, :].flatten()
    u = torch.Tensor([1 / len(position_distributions)] * len(position_distributions)).to(position_distributions.device)
    return mse(position_distributions, u)


def temporal_exposure_bias_in_data_mae(position_distributions):
    position_distributions = position_distributions[1::, :].flatten()
    u = torch.Tensor([1 / len(position_distributions)] * len(position_distributions)).to(position_distributions.device)
    return mae(position_distributions, u)


def kl_divergence(p, q):
    #return (p * torch.log2(p / q + sys.float_info.epsilon)).sum()
    return (p * torch.log2(p / q)).sum()


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return (0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)).cpu().item()


def mse(p, q):
    return torch.pow(p - q, 2).mean().cpu().item()


def mae(p, q):
    return torch.abs(p - q).mean().cpu().item()
