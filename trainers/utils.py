import torch
import numpy as np
import sys
from itertools import combinations


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def recall_(hits, labels, k):
    return (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()


def ndcg_(hits, labels, k):
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)]).to(dcg.device)
    ndcg = (dcg / idcg).mean().cpu().item()
    return ndcg


def avg_popularity(cut, device, popularity_vector):
    cut = torch.flatten(cut).tolist()
    return popularity_vector.to(device)[cut].mean().cpu().item()


def efd(cut, device, popularity_vector):
    cut = torch.flatten(cut).tolist()
    return (- torch.log2(popularity_vector.to(device)[cut] + torch.Tensor([sys.float_info.epsilon]).to(device))).mean().cpu().item()


def avg_pairwise_similarity(cut, device, item_similarity_matrix):
    item_combinations = torch.cat([torch.combinations(x) for x in cut])
    item_similarity_matrix = item_similarity_matrix.to(device)
    return item_similarity_matrix[torch.transpose(item_combinations, 0, 1).tolist()].mean().cpu().item()


def positional_frequency(recommendations, recommendation_positions, position_distributions):
    #return np.mean([sum(position_distributions[item, max(0, int(position) - neighborhood // 2):min(position_distributions.shape[1] - 1, int(position) + neighborhood // 2)]) for (item, position) in zip(recommendations, recommendation_positions)])
    return np.mean([position_distributions[item, int(position)] for (item, position) in zip(recommendations, recommendation_positions)])


def top_position_matching(recommendations, recommendation_positions, position_distributions):
    top_train_positions = np.argmax(position_distributions, axis=1)
    return np.mean([position == top_train_positions[item] for (item, position) in zip(recommendations, recommendation_positions)])


def recalls_and_ndcgs_for_ks(scores, labels, ks, popularity_vector, item_similarity_matrix):
    metrics = {}

    scores = scores
    labels = labels

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = recall_(hits, labels, k)
        metrics['NDCG@%d' % k] = ndcg_(hits, labels, k)
        metrics['AvgPop@%d' % k] = avg_popularity(cut, labels.device, popularity_vector)
        metrics['EFD@%d' % k] = efd(cut, labels.device, popularity_vector)
        metrics['Diversity@%d' % k] = avg_pairwise_similarity(cut, labels.device, item_similarity_matrix)

    return metrics
