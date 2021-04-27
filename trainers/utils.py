import torch
import numpy as np
import sys
#from itertools import combinations
import pandas as pd
import os
from sklearn.metrics import roc_auc_score


#def recall(scores, labels, k):
#    scores = scores
#    labels = labels
#    rank = (-scores).argsort(dim=1)
#    cut = rank[:, :k]
#    hit = labels.gather(1, cut)
#    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


#def ndcg(scores, labels, k):
#    scores = scores.cpu()
#    labels = labels.cpu()
#    rank = (-scores).argsort(dim=1)
#    cut = rank[:, :k]
#    hits = labels.gather(1, cut)
#    position = torch.arange(2, 2+k)
#    weights = 1 / torch.log2(position.float())
#    dcg = (hits.float() * weights).sum(1)
#    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
#    ndcg = dcg / idcg
#    return ndcg.mean()

def recall(hits, labels, k):
    return hits.sum(1).mean().cpu().item()
    #return (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(hits, labels, k):
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)]).to(dcg.device)
    ndcg = (dcg / idcg).mean().cpu().item()
    return ndcg


def avg_popularity(cut_candidates, device, popularity_vector):
    cut_candidates = torch.flatten(cut_candidates).tolist()
    return popularity_vector.to(device)[cut_candidates].mean().cpu().item()


def efd(cut_candidates, device, popularity_vector):
    cut_candidates = torch.flatten(cut_candidates).tolist()
    return (- torch.log2(popularity_vector.to(device)[cut_candidates] + torch.Tensor([sys.float_info.epsilon]).to(device))).mean().cpu().item()


def avg_pairwise_similarity(cut_candidates, device, item_similarity_matrix):
    item_combinations = torch.cat([torch.combinations(x) for x in cut_candidates])
    item_similarity_matrix = item_similarity_matrix.to(device)
    return item_similarity_matrix[torch.transpose(item_combinations, 0, 1).tolist()].mean().cpu().item()


def positional_frequency(recommendations, recommendation_positions, position_distributions):
    #return np.mean([sum(position_distributions[item, max(0, int(position) - neighborhood // 2):min(position_distributions.shape[1] - 1, int(position) + neighborhood // 2)]) for (item, position) in zip(recommendations, recommendation_positions)])
    return np.mean([position_distributions[item, int(position)] for (item, position) in zip(recommendations, recommendation_positions)])


def top_position_matching(recommendations, recommendation_positions, position_distributions):
    top_train_positions = np.argmax(position_distributions, axis=1)
    return np.mean([position == top_train_positions[item] for (item, position) in zip(recommendations, recommendation_positions)])


def mse(p, q):
    return torch.pow(p - q, 2).mean().cpu().item()


def auc_score(p, q):
    return roc_auc_score(q.tolist(), p.tolist(), labels=[0, 1])


def metrics_for_ks(scores, labels, candidates, ks, popularity_vector, item_similarity_matrix):
    metrics = {}

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    ranked_candidates = candidates.gather(1, rank)
    for k in sorted(ks, reverse=True):
        cut = rank[:, :k]
        cut_candidates = ranked_candidates[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = recall(hits, labels, k)
        metrics['NDCG@%d' % k] = ndcg(hits, labels, k)
        metrics['AvgPop@%d' % k] = avg_popularity(cut_candidates, labels.device, popularity_vector)
        metrics['EFD@%d' % k] = efd(cut_candidates, labels.device, popularity_vector)
        metrics['Diversity@%d' % k] = avg_pairwise_similarity(cut_candidates, labels.device, item_similarity_matrix)

    return metrics


def metrics_for_ks_explicit(scores, ratings, tf_target):
    metrics = {}
    if tf_target == 'exposure':
        metrics['AUC'] = auc_score(scores, ratings)
    else:
        metrics['MSE'] = mse(scores, ratings)
    return metrics


def save_reconstructed(rec_columns, score_type, data_root):
    seqs = rec_columns[0]
    items = rec_columns[1]
    times = rec_columns[2]
    scores = rec_columns[3]
    rec_frame = pd.DataFrame({'seq': seqs, 'item': items, 'time': times, score_type: scores})
    rec_frame['item'] = rec_frame['item'] + 1
    print(rec_frame)
    rec_frame = np.array(rec_frame)
    if not os.path.exists(os.path.join(data_root, 'generated')):
        os.mkdir(os.path.join(data_root, 'generated'))
    #rec_frame.to_csv(os.path.join(export_root, 'generated', score_type + '.csv'))
    np.save(os.path.join(data_root, 'generated', score_type + '.npy'), rec_frame)
