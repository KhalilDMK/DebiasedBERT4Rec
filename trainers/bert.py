from .base import AbstractTrainer
from .utils import metrics_for_ks

import torch
import torch.nn as nn
import sys


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, pos_dist, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, pos_dist, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader)
        #self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.nll = nn.NLLLoss(ignore_index=0)
        self.log_softmax = nn.LogSoftmax(dim=1)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch, model_code):
        seqs, labels = batch
        #temp_prop_enc = torch.cat(
        #    [self.pos_dist[seqs[i].cpu(), range(seqs.shape[1])].view(-1, self.max_len) for i in range(seqs.shape[0])], 0)
        #temp_prop_enc = torch.cat(
        #    [self.pos_dist[labels[i].cpu(), range(labels.shape[1])].view(-1, self.max_len) for i in range(labels.shape[0])],
        #    0)
        temp_prop_enc = self.pos_dist[labels.flatten().cpu(), list(range(labels.shape[1])) * labels.shape[0]].view(-1, self.max_len)
        temp_prop_enc = temp_prop_enc.flatten()
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        stat_prop_enc = self.train_popularity_vector[labels]
        #loss = self.ce(logits, labels)
        logits = self.log_softmax(logits)
        if self.args.loss_debiasing in ['temporal', 'exposure']:
            print('loss_temporal')
            logits = torch.div(logits, temp_prop_enc.unsqueeze(1))
        if self.args.loss_debiasing in ['static', 'exposure']:
            print('loss_static')
            logits = torch.div(logits, stat_prop_enc.unsqueeze(1))
        loss = self.nll(logits, labels)

        return loss

    def calculate_metrics(self, batch, popularity_vector, item_similarity_matrix):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = metrics_for_ks(scores, labels, self.metric_ks, popularity_vector, item_similarity_matrix)
        return metrics
