from .base import AbstractTrainer
from .utils import metrics_for_ks
import torch.nn as nn
from utils import AverageMeterSet
from tqdm import tqdm
import pickle
from pathlib import Path
from dataloaders.utils import *
from trainers.utils import positional_frequency
import os
import json
from torch.utils.tensorboard import SummaryWriter
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from loggers import *


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, train_temporal_popularity, train_popularity_loader, val_popularity_loader, test_popularity_loader):
        super().__init__(args, model, train_loader, val_loader, test_loader)

        self.max_len = args.bert_max_len
        self.preprocess_real_properties(train_temporal_popularity, train_popularity_loader, val_popularity_loader,
                                   test_popularity_loader)
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

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, self.val_popularity_vector, self.val_item_similarity_matrix)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                      ['AvgPop@%d' % k for k in self.metric_ks[:3]] + \
                                      ['EFD@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Diversity@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)

    def test(self):
        if self.args.mode == 'tune':
            print('Testing best model on validation set...')
        else:
            print('Testing best model on test set...')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            if self.args.mode == 'tune':
                tqdm_dataloader = tqdm(self.val_loader)
            else:
                tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, self.test_popularity_vector, self.test_item_similarity_matrix)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                      ['AvgPop@%d' % k for k in self.metric_ks[:3]] + \
                                      ['EFD@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Diversity@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            self.average_metrics = average_meter_set.averages()
            print(self.average_metrics)

    def recommend(self):
        print('Generating recommendations for test sessions...')
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()
        recommendations = None
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                seqs, candidates, labels = batch
                scores = self.model(seqs)  # B x T x V
                scores = scores[:, -1, :]  # B x V
                num_items = scores.shape[1]
                train_batch_indices, train_item_indices = self.get_indices_to_ignore(seqs, num_items)
                scores = softmax(scores)
                scores[train_batch_indices, train_item_indices] = float('-inf')
                batch_recommendations = self.generate_batch_recommendations(seqs, scores)
                if recommendations == None:
                    recommendations = batch_recommendations
                else:
                    recommendations = torch.cat((recommendations, batch_recommendations))
        if self.args.mode in ['tune_bert_real', 'tune_bert_semi_synthetic', 'tune_tf']:
            with Path(self.export_root).joinpath('recommendations', 'rec_config_(' + str(self.args.bert_hidden_units) + ', ' + str(self.args.bert_num_blocks) + ', ' + str(self.args.bert_num_heads) + ', ' + str(self.args.bert_dropout) + ', ' + str(self.args.bert_mask_prob) + ', ' + str(self.args.skew_power) + ')_rep_' + str(self.args.rep) + '.pkl').open('wb') as f:
                pickle.dump(recommendations.tolist(), f)
        else:
            with Path(self.export_root).joinpath('recommendations', 'rec_iter_' + str(self.args.iteration) + '.pkl').open('wb') as f:
                pickle.dump(recommendations.tolist(), f)
        print('recommendations: ' + str(recommendations))
        return recommendations.to(self.device)

    def final_data_eval_save_results(self):
        data_temp_prop_bias = position_bias_in_data(self.train_temporal_popularity)
        data_stat_prop_bias = propensity_bias_in_data(self.train_popularity_vector)
        data_stat_prop_bias_kl_p_u = propensity_bias_in_data_kl_p_u(self.train_popularity_vector)
        data_stat_prop_bias_kl_u_p = propensity_bias_in_data_kl_u_p(self.train_popularity_vector)
        data_stat_prop_bias_mse = propensity_bias_in_data_mse(self.train_popularity_vector)
        data_stat_prop_bias_mae = propensity_bias_in_data_mae(self.train_popularity_vector)
        data_temp_expo_bias = temporal_exposure_bias_in_data(self.train_temporal_popularity)
        data_temp_expo_bias_kl_p_u = temporal_exposure_bias_in_data_kl_p_u(self.train_temporal_popularity)
        data_temp_expo_bias_kl_u_p = temporal_exposure_bias_in_data_kl_u_p(self.train_temporal_popularity)
        data_temp_expo_bias_mse = temporal_exposure_bias_in_data_mse(self.train_temporal_popularity)
        data_temp_expo_bias_mae = temporal_exposure_bias_in_data_mae(self.train_temporal_popularity)
        ips_bias_condition = bias_relaxed_condition(self.train_temporal_popularity)
        self.average_metrics['data_temp_prop_bias'] = float(data_temp_prop_bias)
        self.average_metrics['data_stat_prop_bias'] = float(data_stat_prop_bias)
        self.average_metrics['data_stat_prop_bias_kl_p_u'] = float(data_stat_prop_bias_kl_p_u)
        self.average_metrics['data_stat_prop_bias_kl_u_p'] = float(data_stat_prop_bias_kl_u_p)
        self.average_metrics['data_stat_prop_bias_mse'] = float(data_stat_prop_bias_mse)
        self.average_metrics['data_stat_prop_bias_mae'] = float(data_stat_prop_bias_mae)
        self.average_metrics['data_temp_expo_bias'] = float(data_temp_expo_bias)
        self.average_metrics['data_temp_expo_bias_kl_p_u'] = float(data_temp_expo_bias_kl_p_u)
        self.average_metrics['data_temp_expo_bias_kl_u_p'] = float(data_temp_expo_bias_kl_u_p)
        self.average_metrics['data_temp_expo_bias_mse'] = float(data_temp_expo_bias_mse)
        self.average_metrics['data_temp_expo_bias_mae'] = float(data_temp_expo_bias_mae)
        self.average_metrics['ips_bias_condition'] = float(ips_bias_condition)
        if self.args.mode in ['tune_bert_real', 'tune_bert_semi_synthetic', 'tune_tf']:
            with open(os.path.join(self.export_root, 'logs', 'test_metrics_config_(' + str(self.args.bert_hidden_units) + ', ' + str(self.args.bert_num_blocks) + ', ' + str(self.args.bert_num_heads) + ', ' + str(self.args.train_batch_size) + ', ' + str(self.args.bert_dropout) + ', ' + str(self.args.bert_mask_prob) + ', ' + str(self.args.skew_power) + ')_rep_' + str(self.args.rep) + '.json'), 'w') as f:
                json.dump(self.average_metrics, f, indent=4)
        else:
            with open(os.path.join(self.export_root, 'logs', 'test_metrics_iter_' + str(self.args.iteration) + '.json'),
                      'w') as f:
                json.dump(self.average_metrics, f, indent=4)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def calculate_loss(self, batch, model_code):
        seqs, labels = batch
        batch_size = labels.shape[0]
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        logits = self.log_softmax(logits)
        if self.args.loss_debiasing == 'temporal_popularity':
            temp_prop_enc = self.train_temporal_popularity[
                labels.cpu(), list(range(self.args.bert_max_len)) * batch_size].view(-1, self.max_len)
            temp_prop_enc = temp_prop_enc.flatten()
            logits = torch.div(logits, torch.pow(temp_prop_enc.unsqueeze(1), self.args.skew_power))
        elif self.args.loss_debiasing == 'static_popularity':
            stat_prop_enc = self.train_popularity_vector[labels]
            logits = torch.div(logits, torch.pow(stat_prop_enc.unsqueeze(1), self.args.skew_power))
        loss = self.nll(logits, labels)

        return loss

    def calculate_metrics(self, batch, popularity_vector, item_similarity_matrix):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = metrics_for_ks(scores, labels, candidates, self.metric_ks, popularity_vector, item_similarity_matrix)
        return metrics

    def preprocess_real_properties(self, train_temporal_popularity, train_popularity_loader, val_popularity_loader, test_popularity_loader):
        self.train_temporal_popularity = self.preprocess_temporal_popularity(train_temporal_popularity)
        self.train_popularity_vector = self.preprocess_popularity_vector(train_popularity_loader)
        self.train_item_similarity_matrix = self.preprocess_item_similarity_matrix(train_popularity_loader)
        self.val_popularity_vector = self.preprocess_popularity_vector(val_popularity_loader)
        self.val_item_similarity_matrix = self.preprocess_item_similarity_matrix(val_popularity_loader)
        self.test_popularity_vector = self.preprocess_popularity_vector(test_popularity_loader)
        self.test_item_similarity_matrix = self.preprocess_item_similarity_matrix(test_popularity_loader)

    def preprocess_temporal_popularity(self, temporal_popularity):
        temporal_popularity = temporal_popularity.to(self.device)
        temporal_popularity = torch.cat((torch.zeros(1, self.max_len).to(self.device), temporal_popularity), 0) + sys.float_info.epsilon
        return temporal_popularity

    def preprocess_popularity_vector(self, popularity_loader):
        popularity_vector = popularity_loader.popularity_vector.to(self.device)
        popularity_vector = torch.cat((torch.FloatTensor([0]).to(self.device), popularity_vector)) + sys.float_info.epsilon
        return popularity_vector

    def preprocess_item_similarity_matrix(self, popularity_loader):
        item_similarity_matrix = popularity_loader.item_similarity_matrix.to(self.device)
        item_similarity_matrix = torch.cat((torch.zeros(1, item_similarity_matrix.shape[1]).to(self.device), item_similarity_matrix), 0)
        item_similarity_matrix = torch.cat((torch.zeros(item_similarity_matrix.shape[0], 1).to(self.device), item_similarity_matrix), 1)
        return item_similarity_matrix
