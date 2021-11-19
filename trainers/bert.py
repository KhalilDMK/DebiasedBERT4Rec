from .base import AbstractTrainer
from .utils import metrics_for_ks, maximum_likelihood_estimation_loss
import torch.nn as nn
from utils import AverageMeterSet
from tqdm import tqdm
import pickle
from pathlib import Path
from dataloaders.utils import *
from trainers.utils import positional_frequency, avg_popularity, efd
import os
import json
from torch.utils.tensorboard import SummaryWriter
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from loggers import *
import sys


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, train_temporal_popularity, val_temporal_popularity, test_temporal_popularity, train_popularity, val_popularity, test_popularity, temporal_propensity, temporal_relevance, static_propensity):
        super().__init__(args, model, train_loader, val_loader, test_loader)

        self.max_len = args.bert_max_len
        if args.mode in ['train_bert_real', 'tune_bert_real', 'loop_bert_real']:
            self.preprocess_real_properties(train_temporal_popularity, val_temporal_popularity, test_temporal_popularity, train_popularity, val_popularity,
                                       test_popularity)
        if args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
            self.preprocess_semi_synthetic_properties(temporal_propensity, temporal_relevance, static_propensity)
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

                if self.args.mode in ['train_bert_real', 'tune_bert_real', 'loop_bert_real']:
                    metrics = self.calculate_metrics(batch, self.val_popularity_vector, self.val_temporal_popularity)
                if self.args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
                    metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                      ['AvgPop@%d' % k for k in self.metric_ks[:3]] + \
                                      ['EFD@%d' % k for k in self.metric_ks[:3]]
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

                if self.args.mode in ['train_bert_real', 'tune_bert_real', 'loop_bert_real']:
                    metrics = self.calculate_metrics(batch, self.test_popularity_vector, self.test_temporal_popularity)
                if self.args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
                    metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                      ['AvgPop@%d' % k for k in self.metric_ks[:3]] + \
                                      ['EFD@%d' % k for k in self.metric_ks[:3]]
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
        interacted_recommendations = None
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
                batch_recommendations, batch_interacted_recommendations = self.generate_batch_recommendations(seqs, scores)
                if interacted_recommendations == None:
                    interacted_recommendations = batch_interacted_recommendations
                else:
                    interacted_recommendations = torch.cat((interacted_recommendations, batch_interacted_recommendations))
                if recommendations == None:
                    recommendations = batch_recommendations
                else:
                    recommendations = torch.cat((recommendations, batch_recommendations))
        if self.args.mode in ['tune_bert_real', 'tune_bert_semi_synthetic', 'tune_tf']:
            with Path(self.export_root).joinpath('recommendations', 'rec_config_(' + str(self.args.bert_hidden_units) + ', ' + str(self.args.bert_num_blocks) + ', ' + str(self.args.bert_num_heads) + ', ' + str(self.args.bert_dropout) + ', ' + str(self.args.bert_mask_prob) + ', ' + str(self.args.skew_power) + ')_rep_' + str(self.args.rep) + '.pkl').open('wb') as f:
                pickle.dump(interacted_recommendations.tolist(), f)
        else:
            with Path(self.export_root).joinpath('recommendations', 'rec_iter_' + str(self.args.iteration) + '.pkl').open('wb') as f:
                pickle.dump(interacted_recommendations.tolist(), f)
        print('Interacted recommendations: ' + str(interacted_recommendations))
        return interacted_recommendations.to(self.device), recommendations.to(self.device)

    def final_data_eval_save_results(self, recommendations):
        if self.args.mode in ['train_bert_real', 'tune_bert_real', 'loop_bert_real']:
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
            avg_pop_recs_at_k = avg_popularity(recommendations, self.device, self.test_popularity_vector)
            efd_recs_at_k = efd(recommendations, self.device, self.test_popularity_vector)
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
            self.average_metrics['AvgPop_recs@' + str(self.args.top_k_recom)] = float(avg_pop_recs_at_k)
            self.average_metrics['EFD_recs@' + str(self.args.top_k_recom)] = float(efd_recs_at_k)
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
        indices, seqs, labels = batch
        batch_size = labels.shape[0]
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        logits = self.log_softmax(logits)
        if self.args.loss_debiasing == 'temporal_popularity':
            temp_pop_enc = self.train_temporal_popularity[
                labels.cpu(), list(range(self.args.bert_max_len)) * batch_size].view(-1, self.max_len)
            temp_pop_enc = temp_pop_enc.flatten()
            logits = torch.div(logits, torch.pow(temp_pop_enc.unsqueeze(1), self.args.skew_power))
        elif self.args.loss_debiasing == 'static_popularity':
            stat_pop_enc = self.train_popularity_vector[labels]
            logits = torch.div(logits, torch.pow(stat_pop_enc.unsqueeze(1), self.args.skew_power))
        elif self.args.loss_debiasing == 'relevance':
            relevance_enc = self.temporal_relevance[indices]
            relevance_enc = torch.transpose(relevance_enc, 1, 2)
            relevance_enc = relevance_enc.reshape(-1, relevance_enc.size(-1))
            labels = torch.where(labels > 0, relevance_enc[:, 1:].max(1)[1] + 1, 0)
            logits = logits * relevance_enc
        elif self.args.loss_debiasing == 'temporal_propensity':
            temp_prop_enc = self.temporal_propensity[indices]
            temp_prop_enc = torch.where(temp_prop_enc.double() > self.args.propensity_clipping, temp_prop_enc.double(), self.args.propensity_clipping)  # Propensity clipping
            temp_prop_enc = torch.transpose(temp_prop_enc, 1, 2)
            temp_prop_enc = temp_prop_enc.reshape(-1, temp_prop_enc.size(-1))
            logits = torch.div(logits, temp_prop_enc)
            #logits = logits[range(logits.shape[0]), labels.tolist()]
            #logits = logits[labels != 0]
            #loss = torch.min(torch.mean(- logits))
        elif self.args.loss_debiasing == 'static_propensity':
            stat_prop_enc = self.static_propensity[indices]
            stat_prop_enc = torch.where(stat_prop_enc.double() > self.args.propensity_clipping, stat_prop_enc.double(),
                                        self.args.propensity_clipping)  # Propensity clipping
            stat_prop_enc = stat_prop_enc.unsqueeze(1)
            stat_prop_enc = stat_prop_enc.repeat(1, self.args.bert_max_len, 1)
            stat_prop_enc = stat_prop_enc.reshape(-1, stat_prop_enc.size(-1))
            logits = torch.div(logits, stat_prop_enc)
        loss = self.nll(logits, labels)
        return loss

    def calculate_metrics(self, batch, popularity_vector=[], temporal_popularity=[]):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = metrics_for_ks(self.args, scores, labels, candidates, self.metric_ks, popularity_vector, temporal_popularity)
        return metrics

    def preprocess_real_properties(self, train_temporal_popularity, val_temporal_popularity, test_temporal_popularity, train_popularity, val_popularity, test_popularity):
        self.train_temporal_popularity = self.preprocess_temporal_popularity(train_temporal_popularity)
        self.val_temporal_popularity = self.preprocess_popularity_vector(val_temporal_popularity)
        self.test_temporal_popularity = self.preprocess_popularity_vector(test_temporal_popularity)
        self.train_popularity_vector = self.preprocess_popularity_vector(train_popularity)
        self.val_popularity_vector = self.preprocess_popularity_vector(val_popularity)
        self.test_popularity_vector = self.preprocess_popularity_vector(test_popularity)

    def preprocess_temporal_popularity(self, temporal_popularity):
        temporal_popularity = temporal_popularity.to(self.device)
        temporal_popularity = torch.cat((torch.zeros(1, self.max_len).to(self.device), temporal_popularity), 0) + sys.float_info.epsilon
        return temporal_popularity

    def preprocess_popularity_vector(self, popularity):
        popularity_vector = popularity.to(self.device)
        popularity_vector = torch.cat((torch.FloatTensor([0]).to(self.device), popularity_vector)) + sys.float_info.epsilon
        return popularity_vector

    def preprocess_semi_synthetic_properties(self, temporal_propensity, temporal_relevance, static_propensity):
        self.temporal_propensity = torch.cat((torch.zeros(temporal_propensity.shape[0], 1, temporal_propensity.shape[2]).to(self.device), temporal_propensity.to(self.device)), 1) + sys.float_info.epsilon
        self.temporal_relevance = torch.cat((torch.zeros(temporal_relevance.shape[0], 1, temporal_relevance.shape[2]).to(self.device), temporal_relevance.to(self.device)), 1)
        self.static_propensity = torch.cat((torch.zeros(static_propensity.shape[0], 1).to(self.device), static_propensity.to(self.device)), 1) + sys.float_info.epsilon
