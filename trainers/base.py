from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
from dataloaders.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
from trainers.utils import positional_frequency, top_position_matching
import pickle
import random
import sys


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, pos_dist, train_popularity_vector_loader, val_popularity_vector_loader, test_popularity_vector_loader):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.max_len = args.bert_max_len
        self.pos_dist = pos_dist.to(self.device)
        self.pos_dist = torch.cat((torch.zeros(1, self.max_len).to(self.device), self.pos_dist), 0) + sys.float_info.epsilon
        self.train_popularity_vector = train_popularity_vector_loader.popularity_vector.to(self.device)
        self.train_popularity_vector = torch.cat((torch.FloatTensor([0]).to(self.device), self.train_popularity_vector)) + sys.float_info.epsilon
        self.train_item_similarity_matrix = train_popularity_vector_loader.item_similarity_matrix.to(self.device)
        self.train_item_similarity_matrix = torch.cat((torch.zeros(1, self.train_item_similarity_matrix.shape[1]).to(self.device), self.train_item_similarity_matrix), 0)
        self.train_item_similarity_matrix = torch.cat((torch.zeros(self.train_item_similarity_matrix.shape[0], 1).to(self.device), self.train_item_similarity_matrix), 1)
        self.val_popularity_vector = val_popularity_vector_loader.popularity_vector.to(self.device)
        self.val_popularity_vector = torch.cat(
            (torch.FloatTensor([0]).to(self.device), self.val_popularity_vector)) + sys.float_info.epsilon
        self.val_item_similarity_matrix = val_popularity_vector_loader.item_similarity_matrix.to(self.device)
        self.val_item_similarity_matrix = torch.cat((torch.zeros(1, self.val_item_similarity_matrix.shape[1]).to(
            self.device), self.val_item_similarity_matrix), 0)
        self.val_item_similarity_matrix = torch.cat((torch.zeros(self.val_item_similarity_matrix.shape[0], 1).to(
            self.device), self.val_item_similarity_matrix), 1)
        self.test_popularity_vector = test_popularity_vector_loader.popularity_vector.to(self.device)
        self.test_popularity_vector = torch.cat(
            (torch.FloatTensor([0]).to(self.device), self.test_popularity_vector)) + sys.float_info.epsilon
        self.test_item_similarity_matrix = test_popularity_vector_loader.item_similarity_matrix.to(self.device)
        self.test_item_similarity_matrix = torch.cat((torch.zeros(1, self.test_item_similarity_matrix.shape[1]).to(
            self.device), self.test_item_similarity_matrix), 0)
        self.test_item_similarity_matrix = torch.cat((torch.zeros(self.test_item_similarity_matrix.shape[0], 1).to(
            self.device), self.test_item_similarity_matrix), 1)

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch, popularity_vector, item_similarity_matrix):
        pass

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch, self.args.model_code)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

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
        print('Testing best model on test set...')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
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
            #with open(os.path.join(self.export_root, 'logs', 'test_metrics_iter_' + str(self.args.iteration) + '.json'), 'w') as f:
            #    json.dump(self.average_metrics, f, indent=4)
            print(self.average_metrics)

    def recommend(self):
        print('Generating recommendations for test sessions...')
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()
        recommendations = None
        recommendation_positions = None
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
                #batch_recommendations = torch.argmax(scores, dim=1)
                batch_recommendations = self.generate_batch_recommendations(seqs, scores)
                if recommendations == None:
                    recommendations = batch_recommendations
                    recommendation_positions = torch.Tensor([len([i for i in x if i != 0]) - 1 for x in seqs])
                else:
                    recommendations = torch.cat((recommendations, batch_recommendations))
                    recommendation_positions = torch.cat((recommendation_positions, torch.tensor([len([i for i in x if i != 0]) - 1 for x in seqs])))
        with Path(self.export_root).joinpath('recommendations', 'rec_iter_' + str(self.args.iteration) + '.pkl').open('wb') as f:
            pickle.dump(recommendations.tolist(), f)
        print('recommendations: ' + str(recommendations))
        print('recommendation positions: ' + str(recommendation_positions))
        return recommendations.to(self.device), recommendation_positions.to(self.device)

    def eval_position_bias(self, recommendations, recommendation_positions):
        print('Position Bias Evaluation...')
        test_items = None
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                seqs, candidates, labels = batch
                if test_items == None:
                    test_items = candidates[:, 0]
                else:
                    test_items = torch.cat((test_items, candidates[:, 0]))
        temp_prop_rec = positional_frequency(recommendations.cpu().numpy(), recommendation_positions.cpu().numpy(),
                                 self.pos_dist.cpu().numpy())
        temp_prop_test = positional_frequency(test_items.cpu().numpy(), recommendation_positions.cpu().numpy(),
                                 self.pos_dist.cpu().numpy())
        model_temp_prop_bias = temp_prop_rec - temp_prop_test
        data_temp_prop_bias = position_bias_in_data(self.pos_dist)
        data_stat_prop_bias = propensity_bias_in_data(self.train_popularity_vector)
        data_stat_prop_bias_kl_p_u = propensity_bias_in_data_kl_p_u(self.train_popularity_vector)
        data_stat_prop_bias_kl_u_p = propensity_bias_in_data_kl_u_p(self.train_popularity_vector)
        data_stat_prop_bias_mse = propensity_bias_in_data_mse(self.train_popularity_vector)
        data_stat_prop_bias_mae = propensity_bias_in_data_mae(self.train_popularity_vector)
        data_temp_expo_bias = temporal_exposure_bias_in_data(self.pos_dist)
        data_temp_expo_bias_kl_p_u = temporal_exposure_bias_in_data_kl_p_u(self.pos_dist)
        data_temp_expo_bias_kl_u_p = temporal_exposure_bias_in_data_kl_u_p(self.pos_dist)
        data_temp_expo_bias_mse = temporal_exposure_bias_in_data_mse(self.pos_dist)
        data_temp_expo_bias_mae = temporal_exposure_bias_in_data_mae(self.pos_dist)
        self.average_metrics['temp_prop_rec'] = float(temp_prop_rec)
        self.average_metrics['temp_prop_test'] = float(temp_prop_test)
        self.average_metrics['model_temp_prop_bias'] = float(model_temp_prop_bias)
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
        print('Average training positional frequency of recommendations: ' + str(temp_prop_rec))
        print('Average training positional frequency of test items: ' + str(temp_prop_test))
        print('Model temporal propensity bias: ' + str(model_temp_prop_bias))
        #print('Average Position Matching of Recommendations with Top 1 Training Positions: ' + str(
        #    top_position_matching(recommendations.cpu().numpy(), recommendation_positions.cpu().numpy(),
        #                          position_distributions[:].cpu().numpy())))
        with open(os.path.join(self.export_root, 'logs', 'test_metrics_iter_' + str(self.args.iteration) + '.json'),
                  'w') as f:
            json.dump(self.average_metrics, f, indent=4)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

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

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0

    def get_indices_to_ignore(self, seqs, num_items):
        train_batch_indices = torch.cat([torch.Tensor([i] * self.max_len) for i in range(seqs.shape[0])]).tolist()
        train_item_indices = seqs.flatten().tolist()
        train_batch_indices = [int(x) for x in train_batch_indices]
        train_item_indices = [0 if x == num_items else int(x) for x in train_item_indices]
        return train_batch_indices, train_item_indices

    def generate_batch_recommendations(self, seqs, scores):
        batch_recommendations = torch.topk(scores, k=self.args.top_k_recom, dim=1)
        batch_recommendations = batch_recommendations[1]
        batch_indices = range(seqs.shape[0])
        item_random_indices = random.choices(range(self.args.top_k_recom), k=seqs.shape[0])
        batch_recommendations = batch_recommendations[batch_indices, item_random_indices]
        return batch_recommendations
