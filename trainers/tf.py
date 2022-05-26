from .base import AbstractTrainer
from .utils import metrics_for_ks_explicit, save_reconstructed
from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path


class TFTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, train_temporal_popularity, val_temporal_popularity, test_temporal_popularity, train_popularity, val_popularity, test_popularity, temporal_propensity, temporal_relevance, static_propensity):
        super().__init__(args, model, train_loader, val_loader, test_loader)

        self.mse = nn.MSELoss()
        self.ce = nn.BCELoss()

    @classmethod
    def code(cls):
        return 'tf'

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

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if self.args.tf_target == 'exposure':
                    description_metrics = ['AUC']
                else:
                    description_metrics = ['MSE']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
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

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if self.args.tf_target == 'exposure':
                    description_metrics = ['AUC']
                else:
                    description_metrics = ['MSE']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            self.average_metrics = average_meter_set.averages()
            print(self.average_metrics)

    def save_test_performance(self):
        print('Saving test performance...')
        if self.args.mode in ['tune_bert_real', 'tune_bert_semi_synthetic', 'tune_tf']:
            with open(os.path.join(self.export_root, 'logs', 'test_metrics_config_(' + str(self.args.bert_hidden_units) + ', ' + str(self.args.train_batch_size) + ')_rep_' + str(self.args.rep) + '.json'), 'w') as f:
                json.dump(self.average_metrics, f, indent=4)
        else:
            with open(os.path.join(self.export_root, 'logs', 'test_metrics_iter_' + str(self.args.iteration) + '.json'),
                      'w') as f:
                json.dump(self.average_metrics, f, indent=4)

    def reconstruct(self, gen_loader):
        print('Generate ' + self.args.tf_target + ' tensor...')
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()
        with torch.no_grad():
            tqdm_dataloader = tqdm(gen_loader)
            gen_seqs, gen_items, gen_times, gen_score = torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                seqs, items, times = batch
                scores = self.model(seqs, items, times)
                if self.args.tf_target == 'exposure':
                    scores = torch.sigmoid(scores)
                gen_seqs = torch.cat((gen_seqs, seqs.float()))
                gen_items = torch.cat((gen_items, items.float()))
                gen_times = torch.cat((gen_times, times.float()))
                gen_score = torch.cat((gen_score, scores))
        if self.args.tf_target == 'exposure':
            score_type = 'interaction'
        elif self.args.tf_target == 'relevance':
            score_type = 'rating'
        save_reconstructed([gen_seqs.tolist(), gen_items.tolist(), gen_times.tolist(), gen_score.tolist()], score_type, self.args.data_root)

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
        if self.args.tf_target == 'exposure':
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric, higher_better=True))
        else:
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric, higher_better=False))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def calculate_loss(self, batch, model_code):
        seqs, items, times, ratings = batch
        predictions = self.model(seqs, items, times)
        ratings = ratings.float()
        if self.args.tf_target == 'exposure':
            loss = self.ce(torch.sigmoid(predictions), ratings)
        else:
            loss = self.mse(predictions, ratings)
        return loss

    def calculate_metrics(self, batch):
        seqs, items, times, ratings = batch
        scores = self.model(seqs, items, times)
        if self.args.tf_target == 'exposure':
            scores = torch.sigmoid(scores)
        metrics = metrics_for_ks_explicit(scores, ratings, self.args.tf_target)
        return metrics
