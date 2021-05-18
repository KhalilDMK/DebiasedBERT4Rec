from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
import torch
import torch.utils.data as data_utils
import numpy as np
from collections import Counter
import pandas as pd


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed, self.exposure)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed, self.exposure)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def get_semi_synthetic_properties(self):
        temporal_propensity = self.get_temporal_propensity()
        temporal_relevance = self.get_temporal_relevance()
        static_propensity = self.get_static_propensity()
        return temporal_propensity, temporal_relevance, static_propensity

    def get_real_properties(self):
        train_temporal_popularity = self.get_train_temporal_popularity()
        train_popularity = self.get_popularity(include_test=False)
        val_popularity = self.get_popularity(include_test=True, mode='val')
        test_popularity = self.get_popularity(include_test=True, mode='test')
        return train_temporal_popularity, train_popularity, val_popularity, test_popularity

    def get_train_temporal_popularity(self):
        item_count = self.args.item_count
        u2seq = self.train
        max_len = self.max_len
        u2seq = [x[-max_len:] for x in u2seq.values()]
        seq = np.zeros([len(u2seq), len(max(u2seq, key=lambda x: len(x)))])
        for i, j in enumerate(u2seq):
            seq[i][-len(j)::] = j
        if seq.shape[1] < max_len:
            seq = np.concatenate((np.zeros((seq.shape[0], max_len - seq.shape[1])), seq), axis=1)
        occurrences = [Counter(seq[:, j]) for j in range(max_len)]
        item_index = pd.Index(range(item_count + 1))
        occurrences = pd.DataFrame(occurrences).transpose().reindex(item_index).sort_index().fillna(0).values[1::]
        occurrences = torch.Tensor(occurrences)
        train_temporal_popularity = occurrences / torch.sum(occurrences)
        return train_temporal_popularity

    def get_popularity(self, include_test=False, mode='test'):
        answers = self.val if mode == 'val' else self.test
        if include_test:
            print('Generating ' + mode + ' popularity vector...')
        else:
            print('Generating train popularity vector...')
        popularity_vector = self.get_popularity_vector(answers, include_test)
        return popularity_vector

    def get_popularity_vector(self, answers, include_test):
        item_count = self.args.item_count
        u2seq = self.train
        max_len = self.max_len
        u2seq = [x[-max_len:] for x in u2seq.values()]
        seq = np.zeros([len(u2seq), len(max(u2seq, key=lambda x: len(x)))])
        for i, j in enumerate(u2seq):
            seq[i][-len(j)::] = j
        if seq.shape[1] < max_len:
            seq = np.concatenate((np.zeros((seq.shape[0], max_len - seq.shape[1])), seq), axis=1)
        if include_test:
            seq = np.append(seq, np.array(list(answers.values())), axis=1)
        occurrences = [Counter(seq[:, j]) for j in range(seq.shape[1])]
        item_index = pd.Index(range(item_count + 1))
        occurrences = pd.DataFrame(occurrences).transpose().reindex(item_index).sort_index().fillna(0).values[1::]
        occurrences = torch.Tensor(occurrences)
        popularity = torch.sum(occurrences / torch.sum(occurrences), dim=1)
        return popularity

    def get_temporal_propensity(self):
        return torch.Tensor(self.theta[:, :, -self.args.bert_max_len:])

    def get_temporal_relevance(self):
        return torch.Tensor(self.gamma[:, :, -self.args.bert_max_len:])

    def get_static_propensity(self):
        return torch.Tensor(self.theta[:, :, -self.args.bert_max_len:]).mean(-1)

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, item_count, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.item_count = item_count
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                tokens.append(self.mask_token)
                #prob /= self.mask_prob
                #if prob < 0.8:
                #    tokens.append(self.mask_token)
                #elif prob < 0.9:
                #    tokens.append(self.rng.randint(1, self.item_count))
                #else:
                #    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        if 0 in tokens:
            print(True)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return index, torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
