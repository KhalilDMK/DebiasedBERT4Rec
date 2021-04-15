from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
from .utils import position_bias_in_data, propensity_bias_in_data

import torch
import torch.utils.data as data_utils
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed)
        #                                                 , self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed)
        #                                                , self.save_folder)

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

    def get_train_position_distributions_dataloader(self):
        position_distribution_loader = BertPositionDistribution(self.train, self.max_len, self.args.num_items)
        #pos_bias_in_data = position_bias_in_data(position_distribution_loader.position_distributions)
        #print('Temporal propensity bias in data: ' + str(pos_bias_in_data))
        return position_distribution_loader.position_distributions

    def get_popularity_vector_dataloader(self, include_test=False, mode='test'):
        answers = self.val if mode == 'val' else self.test
        if include_test:
            print('Generating ' + mode + ' popularity vector...')
        else:
            print('Generating train popularity vector...')
        popularity_vector_loader = BertPopularityVector(self.train, answers, self.user_count, self.item_count, self.max_len, include_test)
        #if not include_test:
        #    prop_bias_in_data = propensity_bias_in_data(popularity_vector_loader.popularity_vector)
        #    print('Propensity bias in data: ' + str(prop_bias_in_data))
        return popularity_vector_loader

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
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        #self.shuffle_train(seq)

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
                #    tokens.append(self.rng.randint(1, self.num_items))
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


        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

    #def shuffle_train(self, seq):
    #    seq_lengths = sorted(set([len(x) for x in seq]))
    #    #for length in seq_lengths:
    #    print(seq_lengths)


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


class BertPositionDistribution(data_utils.Dataset):
    def __init__(self, u2seq, max_len, num_items):
        self.num_items = num_items
        self.u2seq = u2seq
        self.items = list(range(1, self.num_items + 1))
        self.max_len = max_len
        self.u2seq = [x[-self.max_len:] for x in self.u2seq.values()]
        self.seq = np.zeros([len(self.u2seq), len(max(self.u2seq, key=lambda x: len(x)))])
        for i, j in enumerate(self.u2seq):
            #self.seq[i][0:len(j)] = j
            self.seq[i][-len(j)::] = j
        occurrences = [Counter(self.seq[:, j]) for j in range(self.max_len)]
        item_index = pd.Index(range(self.num_items + 1))
        occurrences = pd.DataFrame(occurrences).transpose().reindex(item_index).sort_index().fillna(0).values[1::]
        #occurrences = np.concatenate((np.zeros((1, self.max_len)), occurrences), axis=0)
        softmax = torch.nn.Softmax()
        occurrences = torch.Tensor(occurrences)
        #self.position_distributions = torch.pow(occurrences / torch.max(occurrences, dim=0)[0], 0.5)
        #self.position_distributions = occurrences / torch.max(occurrences, dim=0)[0]
        self.position_distributions = occurrences / torch.sum(occurrences)
        #self.position_distributions = torch.pow(occurrences / torch.max(occurrences), 0.5)
        #self.position_distributions = softmax(self.position_distributions.flatten()).view_as(self.position_distributions)
        #self.position_distributions = softmax(self.position_distributions)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.position_distributions[index]


class BertPopularityVector(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, user_count, item_count, max_len, include_test=False):
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.u2seq = [x[-max_len:] for x in self.u2seq.values()]
        self.seq = np.zeros([len(self.u2seq), len(max(self.u2seq, key=lambda x: len(x)))])
        for i, j in enumerate(self.u2seq):
            #self.seq[i][0:len(j)] = j
            self.seq[i][-len(j)::] = j
        self.seq = {i: list(self.seq[i]) for i in range(len(self.seq))}
        if include_test:
            self.seq = {k:self.seq[k] + self.u2answer[k] for k in self.seq}
        users = np.array([x for x in self.seq.keys() for i in range(len(self.seq[x]))])
        items = np.array([int(x) for y in self.seq.values() for x in y])
        self.interaction_matrix = pd.crosstab(users, items)
        del self.interaction_matrix[0]
        missing_columns = list(set(range(1, item_count + 1)) - set(list(self.interaction_matrix)))
        #missing_columns = list(set(range(item_count + 1)) - set(list(self.interaction_matrix)))
        missing_rows = list(set(range(user_count)) - set(self.interaction_matrix.index))
        for missing_column in missing_columns:
            self.interaction_matrix[missing_column] = [0] * len(self.interaction_matrix)
        for missing_row in missing_rows:
            self.interaction_matrix.loc[missing_row] = [0] * item_count
            #self.interaction_matrix.loc[missing_row] = [0] * (item_count + 1)
        self.interaction_matrix = np.array(self.interaction_matrix[list(range(1, item_count + 1))].sort_index())
        #self.interaction_matrix = np.array(self.interaction_matrix[list(range(item_count + 1))].sort_index())
        self.popularity_vector = np.sum(self.interaction_matrix, axis=0)
        #self.popularity_vector = (self.popularity_vector / max(self.popularity_vector)) ** 0.5
        softmax = torch.nn.Softmax()
        #self.popularity_vector = softmax(torch.Tensor(self.popularity_vector))
        self.popularity_vector = torch.Tensor(self.popularity_vector / sum(self.popularity_vector))
        #self.popularity_vector = torch.Tensor((self.popularity_vector / max(self.popularity_vector)) ** 0.5)
        #self.popularity_vector = softmax(self.popularity_vector)
        self.item_similarity_matrix = cosine_similarity(self.interaction_matrix.T)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        #self.popularity_vector = torch.Tensor(self.popularity_vector)
        self.item_similarity_matrix = torch.Tensor(self.item_similarity_matrix)

    def __len__(self):
        return len(self.popularity_vector)

    def __getitem__(self, index):
        return self.popularity_vector[index]
