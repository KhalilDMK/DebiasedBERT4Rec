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
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

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
        position_distributions = BertPositionDistribution(self.train, self.max_len, self.args.num_items)
        return position_distributions

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
        print(self.mask_token)
        self.num_items = num_items
        print(self.num_items)
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        #self.shuffle_train(seq)

        tokens = []
        labels = []
        #print(seq)
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                #elif prob < 0.9:
                #    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

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

        #print(tokens)

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
        #print(sum([self.num_items in x for x in self.u2seq.values()]))
        self.u2seq = [x[-self.max_len:] for x in self.u2seq.values()]
        #print(sum([self.num_items in x for x in self.u2seq]))
        self.seq = np.zeros([len(self.u2seq), len(max(self.u2seq, key=lambda x: len(x)))])
        for i, j in enumerate(self.u2seq):
            self.seq[i][0:len(j)] = j
        #print(self.seq)
        self.occurrences = [Counter(self.seq[:, j]) for j in range(self.max_len)]
        #print([self.num_items in x for x in self.occurrences])
        item_index = pd.Index(range(self.num_items + 1))
        #print(pd.DataFrame(self.occurrences).transpose().reindex(item_index).sort_index().fillna(0))
        self.occurrences = pd.DataFrame(self.occurrences).transpose().reindex(item_index).sort_index().fillna(0).values

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item_pos_dist = self.occurrences[index]
        softmax = torch.nn.Softmax()

        return softmax(torch.Tensor(item_pos_dist))

