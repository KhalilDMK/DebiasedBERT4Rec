from .base import AbstractDataloader
import torch
import torch.utils.data as data_utils


class TFDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.bert_max_len

    @classmethod
    def code(cls):
        return 'tf'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = TFTrainDataset(self.train)
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

    def get_gen_loader(self):
        batch_size = self.args.test_batch_size
        dataset = self._get_gen_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        eval = self.val if mode == 'val' else self.test
        dataset = TFEvalDataset(eval)
        return dataset

    def _get_gen_dataset(self):
        dataset = TFGenDataset(self.args)
        return dataset


class TFTrainDataset(data_utils.Dataset):
    def __init__(self, train):
        self.train_seqs = torch.LongTensor(train[0])
        self.train_items = torch.LongTensor(train[1])
        self.train_times = torch.LongTensor(train[2])
        self.train_ratings = torch.LongTensor(train[3])

    def __len__(self):
        return len(self.train_seqs)

    def __getitem__(self, index):
        return self.train_seqs[index], self.train_items[index], self.train_times[index], self.train_ratings[index]


class TFEvalDataset(data_utils.Dataset):
    def __init__(self, eval):
        self.eval_seqs = torch.LongTensor(eval[0])
        self.eval_items = torch.LongTensor(eval[1])
        self.eval_times = torch.LongTensor(eval[2])
        self.eval_ratings = torch.LongTensor(eval[3])

    def __len__(self):
        return len(self.eval_seqs)

    def __getitem__(self, index):
        return self.eval_seqs[index], self.eval_items[index], self.eval_times[index], self.eval_ratings[index]


class TFGenDataset(data_utils.Dataset):
    def __init__(self, args):
        self.gen_seqs = torch.LongTensor(list(range(args.user_count)))
        self.gen_items = torch.LongTensor(list(range(args.item_count)))
        self.gen_times = torch.LongTensor(list(range(args.bert_max_len)))
        cartesian = torch.cartesian_prod(self.gen_seqs, self.gen_items, self.gen_times)
        self.gen_seqs = cartesian[:, 0]
        self.gen_items = cartesian[:, 1]
        self.gen_times = cartesian[:, 2]

    def __len__(self):
        return len(self.gen_seqs)

    def __getitem__(self, index):
        return self.gen_seqs[index], self.gen_items[index], self.gen_times[index]
