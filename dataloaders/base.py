from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.theta, self.gamma, self.exposure = None, None, None
        if args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
            self.theta = dataset['theta']
            self.gamma = dataset['gamma']
            self.exposure = dataset['exposure']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.args.user_count = self.user_count
        self.args.item_count = self.item_count

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
