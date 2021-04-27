from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.semi_synthetic = 'real'
        if args.dataloader_code == 'tf':
            self.semi_synthetic = 'generate'
        elif args.mode == 'train_semi_synthetic':
            self.semi_synthetic = 'train'
        #self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset(self.semi_synthetic)
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.theta, self.gamma, self.exposure = None, None, None
        if args.mode == 'train_semi_synthetic':
            self.theta = dataset['theta']
            self.gamma = dataset['gamma']
            self.exposure = dataset['exposure']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
