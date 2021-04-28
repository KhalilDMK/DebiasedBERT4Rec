from abc import *
from pathlib import Path
import pickle


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, exposure):
        self.train = train
        self.val = val
        self.test = test
        self.exposure = exposure
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        negative_samples = self.generate_negative_samples()
        return negative_samples
