from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np
import random


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items...')
        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])
            sampling_set = set(range(1, self.item_count + 1)) - seen
            assert len(sampling_set) >= self.sample_size, 'Not enough items to sample from.'
            samples = random.sample(sampling_set, self.sample_size)
            negative_samples[user] = samples
        return negative_samples
