from .base import AbstractNegativeSampler
from tqdm import trange
from collections import Counter
import numpy as np


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        pop_distribution = self.popularity_distribution()

        negative_samples = {}
        print('Sampling negative items...')
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            sampling_set = set(range(1, self.item_count + 1)) - seen
            assert len(sampling_set) >= self.sample_size, 'Not enough items to sample from.'
            user_popularity_distribution = [pop_distribution[i] for i in range(len(pop_distribution)) if i + 1 in sampling_set]
            sum_probabilities = sum(user_popularity_distribution)
            user_popularity_distribution = [x / sum_probabilities for x in user_popularity_distribution]
            samples = np.random.choice(list(sampling_set), size=self.sample_size, replace=False, p=user_popularity_distribution)
            negative_samples[user] = list(samples)
        return negative_samples

    def popularity_distribution(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        num_interactions = sum(popularity.values())
        popularity = {k: v / num_interactions for (k, v) in popularity.items()}
        return list(dict(sorted(popularity.items())).values())
