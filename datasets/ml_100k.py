from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML100KDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):
        return 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README', 'u.data']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('u.data')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
