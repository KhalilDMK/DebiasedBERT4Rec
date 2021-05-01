from .base import AbstractDataset

import pandas as pd

from datetime import date


class AMAZONBEAUTY(AbstractDataset):
    @classmethod
    def code(cls):
        return 'amazon-beauty'

    @classmethod
    def is_zipfile(cls):
        return False

    @classmethod
    def url(cls):
        return 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty.csv'

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


