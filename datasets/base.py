from .utils import *
from config import RAW_DATASET_ROOT_FOLDER
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle
from sklearn.model_selection import train_test_split
import itertools
import random


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.export_root = self.args.export_root
        self.data_root = self._get_rawdata_folder_path()

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        if self.args.mode in ['generate_semi_synthetic', 'train_tf', 'tune_tf']:
            dataset = self.preprocess_for_semi_synthetic_generation()
        elif self.args.mode in ['train_bert_semi_synthetic', 'tune_bert_semi_synthetic']:
            dataset = self.preprocess_for_semi_synthetic_training()
        else:
            dataset = self.preprocess_for_real_training()
        return dataset

    def preprocess_for_real_training(self):
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        df = self.append_recommendations(df)
        train, val, test = self.split_implicit(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        return dataset

    def preprocess_for_semi_synthetic_generation(self):
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        #df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df, padding_item = False)
        df = df.sort_values(['uid', 'timestamp'])
        df = self.create_timesteps(df)
        if self.args.tf_target == 'exposure':
            df = self.convert_to_exposure(df, umap, smap)
            if self.args.frac_exposure_negatives:
                df = self.sample_negative_exposure(df, self.args.frac_exposure_negatives)
        train, val, test = self.split_explicit(df, test_rate=0.1)
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        return dataset

    def preprocess_for_semi_synthetic_training(self):
        self.check_semi_synthetic_data()
        df = self.load_semi_synthetic_scores()
        df, theta, gamma, exposure, relevance = self.generate_semi_synthetic_data(df, self.args.skewness_parameter)
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_implicit(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'theta': theta,
                   'gamma': gamma,
                   'exposure': exposure}
        return dataset

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw data doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def check_semi_synthetic_data(self):
        folder_path = self._get_rawdata_folder_path()
        if all(folder_path.joinpath('generated', score + '.npy') for score in ['interaction', 'rating']):
            print('Semi-synthetic data exists.')
            return
        else:
            raise ValueError('Semi-synthetic data does not exist. Generate it first.')

    def load_semi_synthetic_scores(self):
        folder_path = self._get_rawdata_folder_path()
        interaction = pd.DataFrame(np.load(folder_path.joinpath("generated", "interaction.npy")), columns=['uid', 'sid', 'timestamp', 'interaction'])
        rating = np.load(folder_path.joinpath("generated", "rating.npy"))
        interaction['rating'] = rating[:, -1]
        return interaction

    def generate_semi_synthetic_data(self, df, skewness):
        print('Generating semi_synthetic dataset...')
        df['gamma'] = df['rating'].apply(lambda x: 1 / (1 + np.exp(- x)))
        df['theta'] = df['interaction'].apply(lambda x: x ** skewness)
        df['r'] = np.random.binomial(n=1, p=df['gamma'])
        df['o'] = np.random.binomial(n=1, p=df['theta'])
        num_sessions = len(set(df['uid']))
        num_items = len(set(df['sid']))
        num_timesteps = len(set(df['timestamp']))
        theta = np.array(df['theta']).reshape(num_sessions, num_items, num_timesteps)
        gamma = np.array(df['gamma']).reshape(num_sessions, num_items, num_timesteps)
        exposure = np.array(df['o']).reshape(num_sessions, num_items, num_timesteps)
        relevance = np.array(df['r']).reshape(num_sessions, num_items, num_timesteps)
        df['y'] = df['o'] * df['r']
        df = df[df['y'] == 1]
        max_gamma = df[['uid', 'timestamp', 'gamma']].groupby(['uid', 'timestamp']).max().reset_index()
        max_gamma.columns = ['uid', 'timestamp', 'max_gamma']
        df = pd.merge(df, max_gamma, how='left', on=['uid', 'timestamp'])
        df['most_relevant'] = df['gamma'] == df['max_gamma']
        df = df[df['most_relevant'] == True]
        return df[['uid', 'sid', 'timestamp', 'rating']], theta, gamma, exposure, relevance

    def make_implicit(self, df):
        print('Converting ratings to interactions...')
        df = df[df['rating'] >= self.min_rating]
        return df

    def create_timesteps(self, df):
        timesteps = list(df[['uid', 'sid']].groupby('uid').count().reset_index(drop=True)['sid'])
        timesteps = sum([list(range(self.args.bert_max_len - count, self.args.bert_max_len)) for count in timesteps],
                        [])
        df['timestep'] = timesteps
        df = df[df['timestep'] >= 0].reset_index(drop=True)
        return df

    def convert_to_exposure(self, df, umap, smap):
        print('Converting ratings to exposure random variables...')
        df['rating'] = 1
        cartesian = pd.DataFrame(itertools.product(range(len(umap)), range(len(smap)), range(self.args.bert_max_len)), columns=['uid', 'sid', 'timestep'])
        cartesian['rating'] = 0
        pos_data = df[['uid', 'sid', 'timestep', 'rating']]
        df = pd.merge(cartesian, pos_data, on=['uid', 'sid', 'timestep'], how='left').fillna(0)
        df['rating'] = df['rating_x'] + df['rating_y']
        df = df[['uid', 'sid', 'timestep', 'rating']]
        return df

    def sample_negative_exposure(self, df, num_negatives):
        neg = df[df['rating'] == 0].reset_index(drop=True)
        pos = df[df['rating'] == 1].reset_index(drop=True)
        indices = random.sample(range(len(neg)), int(len(pos) * num_negatives))
        neg = neg.iloc[indices]
        df = pd.concat([pos, neg]).reset_index(drop=True)
        return df

    def filter_triplets(self, df):
        print('Filtering users and items with few interactions...')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df, padding_item=True):
        print('Densifying index...')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        if padding_item:
            smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        else:
            smap = {s: i for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_implicit(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting data...')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting data...')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[:-2*eval_set_size]
            val_user_index = permuted_index[-2*eval_set_size:-eval_set_size]
            test_user_index = permuted_index[-eval_set_size:]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df = df.loc[df['uid'].isin(val_user_index)]
            test_df = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def split_explicit(self, df, test_rate):
        print('Splitting data...')
        train, test = train_test_split(df, test_size=test_rate)
        train, val = train_test_split(train, test_size=test_rate / (1 - test_rate))
        return [list(train['uid']), list(train['sid']), list(train['timestep']), list(train['rating'])], [list(val['uid']), list(val['sid']), list(val['timestep']), list(val['rating'])], [list(test['uid']), list(test['sid']), list(test['timestep']), list(test['rating'])]

    def append_recommendations(self, df):
        if Path(self.export_root).joinpath('recommendations', 'rec_iter_' + str(self.args.iteration - 1) + '.pkl').is_file():
            uid = sorted(list(set(df['uid'])))
            rating = [5] * len(uid)
            for i in range(self.args.iteration):
                next_timestamp = max(df['timestamp']) + 1
                timestamp = [next_timestamp] * len(uid)
                sid = pickle.load(Path(self.export_root).joinpath('recommendations', 'rec_iter_' + str(i) + '.pkl').open('rb'))
                df = pd.concat([df, pd.DataFrame({'uid': uid, 'sid': sid, 'rating': rating, 'timestamp': timestamp})]).reset_index(drop=True)
        return df

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())
