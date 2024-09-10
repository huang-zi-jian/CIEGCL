"""
author: hzj
date: 2024-7-3
file info:
"""
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import scipy.io as scio


def generate_record(dataset_dir):
    total_record = []
    with open(os.path.join(dataset_dir, 'train.txt')) as f:
        for line in f.readlines():
            line = line.strip('\n').strip(' ').split(' ')
            if len(line) > 1:
                user = int(line[0])
                items = [int(i) for i in line[1:]]
                for item in items:
                    total_record.append((user, item))

    with open(os.path.join(dataset_dir, 'test.txt')) as f:
        for line in f.readlines():
            line = line.strip('\n').strip(' ').split(' ')
            if len(line) > 1:
                user = int(line[0])
                items = [int(i) for i in line[1:]]
                for item in items:
                    total_record.append((user, item))

    record = pd.DataFrame(total_record, columns=['uid', 'iid'])
    record = record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
    record.to_csv(os.path.join(dataset_dir, 'record.csv'))


class Splitter(object):
    def __init__(self, record):
        super(Splitter, self).__init__()

        self.record = record
        self.num_users = record['uid'].nunique()
        self.num_items = record['iid'].nunique()
        self.num_records = len(record)

    @staticmethod
    def rank(record):
        if 'ts' in record.columns:
            record['rank'] = record['ts'].groupby(record['uid']).rank(method='first', pct=True, ascending=False)
        else:
            record['rank'] = np.random.uniform(low=0, high=1, size=len(record))

        return record


class RandomSplitter(Splitter):
    def __init__(self, record):
        super(RandomSplitter, self).__init__(record)

    def split(self, splits):
        record = self.record
        splits = np.array(splits)
        splits = (self.num_records * splits).astype(np.int32)

        results = []
        for n in splits[:-1]:
            sample_record = record.sample(n, random_state=np.random.RandomState()).reset_index(drop=True)
            results.append(sample_record)
            record = pd.concat([record, sample_record]).drop_duplicates(keep=False).reset_index(drop=True)

        results.append(record)

        return results


class SkewSplitter(Splitter):
    def __init__(self, record):
        super(SkewSplitter, self).__init__(record)
        self.valid_test_record = None
        self.valid_record = None
        self.test_record = None
        self.train_record = None

    def split_double(self, splits, sample_type, cap=None):
        record = self.record

        if sample_type == 'popularity':
            popularity = record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
            record = record.merge(popularity, on='iid')
        elif sample_type == 'activity':
            activity = record[['iid', 'uid']].groupby('uid').count().reset_index().rename(columns={'iid': 'count'})
            record = record.merge(activity, on='uid')
        else:
            raise

        record['count'] = record['count'].apply(lambda x: 1 / x)
        if cap is not None:
            count = record['count'].to_numpy()
            count = np.unique(count)

            cap_threshold = np.percentile(count, cap)
            record['count'] = record['count'].apply(lambda x: min(x, cap_threshold))

        # self.valid_test_record = record.groupby('iid').apply(pd.DataFrame.sample, frac=splits[1],
        #                                                      weights='count').reset_index(drop=True)
        self.valid_test_record = record.sample(frac=splits[1], weights='count',
                                               random_state=np.random.RandomState()).reset_index(drop=True)
        self.train_record = pd.concat([record, self.valid_test_record]).drop_duplicates(keep=False).reset_index(
            drop=True)
        self.drop_and_reset_index_double()

        return self.train_record, self.valid_test_record

    def split_triple(self, splits, sample_type, cap=None):
        record = self.record

        popularity = record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
        record = record.merge(popularity, on='iid')
        record['count'] = record['count'].apply(lambda x: 1 / x)

        if cap is not None:
            pop = record['count'].to_numpy()
            pop = np.unique(pop)

            cap_threshold = np.percentile(pop, cap)
            record['count'] = record['count'].apply(lambda x: min(x, cap_threshold))

        self.test_record = record.groupby('uid').apply(pd.DataFrame.sample, frac=splits[2],
                                                       weights='count').reset_index(
            drop=True)
        train_valid_record = pd.concat([record, self.test_record]).drop_duplicates(keep=False).reset_index(drop=True)
        train_valid_record = self.rank(train_valid_record)

        self.train_record = train_valid_record[train_valid_record['rank'] >= splits[1] / (splits[0] + splits[1])]
        self.valid_record = train_valid_record[train_valid_record['rank'] < splits[1] / (splits[0] + splits[1])]
        self.drop_and_reset_index_triple()

        return self.train_record, self.valid_record, self.test_record

    def split(self, splits, sample_type='popularity', cap=None):
        if len(splits) == 3:
            return self.split_triple(splits, sample_type, cap)
        elif len(splits) == 2:
            return self.split_double(splits, sample_type, cap)
        else:
            return None

    def unbiased_split(self, splits, cap=None):
        popularity = self.record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
        record = self.record.merge(popularity, on='iid')
        record['count'] = record['count'].apply(lambda x: 1 / x)

        if cap is not None:
            pop = record['count'].to_numpy()
            pop = np.unique(pop)

            cap_threshold = np.percentile(pop, cap)
            record['count'] = record['count'].apply(lambda x: min(x, cap_threshold))

        self.valid_test_record = record.sample(frac=splits[1], weights='count',
                                               random_state=np.random.RandomState()).reset_index(drop=True)
        self.train_record = pd.concat([record, self.valid_test_record]).drop_duplicates(keep=False).reset_index(
            drop=True)
        self.drop_and_reset_index_double()

        return self.train_record, self.valid_test_record

    def drop_and_reset_index_double(self):
        self.train_record = self.train_record.drop(columns=['count']).reset_index(drop=True)
        self.valid_test_record = self.valid_test_record.drop(columns=['count']).reset_index(drop=True)

    def drop_and_reset_index_triple(self):
        self.train_record = self.train_record.drop(columns=['rank', 'count']).reset_index(drop=True)
        self.valid_record = self.valid_record.drop(columns=['rank', 'count']).reset_index(drop=True)
        self.test_record = self.test_record.drop(columns=['count']).reset_index(drop=True)


class TemporalSplitter(Splitter):
    def __init__(self, record):
        super(TemporalSplitter, self).__init__(record)
        self.early_record = None
        self.middle_record = None
        self.late_record = None

    def split_double(self, splits):
        """

        :param splits: splits=[a, b]表示early_record占比为a;late_record占比为b
        :return:
        """
        record = self.rank(self.record)
        self.early_record = record[record['rank'] >= splits[1]]
        self.late_record = record[record['rank'] < splits[1]]

        self.drop_and_reset_index_double()

        return self.early_record, self.late_record

    def split_triple(self, splits):
        """

        :param splits: splits=[a, b, c]
        :return:
        """
        record = self.rank(self.record)
        record_copy = record.copy()
        record_copy['rank'] = record_copy.groupby('uid')['rank'].transform(np.random.permutation)  # todo: 混杂rank是为什么？

        self.early_record = record_copy[record_copy['rank'] >= splits[1] + splits[2]]
        self.middle_record = record_copy[
            (record_copy['rank'] < splits[1] + splits[2]) & (record_copy['rank'] > splits[2])]
        self.late_record = record_copy[record_copy['rank'] <= splits[2]]

        self.drop_and_reset_index_triple()

        return self.early_record, self.middle_record, self.late_record

    def split(self, splits):
        if len(splits) == 3:
            return self.split_triple(splits)
        elif len(splits) == 2:
            return self.split_double(splits)
        else:
            return None

    def drop_and_reset_index_double(self):
        self.early_record = self.early_record.drop(columns=['rank']).reset_index(drop=True)
        self.late_record = self.late_record.drop(columns=['rank']).reset_index(drop=True)

    def drop_and_reset_index_triple(self):
        self.early_record = self.early_record.drop(columns=['rank']).reset_index(drop=True)
        self.middle_record = self.middle_record.drop(columns=['rank']).reset_index(drop=True)
        self.late_record = self.late_record.drop(columns=['rank']).reset_index(drop=True)


class CsvGenerator(object):
    def __init__(self, dataset_dir):
        super(CsvGenerator, self).__init__()
        self.dataset_dir = dataset_dir
        self.record = None
        self.num_users = 0
        self.num_items = 0

        self.init()

    def init(self):
        try:
            record = pd.read_csv(os.path.join(self.dataset_dir, 'record.csv'), index_col=0)
        except:
            total_record = []
            with open(os.path.join(dataset_dir, 'train.txt')) as f:
                for line in f.readlines():
                    line = line.strip('\n').strip(' ').split(' ')
                    if len(line) > 1:
                        user = int(line[0])
                        items = [int(i) for i in line[1:]]
                        for item in items:
                            total_record.append((user, item))

            with open(os.path.join(dataset_dir, 'test.txt')) as f:
                for line in f.readlines():
                    line = line.strip('\n').strip(' ').split(' ')
                    if len(line) > 1:
                        user = int(line[0])
                        items = [int(i) for i in line[1:]]
                        for item in items:
                            total_record.append((user, item))

            record = pd.DataFrame(total_record, columns=['uid', 'iid'])
            record = record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
            # record.to_csv(os.path.join(dataset_dir, 'record.csv'))
            # self.record = record

        # finally:
        preprocessor = preprocess(record, scope=[3, 1e8])
        self.record = preprocessor.get_reset_record()

        self.num_users = max(self.record['uid']) + 1
        self.num_items = max(self.record['iid']) + 1

    def unbiased_split(self, skew_splits, temporal_splits, sample_type):
        """

        :param skew_splits: train:rest
        :param temporal_splits: skew_train:skew_test
        :return:
        """
        skew_splitter = SkewSplitter(self.record)

        train_record, rest_record = skew_splitter.split(splits=skew_splits, sample_type=sample_type)
        # train_num_users = train_record.groupby('uid').count().size

        # if train_num_users == self.num_users:
        train_record = train_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
        rest_record = rest_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
        train_record.to_csv(os.path.join(dataset_dir, 'train_record.csv'))
        rest_record.to_csv(os.path.join(dataset_dir, 'rest_record.csv'))

        if temporal_splits:
            temporal_splitter = TemporalSplitter(rest_record)
            train_skew_record, test_skew_record = temporal_splitter.split(splits=temporal_splits)

            train_skew_record = train_skew_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
            test_skew_record = test_skew_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
            train_skew_record.to_csv(os.path.join(dataset_dir, 'train_skew_record.csv'))
            test_skew_record.to_csv(os.path.join(dataset_dir, 'test_skew_record.csv'))

            train_skew_csr_record = sp.csr_matrix(
                (np.ones(len(train_skew_record)), (train_skew_record['uid'], train_skew_record['iid'])),
                shape=(self.num_users, self.num_items), dtype=np.int32)
            test_skew_csr_record = sp.csr_matrix(
                (np.ones(len(test_skew_record)), (test_skew_record['uid'], test_skew_record['iid'])),
                shape=(self.num_users, self.num_items), dtype=np.int32)
            sp.save_npz(os.path.join(self.dataset_dir, 'train_skew_csr_record.npz'), train_skew_csr_record)
            sp.save_npz(os.path.join(self.dataset_dir, 'test_skew_csr_record.npz'), test_skew_csr_record)

        else:
            test_skew_csr_record = sp.csr_matrix((np.ones(len(rest_record)), (rest_record['uid'], rest_record['iid'])),
                                                 shape=(self.num_users, self.num_items), dtype=np.int32)
            sp.save_npz(os.path.join(self.dataset_dir, 'test_skew_csr_record.npz'), test_skew_csr_record)

        train_csr_record = sp.csr_matrix((np.ones(len(train_record)), (train_record['uid'], train_record['iid'])),
                                         shape=(self.num_users, self.num_items), dtype=np.int32)
        sp.save_npz(os.path.join(self.dataset_dir, 'train_csr_record.npz'), train_csr_record)

    def split(self, temporal_splits):
        """

        :param temporal_splits: train:test
        :return:
        """
        temporal_splitter = TemporalSplitter(self.record)
        train_record, test_record = temporal_splitter.split(splits=temporal_splits)

        train_record = train_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
        test_record = test_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
        train_record.to_csv(os.path.join(dataset_dir, 'train_record.csv'))
        test_record.to_csv(os.path.join(dataset_dir, 'test_record.csv'))

        train_csr_record = sp.csr_matrix((np.ones(len(train_record)), (train_record['uid'], train_record['iid'])),
                                         shape=(self.num_users, self.num_items), dtype=np.int32)
        test_csr_record = sp.csr_matrix((np.ones(len(test_record)), (test_record['uid'], test_record['iid'])),
                                        shape=(self.num_users, self.num_items), dtype=np.int32)

        sp.save_npz(os.path.join(self.dataset_dir, 'train_csr_record.npz'), train_csr_record)
        sp.save_npz(os.path.join(self.dataset_dir, 'test_csr_record.npz'), test_csr_record)


def original_dataset(dataset_dir):
    train_record = []
    test_record = []
    with open(os.path.join(dataset_dir, 'train.txt')) as f:
        for line in f.readlines():
            line = line.strip().split()

            if len(line) > 1:
                user = int(line[0])
                items = [int(i) for i in line[1:]]
                for item in items:
                    train_record.append((user, item))

    with open(os.path.join(dataset_dir, 'test.txt')) as f:
        for line in f.readlines():
            line = line.strip().split()

            if len(line) > 1:
                user = int(line[0])
                items = [int(i) for i in line[1:]]
                for item in items:
                    test_record.append((user, item))

    train_record = pd.DataFrame(train_record, columns=['uid', 'iid'])
    train_record = train_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)
    num_users = max(train_record['uid']) + 1
    num_items = max(train_record['iid']) + 1

    test_record = pd.DataFrame(test_record, columns=['uid', 'iid'])
    test_record = test_record.sort_values(by=['uid', 'iid']).reset_index(drop=True)

    train_csr_record = sp.csr_matrix((np.ones(len(train_record)), (train_record['uid'], train_record['iid'])),
                                     shape=(num_users, num_items), dtype=np.int32)
    test_csr_record = sp.csr_matrix((np.ones(len(test_record)), (test_record['uid'], test_record['iid'])),
                                    shape=(num_users, num_items), dtype=np.int32)

    sp.save_npz(os.path.join(dataset_dir, 'train_csr_record.npz'), train_csr_record)
    sp.save_npz(os.path.join(dataset_dir, 'test_csr_record.npz'), test_csr_record)


def coat_ascii_load(dataset_dir):
    train_matrix = []
    test_matrix = []
    with open(os.path.join(dataset_dir, 'train.ascii')) as f:
        for line in f.readlines():
            if len(line) > 0:
                train_matrix.append(line.split())

    with open(os.path.join(dataset_dir, 'test.ascii')) as f:
        for line in f.readlines():
            if len(line) > 0:
                test_matrix.append(line.split())

    train_matrix = np.array(train_matrix).astype(int)
    test_matrix = np.array(test_matrix).astype(int)

    return train_matrix, test_matrix


def yahoo_load(dataset_dir):
    train_matrix = []
    test_matrix = []
    with open(os.path.join(dataset_dir, 'train.txt')) as f:
        for line in f.readlines():
            if len(line) > 0:
                train_matrix.append(line.strip().split())

    with open(os.path.join(dataset_dir, 'test.txt')) as f:
        for line in f.readlines():
            if len(line) > 0:
                test_matrix.append(line.strip().split())

    train_matrix = np.array(train_matrix).astype(int)
    test_matrix = np.array(test_matrix).astype(int)

    # train_coo_record = sp.csr_matrix((np.ones(len(train_record)), (train_record['uid'], train_record['iid'])),
    #                                  shape=(self.num_users, self.num_items))
    # test_coo_record = sp.csr_matrix((np.ones(len(test_record)), (test_record['uid'], test_record['iid'])),
    #                                 shape=(self.num_users, self.num_items))
    #
    # sp.save_npz(os.path.join(self.dataset_dir, 'train_coo_record.npz'), train_coo_record)
    # sp.save_npz(os.path.join(self.dataset_dir, 'test_coo_record.npz'), test_coo_record)

    return train_matrix, test_matrix


class COAT():

    def __init__(self):
        super(COAT, self).__init__()
        self.dataset_dir = '../dataset/coat_3'
        self.train_record = None
        self.test_record = None

        self.init()

    def init(self):
        train_list = []
        test_list = []
        with open(os.path.join(self.dataset_dir, 'train.ascii')) as f:
            for line in f.readlines():
                if len(line) > 0:
                    train_list.append(line.split())

        with open(os.path.join(self.dataset_dir, 'test.ascii')) as f:
            for line in f.readlines():
                if len(line) > 0:
                    test_list.append(line.split())

        train_matrix = np.array(train_list).astype(int)
        test_matrix = np.array(test_list).astype(int)

        row, col = np.nonzero(train_matrix)
        y = train_matrix[row, col]
        x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        df = pd.DataFrame(x, columns=['uid', 'iid', 'rating'])
        train_df = df[df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]
        # train_df.to_csv(os.path.join(self.dataset_dir, 'train_record.csv'))

        row, col = np.nonzero(test_matrix)
        y = test_matrix[row, col]
        x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        df = pd.DataFrame(x, columns=['uid', 'iid', 'rating'])
        test_df = df[df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]
        # test_df.to_csv(os.path.join(self.dataset_dir, 'test_record.csv'))

        uid_map = train_df.groupby('uid').count().reset_index().rename(columns={'iid': 'reset_uid'})
        iid_map = train_df.groupby('iid').count().reset_index().rename(columns={'uid': 'reset_iid'})
        uid_map['reset_uid'] = uid_map.index
        iid_map['reset_iid'] = iid_map.index

        train_df = train_df.merge(uid_map, on='uid')
        train_df = train_df.merge(iid_map, on='iid')
        train_df = train_df[['reset_uid', 'reset_iid']].rename(
            columns={'reset_uid': 'uid', 'reset_iid': 'iid'}).sort_values(by=['uid', 'iid']).reset_index(drop=True)

        test_df = test_df.merge(uid_map, on='uid')
        test_df = test_df.merge(iid_map, on='iid')
        test_df = test_df[['reset_uid', 'reset_iid']].rename(
            columns={'reset_uid': 'uid', 'reset_iid': 'iid'}).sort_values(by=['uid', 'iid']).reset_index(drop=True)

        self.train_record = train_df
        self.test_record = test_df
        train_df.to_csv(os.path.join(self.dataset_dir, 'train_record.csv'))
        test_df.to_csv(os.path.join(self.dataset_dir, 'test_record.csv'))

    def to_sparse(self):
        num_users = max(self.train_record['uid']) + 1
        num_items = max(self.train_record['iid']) + 1
        train_csr_record = sp.csr_matrix(
            (np.ones(len(self.train_record)), (self.train_record['uid'], self.train_record['iid'])),
            shape=(num_users, num_items), dtype=np.int32)
        test_csr_record = sp.csr_matrix(
            (np.ones(len(self.test_record)), (self.test_record['uid'], self.test_record['iid'])),
            shape=(num_users, num_items), dtype=np.int32)
        sp.save_npz(os.path.join(self.dataset_dir, 'train_csr_record.npz'), train_csr_record)
        sp.save_npz(os.path.join(self.dataset_dir, 'test_csr_record.npz'), test_csr_record)


class YAHOO():

    def __init__(self):
        super(YAHOO, self).__init__()
        self.dataset_dir = '../dataset/yahoo_3'
        self.train_record = None
        self.test_record = None

        self.init()

    def init(self):
        train_df = pd.read_table(os.path.join(self.dataset_dir, 'ydata-train.txt'), names=['uid', 'iid', 'rating'])
        train_df['uid'] = train_df['uid'] - min(train_df['uid'])
        train_df['iid'] = train_df['iid'] - min(train_df['iid'])
        test_df = pd.read_table(os.path.join(self.dataset_dir, 'ydata-test.txt'), names=['uid', 'iid', 'rating'])
        test_df['uid'] = test_df['uid'] - min(test_df['uid'])
        test_df['iid'] = test_df['iid'] - min(test_df['iid'])

        train_df = train_df[train_df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]
        test_df = test_df[test_df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]

        uid_map = train_df.groupby('uid').count().reset_index().rename(columns={'iid': 'reset_uid'})
        iid_map = train_df.groupby('iid').count().reset_index().rename(columns={'uid': 'reset_iid'})
        uid_map['reset_uid'] = uid_map.index
        iid_map['reset_iid'] = iid_map.index

        train_df = train_df.merge(uid_map, on='uid')
        train_df = train_df.merge(iid_map, on='iid')
        train_df = train_df[['reset_uid', 'reset_iid']].rename(
            columns={'reset_uid': 'uid', 'reset_iid': 'iid'}).sort_values(by=['uid', 'iid']).reset_index(drop=True)

        test_df = test_df.merge(uid_map, on='uid')
        test_df = test_df.merge(iid_map, on='iid')
        test_df = test_df[['reset_uid', 'reset_iid']].rename(
            columns={'reset_uid': 'uid', 'reset_iid': 'iid'}).sort_values(by=['uid', 'iid']).reset_index(drop=True)

        self.train_record = train_df
        self.test_record = test_df
        train_df.to_csv(os.path.join(self.dataset_dir, 'train_record.csv'))
        test_df.to_csv(os.path.join(self.dataset_dir, 'test_record.csv'))

    def to_sparse(self):
        num_users = max(self.train_record['uid']) + 1
        num_items = max(self.train_record['iid']) + 1
        train_csr_record = sp.csr_matrix(
            (np.ones(len(self.train_record)), (self.train_record['uid'], self.train_record['iid'])),
            shape=(num_users, num_items), dtype=np.int32)
        test_csr_record = sp.csr_matrix(
            (np.ones(len(self.test_record)), (self.test_record['uid'], self.test_record['iid'])),
            shape=(num_users, num_items), dtype=np.int32)
        sp.save_npz(os.path.join(self.dataset_dir, 'train_csr_record.npz'), train_csr_record)
        sp.save_npz(os.path.join(self.dataset_dir, 'test_csr_record.npz'), test_csr_record)


class preprocess(object):
    def __init__(self, original_record, scope):
        super(preprocess, self).__init__()
        self.record = original_record
        self.scope = scope

        # self.init()
        # self.reset_id()
        # self.record.to_csv(os.path.join(dataset_dir, 'record.csv'))

    # def init(self):
    #     raise NotImplementedError

    def get_reset_record(self):
        # self.reset_iid(filtrate=True)

        self.reset_iid(filtrate=True)
        # self.reset_uid(filtrate=True)
        self.reset_uid(filtrate=False)

        return self.record

    def reset_iid(self, filtrate):
        """

        :param filtrate: 
        :return:
        """
        if filtrate:
            popularity = self.record.groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
            record = self.record.merge(popularity, on='iid')
            self.record = record[(record['count'] >= self.scope[0]) & (record['count'] <= self.scope[1])].reset_index(
                drop=True)
        index = self.record.groupby('iid').count().reset_index().rename(columns={'uid': 'reset_iid'})
        index['reset_iid'] = index.index
        record = self.record.merge(index, on='iid')

        self.record = record[['uid', 'reset_iid']].sort_values(by=['uid', 'reset_iid']).reset_index(drop=True).rename(
            columns={'reset_iid': 'iid'})

    def reset_uid(self, filtrate):
        if filtrate:
            activity = self.record.groupby('uid').count().reset_index().rename(columns={'iid': 'count'})
            record = self.record.merge(activity, on='uid')
            self.record = record[(record['count'] >= self.scope[0]) & (record['count'] <= self.scope[1])].reset_index(
                drop=True)
        index = self.record.groupby('uid').count().reset_index().rename(columns={'iid': 'reset_uid'})
        index['reset_uid'] = index.index
        record = self.record.merge(index, on='uid')

        self.record = record[['reset_uid', 'iid']].sort_values(by=['reset_uid', 'iid']).reset_index(drop=True).rename(
            columns={'reset_uid': 'uid'})


def ml_original_record(name):
    df = pd.read_csv('../dataset/' + name + '/ratings.dat', sep='::', names=['uid', 'iid', 'rating', 'timestamp'])
    df['uid'] = df['uid'] - min(df['uid'])
    df['iid'] = df['iid'] - min(df['iid'])
    record = df[df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]
    record.to_csv('../dataset/' + name + '/record.csv')


def lastfm_original_record():
    df = pd.read_csv('../dataset/lastfm/user_artists.dat', sep='\t')
    df = df.rename(columns={'userID': 'uid', 'artistID': 'iid'})
    df['uid'] = df['uid'] - min(df['uid'])
    df['iid'] = df['iid'] - min(df['iid'])
    # user_count = df['uid'].nunique()
    # item_count = df['iid'].nunique()
    record = df.reset_index(drop=True)[['uid', 'iid']]
    record.to_csv('../dataset/lastfm/record.csv')


def ciao_original_record():
    with open('../dataset/Ciao/rating.txt', 'r', encoding='utf8') as f:
        record = []
        for line in f.readlines():
            line = line.strip().split('::::')
            try:
                record.append([int(line[0]), line[1], int(line[3]), line[5]])
                # if line[1] == '':
                #     print(line)
                #     print('kong')
            except:
                continue
    df = pd.DataFrame(record, columns=['uid', 'iid', 'rating', 'time'])
    record = df[df['rating'] >= 40].reset_index(drop=True)[['uid', 'iid']]
    record['uid'] = record['uid'] - min(record['uid'])
    record['iid'] = record['iid'] - min(record['iid'])
    record.to_csv('../dataset/Ciao/record.csv')


def epinions_original_record():
    record = scio.loadmat('../dataset/epinions/rating_with_timestamp.mat')
    record_df = pd.DataFrame(record['rating'],
                             columns=['uid', 'iid', 'typeid', 'rating', 'helpfulness', 'timestamps'])
    record_df = record_df[['uid', 'iid', 'rating']]
    record = record_df[record_df['rating'] >= 4].reset_index(drop=True)[['uid', 'iid']]
    record['uid'] = record['uid'] - min(record['uid'])
    record['iid'] = record['iid'] - min(record['iid'])
    record.to_csv('../dataset/epinions/record.csv')


def gowalla_original_record():
    record_df = pd.read_table('../dataset/gowalla/Gowalla_totalCheckins.txt',
                              names=['uid', 'time', 'latitude', 'longitude', 'iid'])
    record = record_df[['uid', 'iid']]
    record_df['uid'] = record_df['uid'] - min(record_df['uid'])
    record_df['iid'] = record_df['iid'] - min(record_df['iid'])
    record.to_csv('../dataset/gowalla/record.csv')


def amazon_original_record():
    record_df = pd.read_csv('../dataset/amazon/Reviews.csv')
    record_df = record_df.rename(columns={'UserId': 'uid', 'ProductId': 'iid'})
    record = record_df[record_df['Score'] >= 4].reset_index(drop=True)[['uid', 'iid']]
    record.to_csv('../dataset/amazon/record.csv')


def kuairec_original_record():
    kuairec = pd.read_csv('../dataset/KuaiRec 2.0/data/small_matrix.csv')
    df = kuairec[['user_id', 'video_id', 'watch_ratio']]
    df = df.rename(columns={'user_id': 'uid', 'video_id': 'iid'})
    df['uid'] = df['uid'] - min(df['uid'])
    df['iid'] = df['iid'] - min(df['iid'])
    record = df[df['watch_ratio'] >= 2].reset_index(drop=True)[['uid', 'iid']]
    record.to_csv('../dataset/KuaiRec 2.0/record.csv')


def npz_to_txt(dataset_dir):
    train_csr_record = sp.load_npz(os.path.join(dataset_dir, 'train_csr_record.npz'))
    test_csr_record = sp.load_npz(os.path.join(dataset_dir, 'test_csr_record.npz'))
    train_lil_record = train_csr_record.tolil(copy=True)
    test_lil_record = test_csr_record.tolil(copy=True)

    num_user = train_csr_record.shape[0]

    with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f:
        for user in range(num_user):
            items = train_lil_record.rows[user]
            if items:
                f.write(str(user) + ' ')
                for item in train_lil_record.rows[user]:
                    f.write(str(item) + ' ')
                f.write('\n')

    with open(os.path.join(dataset_dir, 'test.txt'), 'w') as f:
        for user in range(num_user):
            items = test_lil_record.rows[user]
            if items:
                f.write(str(user) + ' ')
                for item in test_lil_record.rows[user]:
                    f.write(str(item) + ' ')
                f.write('\n')


if __name__ == '__main__':
    dataset_dir = '../dataset/Ciao'
    # npz_to_txt(dataset_dir)
    # generate_record(dataset_dir)
    # csv_generator = CsvGenerator(dataset_dir)
    # csv_generator.unbiased_split(skew_splits=[0.8, 0.2], temporal_splits=None, sample_type='activity')
    # csv_generator.split(temporal_splits=[0.8, 0.2])

    # original_dataset(dataset_dir)
    # coat = COAT()
    # coat.to_sparse()
    # yahoo = YAHOO()
    # yahoo.to_sparse()
    # yahoo_load(dataset_dir)
    # ml_original_record(name='ml10m')
    # lastfm_original_record()
    ciao_original_record()
    # epinions_original_record()
    # gowalla_original_record()
    # amazon_original_record()
    # kuairec_original_record()
