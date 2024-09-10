"""
author: hzj
date: 2024-6-18
file info:
"""
import random
import torch.nn.functional as func
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import os


class GraphDataset(Dataset):
    def __init__(self, flags_obj):
        super(GraphDataset, self).__init__()
        self.flags_obj = flags_obj
        self.dataset_path = os.path.join(self.flags_obj.dataset, self.flags_obj.dataset_name)

        self.num_users = 0
        self.num_items = 0
        # self.trainInteractionSize = 0
        # self.testInteractionSize = 0
        self.popularity = None
        self.activity = None
        self.lil_train_record = None
        self.lil_test_record = None

        self.symmetric_Graph = None  # (n+m) x (n+m)，symmetric norm
        self.origin_Graph = None  # (n+m) x (n+m)，original

        self.symmetric_sub_graph = None  # n x m，symmetric norm
        self.origin_sub_graph = None  # n x m，original

        self.SVD_symmetric_sub_graph = None  # n x m，symmetric norm，svd
        self.SVD_origin_sub_graph = None  # n x m，original，svd

        self.U_mul_S = None
        self.V_mul_S = None

        self.train_csr_record = sp.load_npz(os.path.join(self.dataset_path, 'train_csr_record.npz'))
        self.train_csr_record = self.train_csr_record.astype(np.bool).astype(np.int)
        self.test_csr_record = sp.load_npz(os.path.join(self.dataset_path, 'test_csr_record.npz'))
        self.test_csr_record = self.test_csr_record.astype(np.bool).astype(np.int)

        self.test_coo_record = self.test_csr_record.tocoo(copy=True)
        self.test_labels = [[] for _ in range(self.test_coo_record.shape[0])]

        self.intermediate_items = {}
        self.structure_weight = {}
        self.distant_items = {}

        self.init()

    def init(self):
        self.num_users = self.train_csr_record.shape[0]
        self.num_items = self.train_csr_record.shape[1]

        # self.lil_train_record = (self.train_coo_record + self.train_skew_coo_record).tolil(copy=True)
        self.lil_train_record = self.train_csr_record.tolil(copy=True)
        self.lil_test_record = self.test_csr_record.tolil(copy=True)
        # self.activity = np.array(self.lil_train_record.sum(axis=1)).flatten()
        # self.popularity = np.array(self.lil_train_record.sum(axis=0)).flatten()
        self.activity = np.array(self.lil_train_record.sum(axis=1))
        self.popularity = np.array(self.lil_train_record.sum(axis=0))

        self.initSparseGraph()

        for i in range(len(self.test_coo_record.data)):
            row = self.test_coo_record.row[i]
            col = self.test_coo_record.col[i]
            self.test_labels[row].append(col)

        if self.flags_obj.model_name in ['AED']:
            self.khops_walk()

    def khops_walk(self):
        try:
            with open(os.path.join(self.dataset_path, 'intermediate_items.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split('::')
                    index = line[0].strip(' ').split(' ')
                    weight = line[1].strip(' ').split(' ')
                    if len(index) > 1:
                        user = int(index[0])
                        user_intermediate_items = [int(i) for i in index[1:]]
                        self.intermediate_items[user] = user_intermediate_items
                        user_intermediate_weight = [int(i) for i in weight]
                        self.structure_weight[user] = user_intermediate_weight
                    else:
                        self.intermediate_items[user] = []
                        self.structure_weight[user] = []
            print("Successfully loaded users intermediate items...")

        except:
            user_1_hops = self.train_csr_record
            item_1_hops = self.train_csr_record.T

            user_2_hops = user_1_hops.dot(item_1_hops)

            diags = sp.diags(np.ones(self.num_users), format='csr', dtype=np.int32)
            user_2_hops = user_2_hops - user_2_hops.multiply(diags)

            weight_user_1_3_hops = user_2_hops.dot(user_1_hops)
            weight_user_3_hops = weight_user_1_3_hops - weight_user_1_3_hops.multiply(user_1_hops)

            weight_user_3_hops = weight_user_3_hops.toarray().astype(np.int32)

            weight_act_pop = (1 - self.flags_obj.temp) * np.log(1 + self.activity * self.popularity).astype(np.float32)
            weight_3_hop = self.flags_obj.temp * np.log(1 + weight_user_3_hops).astype(np.float32)
            self.structure_weight = weight_3_hop + weight_act_pop
            # self.structure_weight = weight_user_3_hops

    def initSparseGraph(self):
        try:
            # pop_adj = sp.load_npz(os.path.join(self.dataset_path, 's_pre_pop_adj_matrix.npz'))
            # print('Successfully loaded popular adjacency matrix...')
            symmetric_matrix = sp.load_npz(os.path.join(self.dataset_path, 'symmetric_matrix_csr.npz'))
            print('Successfully loaded symmetric matrix...')
            original_matrix = sp.load_npz(os.path.join(self.dataset_path, 'original_matrix_csr.npz'))
            print('Successfully loaded original matrix...')
            symmetric_sub_matrix = sp.load_npz(os.path.join(self.dataset_path, 'symmetric_sub_matrix_csr.npz'))
            print('Successfully loaded symmetric sub-matrix...')
        except:
            empty_matrix = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items),
                                         dtype=np.float32)
            original_matrix = empty_matrix.tolil(copy=True)
            # recon_svd_matrix = empty_matrix.tolil(copy=True)

            original_matrix[:self.num_users, self.num_users:] = self.lil_train_record
            original_matrix[self.num_users:, :self.num_users] = self.lil_train_record.T
            original_matrix = original_matrix.todok()

            row_sum = np.array(original_matrix.sum(axis=1))

            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_matrix = sp.diags(d_inv)
            symmetric_matrix = d_matrix.dot(original_matrix).dot(d_matrix)
            # row_sum = np.array(norm_adj.sum(axis=1))
            symmetric_matrix = symmetric_matrix.tocsr()
            sp.save_npz(os.path.join(self.dataset_path, 'symmetric_matrix_csr.npz'), symmetric_matrix)
            print("Successfully generating symmetric matrix")

            original_matrix = original_matrix.tocsr()
            sp.save_npz(os.path.join(self.dataset_path, 'original_matrix_csr.npz'), original_matrix)
            print("Successfully generating original matrix")

            train_dok = self.train_csr_record.todok(copy=True)
            activity_sum = np.array(train_dok.sum(axis=-1))
            d_activity = np.power(activity_sum, -0.5).flatten()
            d_activity = sp.diags(d_activity)
            popular_sum = np.array(train_dok.sum(axis=0))
            d_popular = np.power(popular_sum, -0.5).flatten()
            d_popular = sp.diags(d_popular)
            symmetric_sub_matrix = d_activity.dot(train_dok).dot(d_popular)

            symmetric_sub_matrix = symmetric_sub_matrix.tocsr()
            sp.save_npz(os.path.join(self.dataset_path, 'symmetric_sub_matrix_csr.npz'), symmetric_sub_matrix)
            print("Successfully generating symmetric sub-matrix")

        if self.flags_obj.adj_split:
            self.symmetric_Graph = self._split_A_hat(symmetric_matrix)
        else:
            # self.popular_Graph = self._convert_sp_matrix_to_tensor(pop_adj)
            # self.popular_Graph = self.popular_Graph.coalesce().to(self.flags_obj.device)

            self.symmetric_Graph = self.convert_sp_matrix_to_tensor(symmetric_matrix)
            self.symmetric_Graph = self.symmetric_Graph.coalesce().to(self.flags_obj.device)
            self.symmetric_sub_graph = self.convert_sp_matrix_to_tensor(symmetric_sub_matrix)
            self.symmetric_sub_graph = self.symmetric_sub_graph.coalesce().to(self.flags_obj.device)

            self.origin_Graph = self.convert_sp_matrix_to_tensor(original_matrix)
            self.origin_Graph = self.origin_Graph.coalesce().to(self.flags_obj.device)
            self.origin_sub_graph = self.convert_sp_matrix_to_tensor(self.train_csr_record)
            self.origin_sub_graph = self.origin_sub_graph.coalesce().to(self.flags_obj.device)

            if self.flags_obj.model_name in ['lightGCL']:
                svd_u, s, svd_v = torch.svd_lowrank(self.symmetric_sub_graph, q=self.flags_obj.q)
                self.SVD_symmetric_sub_graph = svd_u @ torch.diag(s) @ svd_v.T
                self.U_mul_S = svd_u @ torch.diag(s)
                self.V_mul_S = svd_v @ torch.diag(s)

            elif self.flags_obj.model_name in ['AED']:
                svd_u, s, svd_v = torch.svd_lowrank(self.origin_sub_graph, q=self.flags_obj.q)
                self.SVD_origin_sub_graph = svd_u @ torch.diag(s) @ svd_v.T
                self.U_mul_S = svd_u @ torch.diag(s)
                self.V_mul_S = svd_v @ torch.diag(s)

    def _split_A_hat(self, adj):
        adj_fold = []
        num_folds = 5
        fold_len = (self.num_users + self.num_items) // num_folds
        for fold in range(num_folds):
            start = fold * fold_len
            if fold == num_folds - 1:
                end = self.num_users + self.num_items
            else:
                end = (fold + 1) * fold_len
            adj_fold.append(self.convert_sp_matrix_to_tensor(adj[start:end]).coalesce().to('cuda:0'))

        return adj_fold

    @staticmethod
    def convert_sp_matrix_to_tensor(x):
        coo = x.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class UniformTrainDataset(Dataset):
    def __init__(self, dataset: GraphDataset):
        super(UniformTrainDataset, self).__init__()
        self.lil_train_record = dataset.lil_train_record

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

    def __len__(self):
        return self.lil_train_record.nnz

    def __getitem__(self, index):
        while True:
            user = np.random.randint(0, self.num_users)
            user_positive_items = self.lil_train_record.rows[user]
            if user_positive_items:
                positive_item = random.choice(user_positive_items)
                break

        while True:
            negative_item = np.random.randint(0, self.num_items)
            if negative_item in user_positive_items:
                continue
            else:
                break

        return user, positive_item, negative_item


class GCNRSTrainDataset(Dataset):
    def __init__(self, dataset: GraphDataset):
        super(GCNRSTrainDataset, self).__init__()
        self.lil_train_record = dataset.lil_train_record
        self.dok_train_record = dataset.lil_train_record.todok(copy=True)
        self.train_record = list(self.dok_train_record.keys())

        self.num_items = dataset.num_items

    def __len__(self):
        return self.lil_train_record.nnz

    def __getitem__(self, index):
        user = self.train_record[index][0]
        positive_item = self.train_record[index][1]
        user_positive_items = self.lil_train_record.rows[user]

        while True:
            negative_item = np.random.randint(0, self.num_items)
            if negative_item in user_positive_items:
                continue
            else:
                break

        return user, positive_item, negative_item


def shuffle(*arrays):
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(array[shuffle_indices] for array in arrays)

    return result


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(tensor[i:i + batch_size] for tensor in tensors)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
