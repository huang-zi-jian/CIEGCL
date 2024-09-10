"""
author: hzj
date: 2024-6-18
file info:
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(LightGCN, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_dim = self.flags_obj.embedding_dim
        self.Graph = dataset.symmetric_Graph

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    @staticmethod
    def __dropout_x(x, static_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()

        random_index = torch.rand(len(values)) + static_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / static_prob

        graph = torch.sparse.FloatTensor(index.t(), values, size)

        return graph

    def __dropout(self, static_prob):
        if self.flags_obj.adj_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, static_prob))
        else:
            graph = self.__dropout_x(self.Graph, static_prob)

        return graph

    def computer(self):
        # users_embed = self.user_embedding
        # items_embed = self.item_embedding
        all_embeds = torch.cat([self.user_embedding, self.item_embedding])
        embeds = [all_embeds]

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        for layer in range(self.flags_obj.n_layers):
            if self.flags_obj.adj_split:
                temp_embed = []
                for i in range(len(graph_droped)):
                    temp_embed.append(torch.sparse.mm(graph_droped[i], embeds))

                side_embed = torch.cat(temp_embed, dim=0)
                all_embeds = side_embed
            else:
                # 聚合邻居节点以及子环信息 todo: 没有自环信息？
                all_embeds = torch.sparse.mm(graph_droped, all_embeds)

            embeds.append(all_embeds)
        embeds = torch.stack(embeds, dim=1)
        light_out = torch.mean(embeds, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items], dim=0)

        return users, items

    def forward(self, users, positive_items, negative_items):
        all_users, all_items = self.computer()
        users_embed = all_users[users]
        positive_embed = all_items[positive_items]
        negative_embed = all_items[negative_items]

        positive_scores = torch.sum(torch.mul(users_embed, positive_embed), dim=-1)
        negative_scores = torch.sum(torch.mul(users_embed, negative_embed), dim=-1)

        bpr_loss = torch.mean(func.softplus(negative_scores - positive_scores))
        # loss = mean_loss + self.flags_obj.weight_decay * regular_loss

        return bpr_loss


class CIEGCL(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(CIEGCL, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_csr = dataset.train_csr_record
        self.embedding_dim = self.flags_obj.embedding_dim
        self.cl_weight = self.flags_obj.cl_weight
        self.csr_to_tensor = dataset.convert_sp_matrix_to_tensor
        # self.num_community = self.flags_obj.num_community
        self.Graph = dataset.symmetric_sub_graph

        # self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        # self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.user_embedding = Parameter(nn.init.xavier_uniform_(torch.empty(self.num_users, self.embedding_dim)))
        self.item_embedding = Parameter(nn.init.xavier_uniform_(torch.empty(self.num_items, self.embedding_dim)))

        # self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    @staticmethod
    def __dropout_x(x, static_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()

        random_index = torch.rand(len(values)) + static_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / static_prob

        graph = torch.sparse.FloatTensor(index.t(), values, size)

        return graph

    def __dropout(self, static_prob):
        # if self.flags_obj.adj_split:
        #     graph = []
        #     for g in self.origin_graph:
        #         graph.append(self.__dropout_x(g, static_prob))
        # else:
        graph = self.__dropout_x(self.Graph, static_prob)
        # svd_graph = self.__dropout_x(self.SVD_Graph, static_prob)

        return graph

    def computer(self):
        user_embedding_list = [self.user_embedding]
        item_embedding_list = [self.item_embedding]

        if self.flags_obj.dropout:
            if self.training:
                graph = self.__dropout(self.flags_obj.static_prob)
            else:
                graph = self.Graph
        else:
            graph = self.Graph

        for layer in range(self.flags_obj.n_layers):
            user_embedding_list.append(graph @ item_embedding_list[layer])
            item_embedding_list.append(graph.T @ user_embedding_list[layer])

        user_embedding = sum(user_embedding_list) / (self.flags_obj.n_layers + 1)
        item_embedding = sum(item_embedding_list) / (self.flags_obj.n_layers + 1)

        return user_embedding, item_embedding

    def inter_computer(self, users, adjacent_items):
        inter_user_embedding_list = [self.user_embedding]
        inter_item_embedding_list = [self.item_embedding]

        # if self.flags_obj.dropout:
        #     if self.training:
        #         graph = self.__dropout(self.flags_obj.static_prob)
        #     else:
        #         graph = self.Graph
        # else:
        #     graph = self.Graph

        sample_csr = sp.csr_matrix((np.ones(users.shape[0]), (users.cpu(), adjacent_items.cpu())),
                                   shape=(self.num_users, self.num_items),
                                   dtype=np.int)
        sample_csr = sample_csr.astype(np.bool).astype(np.int)

        sample_tensor = self.csr_to_tensor(sample_csr).coalesce().to(self.flags_obj.device)
        inter_tensor_graph = self.Graph - self.Graph * sample_tensor

        for layer in range(self.flags_obj.n_layers):
            inter_user_embedding_list.append(inter_tensor_graph @ inter_item_embedding_list[layer])
            inter_item_embedding_list.append(inter_tensor_graph.T @ inter_user_embedding_list[layer])

        inter_user_embedding = sum(inter_user_embedding_list) / (self.flags_obj.n_layers + 1)
        inter_item_embedding = sum(inter_item_embedding_list) / (self.flags_obj.n_layers + 1)

        return inter_user_embedding, inter_item_embedding

    def forward(self, users, positive_items, negative_items):
        user_embedding, item_embedding = self.computer()
        inter_user_embedding, inter_item_embedding = self.inter_computer(users, positive_items)

        # cl loss
        temp = self.flags_obj.temp
        # a = torch.log(torch.exp(inter_user_embedding[users] @ user_embedding.T / temp).sum(1) + 1e-8).mean()
        # c = torch.exp(inter_item_embedding[negative_items] @ item_embedding.T / temp).sum(1)
        # b = torch.log(torch.exp(inter_item_embedding[negative_items] @ item_embedding.T / temp).sum(1) + 1e-8).mean()
        neg_score = torch.log(
            torch.exp(inter_user_embedding[users] @ user_embedding.T / temp).sum(1) + 1e-8).mean() + torch.log(
            torch.exp(inter_item_embedding[negative_items] @ item_embedding.T / temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((inter_user_embedding[users] * user_embedding[users]).sum(1) / temp, -5.0,
                                 5.0)).mean() + (
                        torch.clamp((inter_item_embedding[negative_items] * item_embedding[negative_items]).sum(1) / temp,
                                    -5.0, 5.0)).mean()
        loss_cl = -pos_score + neg_score

        # bpr loss
        users_embed = user_embedding[users]
        positive_embed = item_embedding[positive_items]
        negative_embed = item_embedding[negative_items]

        positive_scores = torch.sum(torch.mul(users_embed, positive_embed), dim=-1)
        negative_scores = torch.sum(torch.mul(users_embed, negative_embed), dim=-1)

        loss_bpr = torch.mean(func.softplus(negative_scores - positive_scores))

        loss = loss_bpr + self.cl_weight * loss_cl

        return loss
