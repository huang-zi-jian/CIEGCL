"""
author: hzj
date: 2024-6-20
file info:
"""
import torch
from CIEGCL import CIEGCL
import os
import torch.nn.functional as func
from dataloader import GraphDataset
import scipy.sparse as sp
import numpy as np


class ModelOperator(object):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(ModelOperator, self).__init__()
        self.flags_obj = flags_obj
        self.workspace = workspace
        self.dataset = dataset

        self.model = None
        self.device = torch.device(flags_obj.device)
        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.workspace, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')

        torch.save(self.model.state_dict(), model_path)

    def load_model(self, epoch):
        ckpt_path = os.path.join(self.workspace, 'ckpt')
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def getUsersRating(self, users):
        users = users.to(self.device)

        # all_users, all_items = self.model.computer()
        users_embed = self.model.user_embedding[users.long()]
        items_embed = self.model.item_embedding
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))
        return rating

    def get_loss(self, sample):
        users, positive_items, negative_items = sample

        users = users.to(self.device)
        positive_items = positive_items.to(self.device)
        negative_items = negative_items.to(self.device)

        loss = self.model(users, positive_items, negative_items)

        return loss


class CIEGCL_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(CIEGCL_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = CIEGCL(self.dataset)
        self.model.to(self.device)

    def getUsersRating(self, users):
        users = users.to(self.device)

        all_users, all_items = self.model.computer()
        users_embed = all_users[users.long()]
        items_embed = all_items
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))

        return rating
