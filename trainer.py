"""
author: hzj
date: 2024-6-20
file info:
"""
import torch
from dataloader import GraphDataset, GCNRSTrainDataset, UniformTrainDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from metrics_v2 import metrics
from tester_v3 import Tester
from absl import logging
from model_operation import ModelOperator, CIEGCL_ModelOperator
from tqdm import tqdm
import numpy as np
import time


class WarmupLR(object):
    def __init__(self, optimizer, warmup_steps):
        super(WarmupLR, self).__init__()

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_count = 0

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    def get_lr(self):
        # step_num = self.step_count + 1
        # warmupLR
        return [base_lr * self.warmup_steps ** 0.5 * min(self.step_count ** -0.5,
                                                         self.step_count * self.warmup_steps ** -1.5) for base_lr in
                self.base_lrs]
        # NoamLR
        # return [
        #     base_lr * self.d_model ** -0.5 * min(self.step_count ** -0.5, self.step_count * self.warmup_steps ** -1.5)
        #     for base_lr in self.base_lrs]

    def step(self):
        self.step_count += 1
        values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr


class TrainManager(object):
    @staticmethod
    def get_trainer(flags_obj, workspace):
        dataset = GraphDataset(flags_obj)  # 加载数据
        if flags_obj.model_name == "CIEGCL":
            model_operator = CIEGCL_ModelOperator(flags_obj, workspace, dataset)
            return Trainer(flags_obj, model_operator)
        else:
            raise Exception


class Trainer(object):
    def __init__(self, flags_obj, model_operator: ModelOperator):
        super(Trainer, self).__init__()
        self.flags_obj = flags_obj
        self.model_operator = model_operator
        self.tester = Tester(flags_obj, model_operator)

        self.dataloader = None
        self.sampler = None
        self.optimizer = None
        self.init()

    def init(self):
        # train_dataset = GCNRSTrainDataset(self.model_operator.dataset)
        train_dataset = UniformTrainDataset(self.model_operator.dataset)
        self.dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.flags_obj.batch_size,
            shuffle=True,
            drop_last=False
        )
        # self.sampler = UniformSample(self.model_operator.dataset)
        # weight_decay=1e-8，但是设置weight_decay不如自己计算的regular_loss
        self.optimizer = optim.Adam(self.model_operator.model.parameters(), lr=self.flags_obj.lr,
                                    weight_decay=self.flags_obj.weight_decay)
        # self.optimizer = optim.Adam(self.model_operator.model.parameters(), lr=self.flags_obj.lr)

    def train(self):
        # 保存最初的随机嵌入
        self.model_operator.save_model(0)
        results = self.tester.test()
        logging.info("Test: {}".format(results))

        # warmup_scheduler = WarmupLR(optimizer=self.optimizer, warmup_steps=self.flags_obj.warmup_steps)
        for epoch in range(self.flags_obj.epochs):
            # current_time = time.ctime()
            # print(current_time)
            # warmup_scheduler.step()
            total_batch = 0
            total_loss = 0.
            for batch_count, batch_samples in enumerate(tqdm(self.dataloader, desc='train epoch {}'.format(epoch))):
                loss = self.model_operator.get_loss(tuple(batch_samples))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_batch = batch_count + 1
                total_loss += loss.item()

            average_loss = total_loss / total_batch
            logging.info(
                "Epoch[{}/{}] loss: {}".format(epoch, self.flags_obj.epochs, average_loss))  # 每个epoch输出一次平均loss

            if (epoch + 1) % 10 == 0:
                self.model_operator.save_model(epoch)
                results = self.tester.test()
                logging.info("Test: {}".format(results))
