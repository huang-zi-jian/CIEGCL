"""
author: hzj
date: 2024-6-26
file info:
"""
import torch
from model_operation import ModelOperator
from metrics_v2 import Metrics
from metrics_v2 import metrics
import numpy as np
from tqdm import tqdm


class Tester(object):
    def __init__(self, flags_obj, model_operator: ModelOperator):
        super(Tester, self).__init__()
        self.flags_obj = flags_obj
        self.model_operator = model_operator
        self.num_users = model_operator.dataset.num_users
        self.train_csr_record = model_operator.dataset.train_csr_record
        self.test_labels = model_operator.dataset.test_labels
        self.Metrics = Metrics()

    def test(self):
        results = [{metric: 0.0 for metric in self.Metrics.metrics} for _ in self.flags_obj.topks]
        # self.model_operator.load_model(epoch)
        with torch.no_grad():
            test_batch_size = self.flags_obj.test_batch_size
            test_users = np.arange(0, self.num_users, 1)
            batch_id = int(np.ceil(self.num_users / test_batch_size))

            # test_labels = self.model_operator.dataset.test_labels

            all_user_num = 0
            for batch in tqdm(range(batch_id), desc='Test'):
                start = batch * self.flags_obj.test_batch_size
                end = min((batch + 1) * self.flags_obj.test_batch_size, len(test_users))
                test_users_numpy = test_users[start:end]

                predictions = self.topk_recommend(test_users_numpy)
                for index, topk in enumerate(self.flags_obj.topks):
                    topk_predictions = predictions[:, :topk]
                    recall, precision, ndcg, hit_ratio, user_num = metrics(test_users_numpy, topk_predictions,
                                                                           self.test_labels)
                    results[index]['precision'] += precision
                    results[index]['recall'] += recall
                    results[index]['hit_ratio'] += hit_ratio
                    results[index]['ndcg'] += ndcg
                    all_user_num += user_num

            all_user_num = all_user_num / len(self.flags_obj.topks)
            for i in range(len(self.flags_obj.topks)):
                results[i]['precision'] = results[i]['precision'] / all_user_num
                results[i]['recall'] = results[i]['recall'] / all_user_num
                results[i]['hit_ratio'] = results[i]['hit_ratio'] / all_user_num
                results[i]['ndcg'] = results[i]['ndcg'] / all_user_num

            return results

    def topk_recommend(self, users):
        users_tensor = torch.LongTensor(users)
        predictions = self.model_operator.getUsersRating(users_tensor)

        mask = self.train_csr_record[users].toarray()
        mask = torch.Tensor(mask).to(self.flags_obj.device)
        predictions = predictions * (1 - mask) - 1e3 * mask
        predictions = predictions.argsort(descending=True)

        max_topk = max(self.flags_obj.topks)

        predictions = np.array(predictions.cpu())[:, :max_topk]

        return predictions
