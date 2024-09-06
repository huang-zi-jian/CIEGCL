"""
author: hzj
date: 2024-6-20
file info:
"""
import numpy as np
from dataloader import GraphDataset
from sklearn.metrics import roc_auc_score


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: GraphDataset
    r_all = np.zeros((dataset.num_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


class Metrics(object):
    def __init__(self):
        self.topk_items = None
        self.test_positive_items = None
        self.metrics = ["precision", "recall", "hit_ratio", "ndcg"]
        # self.metrics = ["precision", "recall", "ndcg"]

    def init_set(self, topk_items, test_positive_items):
        self.topk_items = topk_items
        self.test_positive_items = test_positive_items

    def get_metrics(self, metric):
        metrics_dict = {
            "precision": self.precision,
            "recall": self.recall,
            "hit_ratio": self.hr,
            "ndcg": self.ndcg,
        }

        return metrics_dict[metric]

    def recall(self):
        hit_count = np.isin(self.topk_items, self.test_positive_items).sum()

        return hit_count / np.count_nonzero(self.test_positive_items >= 0)

    def precision(self):
        hit_count = np.isin(self.topk_items, self.test_positive_items).sum()

        return hit_count / len(self.topk_items)

    def hr(self):
        hit_count = np.isin(self.topk_items, self.test_positive_items).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    def ndcg(self):
        index = np.arange(len(self.topk_items))
        k = min(len(self.topk_items), np.count_nonzero(self.test_positive_items >= 0))

        idcg = (1 / np.log2(2 + np.arange(k))).sum()
        dcg = (1 / np.log2(2 + index[np.isin(self.topk_items, self.test_positive_items)])).sum()

        return dcg / idcg


def metrics(users, predictions, test_labels):
    user_num = 0
    all_recall = 0
    all_precision = 0
    all_ndcg = 0
    all_hit = 0
    for i in range(len(users)):
        user = users[i]
        prediction = list(predictions[i])
        label = test_labels[user]
        if len(label) > 0:
            user_num += 1

            hit_count = np.isin(prediction, label).sum()
            all_recall = all_recall + hit_count / len(label)
            all_precision = all_precision + hit_count / len(prediction)

            if hit_count > 0:
                all_hit = all_hit + 1

            index = np.arange(len(prediction))
            k = min(len(prediction), len(label))
            idcg = (1 / np.log2(2 + np.arange(k))).sum()
            dcg = (1 / np.log2(2 + index[np.isin(prediction, label)])).sum()

            all_ndcg = all_ndcg + dcg / idcg

    return all_recall, all_precision, all_ndcg, all_hit, user_num
