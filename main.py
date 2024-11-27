"""
author: hzj
date: 2024-7-10
file info:
"""
from absl import app, flags, logging
from trainer import TrainManager
from config import logging_config
import scipy.sparse as sp
from dataloader import GraphDataset
import torch.nn.functional as func
import numpy as np
import torch

# set_seed(2020)

flags_obj = flags.FLAGS

flags.DEFINE_integer("batch_size", 2048, "Train batch size.")
flags.DEFINE_integer("test_batch_size", 256, "Test batch size.")
flags.DEFINE_integer("epochs", 1000, "The number of epoch for training.")
flags.DEFINE_integer("warmup_steps", 150, "The warmup steps.")
flags.DEFINE_integer("embedding_dim", 64, "Embedding dimension for embedding based models.")
flags.DEFINE_integer("faiss_gpu_id", 0, "GPU ID for faiss search.")
flags.DEFINE_integer("n_layers", 2, "The layer number.")
# flags.DEFINE_integer("topk", 20, "Top k for testing recommendation performance.")
flags.DEFINE_multi_integer("topks", [20], "Top k for testing recommendation performance.")
flags.DEFINE_float("lr", 0.001, "Learning rate.")
flags.DEFINE_float("temp", 0.1, "temperature weight.")
flags.DEFINE_float("static_prob", 0.95, "Rate for dropout.")
flags.DEFINE_float("cl_weight", 0.001, "Weight of cl loss.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay of optimizer.")
flags.DEFINE_string("output", 'C:/Users/admin/Desktop/CIEGCL/output', "Folder for experiment result.")
# flags.DEFINE_string("exp_name", "experiment", "Experiment name.")
flags.DEFINE_string("dataset", "C:/Users/admin/Desktop/CIEGCL/dataset/", "Folder for dataset.")
flags.DEFINE_enum("device", "cuda:0", ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], 'Device setting for training.')
flags.DEFINE_enum("dataset_name", "coat",
                  ["Ciao", "coat", "lastfm", "ml1m", "ml10m", "yahoo"], "Name of dataset.")
flags.DEFINE_enum("model_name", "CIEGCL", ['CIEGCL'],
                  'Model for training.')
flags.DEFINE_enum("discrepancy_loss", "dCor", ['L1', 'L2', 'dCor'], 'Discrepancy loss function.')
flags.DEFINE_enum("watch_metric", "recall", ['precision', 'recall', 'hit_ratio', 'ndcg'],
                  "Metric for scheduler's step.")
flags.DEFINE_enum("data_source", "valid", ['test', 'valid'], 'Which dataset to test.')
flags.DEFINE_multi_string('metrics', ['precision', 'recall', 'hit_ratio', 'ndcg'], 'Metrics list.')
flags.DEFINE_bool("adj_split", False, "Whether split matrix or not.")
flags.DEFINE_bool("dropout", False, "Whether drop graph or not.")
flags.DEFINE_bool("faiss_use_gpu", False, "Use GPU or not for faiss search.")


# logging.set_verbosity(logging.DEBUG)
# logging.use_absl_handler()
# logging.get_absl_handler().setFormatter(None)


def main(argv):

    config = logging_config(flags_obj)
    config.set_train_logging()
    trainer = TrainManager.get_trainer(flags_obj, config.workspace)
    trainer.train()


if __name__ == '__main__':
    app.run(main)
