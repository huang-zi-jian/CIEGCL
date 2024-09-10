"""
author: hzj
date: 2024-7-10
file info:
"""
from absl import logging
import os
import datetime


class logging_config(object):
    def __init__(self, flags_obj):
        self.flags_obj = flags_obj

        self.workspace = self.__get_workspace()
        self.log_path = self.get_log_path()

        self.set_flags_logging()
        self.log_flags()

    def __get_workspace(self):
        data_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        experiment_name = self.flags_obj.dataset_name + '_' + self.flags_obj.model_name + '_' + data_time
        workspace = os.path.join(self.flags_obj.output, experiment_name)
        if not os.path.exists(workspace):
            os.mkdir(workspace)

        return workspace

    def get_log_path(self):
        log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        return log_path

    def set_flags_logging(self):
        """

        :return:
        """
        logging.flush()
        logging.get_absl_handler().use_absl_log_file('flags.log', self.log_path)

    def set_train_logging(self):
        """

        :return:
        """
        logging.flush()
        logging.get_absl_handler().use_absl_log_file('train.log', self.log_path)

    def log_flags(self):
        logging.info('FLAGS')
        for flag, value in self.flags_obj.flag_values_dict().items():
            logging.info("{}: {}".format(flag, value))
