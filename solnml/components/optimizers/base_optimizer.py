import abc
import os
import time
import numpy as np
import pickle as pkl
from solnml.utils.constant import MAX_INT
from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator


class BaseOptimizer(object):
    def __init__(self, evaluator: _BaseEvaluator, config_space, name, seed=None):
        self.evaluator = evaluator
        self.config_space = config_space

        assert name in ['hpo', 'fe']
        self.name = name
        self.seed = np.random.random_integers(MAX_INT) if seed is None else seed
        self.start_time = time.time()
        self.timing_list = list()
        self.incumbent = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.init_hpo_iter_num = None
        self.early_stopped_flag = False

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def iterate(self, budget=MAX_INT):
        pass

    def combine_tmp_config_path(self):
        sorted_list_path = self.evaluator.topk_model_saver.sorted_list_path
        path_list = os.path.split(sorted_list_path)
        tmp_path = 'tmp_' + path_list[-1]
        tmp_filepath = os.path.join(os.path.dirname(sorted_list_path), tmp_path)

        # TODO: How to merge when using multi-process
        if os.path.exists(tmp_filepath):
            self.logger.info('Temporary config path detected!')
            with open(tmp_filepath, 'rb') as f1:
                sorted_file_replica = pkl.load(f1)
            with open(sorted_list_path, 'wb') as f2:
                pkl.dump(sorted_file_replica, f2)
            self.logger.info('Temporary config path merged!')

    def get_evaluation_stats(self):
        return

    def gc(self):
        return
