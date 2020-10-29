import abc
import time
import numpy as np
from solnml.utils.constant import MAX_INT
from solnml.utils.logging_utils import get_logger
from solnml.components.evaluators.base_evaluator import _BaseEvaluator


class BaseHPOptimizer(object):
    def __init__(self, evaluator: _BaseEvaluator, config_space, seed=None):
        self.evaluator = evaluator
        self.config_space = config_space
        self.seed = np.random.random_integers(MAX_INT) if seed is None else seed
        self.start_time = time.time()
        self.timing_list = list()
        self.incumbent = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.init_hpo_iter_num = None

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def iterate(self, budget=MAX_INT):
        pass

    def get_evaluation_stats(self):
        return

    def gc(self):
        return
