import abc
import time
import numpy as np
from automlToolkit.utils.constant import MAX_INT
from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.evaluators.evaluator import Evaluator


class BaseHPOptimizer(object):
    def __init__(self, evaluator: Evaluator, config_space, seed=None):
        self.evaluator = evaluator
        self.config_space = config_space
        self.seed = np.random.random_integers(MAX_INT) if seed is None else seed
        self.start_time = time.time()
        self.timing_list = list()
        self.incumbent = None
        self.logger = get_logger(__class__.__name__)

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def iterate(self):
        pass
