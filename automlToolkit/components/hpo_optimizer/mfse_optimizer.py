import time
import datetime
import numpy as np
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer


class MFESOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = -1.
        self.incumbent_config = self.config_space.get_default_configuration()

    def iterate(self):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config
