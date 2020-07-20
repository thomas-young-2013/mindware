import time
import numpy as np

from solnml.components.hpo_optimizer.base_optimizer import BaseHPOptimizer
from solnml.components.utils.mfse_utils.mfsebase import MfseBase


class MfseOptimizer(BaseHPOptimizer, MfseBase):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1,
                 R=27, eta=3, n_jobs=1):
        BaseHPOptimizer.__init__(self, evaluator, config_space, seed)
        MfseBase.__init__(self, eval_func=self.evaluator, config_space=self.config_space,
                          seed=seed, R=R, eta=eta, n_jobs=n_jobs)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

    def iterate(self, num_iter=1):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(num_iter):
            self._iterate(self.s_values[self.inner_iter_id])
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

        iteration_cost = time.time() - _start_time
        inc_idx = np.argmin(np.array(self.incumbent_perfs))

        for idx in range(len(self.incumbent_perfs)):
            self.eval_dict[(None, self.incumbent_configs[idx])] = -self.incumbent_perfs[idx]

        self.perfs = self.incumbent_perfs
        self.configs = self.incumbent_configs
        self.incumbent_perf = -self.incumbent_perfs[inc_idx]
        self.incumbent_config = self.incumbent_configs[inc_idx]
        # incumbent_perf: the large the better
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_runtime_history(self):
        return self.incumbent_perfs, self.time_ticks, self.incumbent_perf

    # def gc(self):
    #     self.executor.shutdown()
