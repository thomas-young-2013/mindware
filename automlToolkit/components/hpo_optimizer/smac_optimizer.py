import time
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer


class SMACOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space,
                 time_limit=None, evaluation_limit=None, trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter

        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": config_space,
            "deterministic": "true"
        }

        self.optimizer = SMAC(scenario=Scenario(scenario_dict),
                              rng=np.random.RandomState(self.seed),
                              tae_runner=self.evaluator)
        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf, self.incumbent_config = -1, None

    def run(self):
        while True:
            evaluation_num = len(self.perfs)
            if self.evaluation_num_limit is not None and evaluation_num > self.evaluation_num_limit:
                break
            if self.time_limit is not None and time.time() - self.start_time > self.time_limit:
                break
            self.iterate()
        return np.max(self.perfs)

    def iterate(self):
        _start_time = time.time()
        for _ in range(self.trials_per_iter):
            self.optimizer.iterate()

            runhistory = self.optimizer.solver.runhistory
            runkeys = list(runhistory.data.keys())
            for key in runkeys[self.trial_cnt:]:
                _reward = 1. - runhistory.data[key][0]
                _config = runhistory.ids_config[key[0]]
                self.perfs.append(_reward)
                self.configs.append(_config)
                if _reward > self.incumbent_perf:
                    self.incumbent_perf = _reward
                    self.incumbent_config = _config

            self.trial_cnt = len(runhistory.data.keys())
        iteration_cost = time.time() - _start_time

        return self.incumbent_perf, iteration_cost, self.incumbent_config
