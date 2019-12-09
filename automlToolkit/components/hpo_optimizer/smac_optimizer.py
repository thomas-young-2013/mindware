import time
import datetime
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer


class SMACOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, output_dir='./', trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit

        if not output_dir.endswith('/'):
            output_dir += '/'
        output_dir += "smac3_output_%s" % (datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
        self.scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true",
            "cutoff_time": self.per_run_time_limit,
            'output_dir': output_dir
        }

        self.optimizer = SMAC(scenario=Scenario(self.scenario_dict),
                              rng=np.random.RandomState(self.seed),
                              tae_runner=self.evaluator)
        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = -1.
        self.incumbent_config = self.config_space.get_default_configuration()
        # Estimate the size of the hyperparameter space.
        self.config_num_threshold = int(len(set(
            self.config_space.sample_configuration(12500))) * 0.8)

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
        _flag = False
        for _ in range(self.trials_per_iter):
            if len(self.configs) >= self.config_num_threshold:
                _flag = True
                self.logger.warning('Already explored 70 percentage of the '
                                    'hp space: %d!' % self.config_num_threshold)
                break

            self.optimizer.iterate()

            runhistory = self.optimizer.solver.runhistory
            runkeys = list(runhistory.data.keys())
            for key in runkeys[self.trial_cnt:]:
                _reward = 1. - runhistory.data[key][0]
                _config = runhistory.ids_config[key[0]]
                self.perfs.append(_reward)
                self.configs.append(_config)
                if _reward is not None and _reward > self.incumbent_perf:
                    self.incumbent_perf = _reward
                    self.incumbent_config = _config

            self.trial_cnt = len(runhistory.data.keys())
        if not _flag:
            iteration_cost = time.time() - _start_time
        else:
            iteration_cost = None
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def optimize(self):
        self.scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true",
            "runcount-limit": self.evaluation_num_limit,
            "wallclock_limit": self.time_limit
        }
        self.optimizer = SMAC(scenario=Scenario(self.scenario_dict),
                              rng=np.random.RandomState(self.seed),
                              tae_runner=self.evaluator)

        self.optimizer.optimize()

        runhistory = self.optimizer.solver.runhistory
        runkeys = list(runhistory.data.keys())
        for key in runkeys:
            _reward = 1. - runhistory.data[key][0]
            _config = runhistory.ids_config[key[0]]
            self.perfs.append(_reward)
            self.configs.append(_config)
            if _reward > self.incumbent_perf:
                self.incumbent_perf = _reward
                self.incumbent_config = _config
        return self.incumbent_config, self.incumbent_perf
