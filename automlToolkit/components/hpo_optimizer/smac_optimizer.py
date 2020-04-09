import time
import datetime
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer


class SMACOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

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
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        # Estimate the size of the hyperparameter space.
        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(10000))) * 0.75)
            self.config_num_threshold = _threshold
        self.logger.debug('HP_THRESHOLD is: %d' % self.config_num_threshold)
        self.maximum_config_num = min(1000, self.config_num_threshold)
        self.early_stopped_flag = False

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
            if len(self.configs) >= self.maximum_config_num:
                self.early_stopped_flag = True
                self.logger.warning('Already explored 70 percentage of the '
                                    'hp space or maximum configuration number: %d!' % self.maximum_config_num)
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
        iteration_cost = time.time() - _start_time
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
