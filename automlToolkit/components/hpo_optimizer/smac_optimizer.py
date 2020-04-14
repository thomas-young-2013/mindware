import time
import datetime
import numpy as np
from litebo.facade.bo_facade import BayesianOptimization as BO
from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer


class SMACOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./',
                 trials_per_iter=1, seed=1, n_jobs=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.output_dir = output_dir

        self.optimizer = BO(objective_function=self.evaluator,
                            configspace=config_space,
                            max_runs=int(1e10),
                            task_id=None,
                            rng=np.random.RandomState(self.seed))

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
        self.maximum_config_num = min(600, self.config_num_threshold)
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

        runhistory = self.optimizer.get_history()
        self.incumbent_config, self.incumbent_perf = runhistory.get_incumbents()[-1]
        self.incumbent_perf = 1 - self.incumbent_perf
        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def optimize(self):
        self.optimizer.optimize()

        runhistory = self.optimizer.get_history()
        self.incumbent_config, self.incumbent_perf = runhistory.get_incumbents()[-1]
        self.incumbent_perf = 1 - self.incumbent_perf
        return self.incumbent_config, self.incumbent_perf
