import time
import numpy as np
from litebo.utils.constants import SUCCESS, FAILDED
from solnml.components.hpo_optimizer.base_optimizer import BaseHPOptimizer
from solnml.components.transfer_learning.tlbo.models.kde import TPE


class TPEOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, output_dir='./',
                 trials_per_iter=1, seed=1, n_jobs=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.output_dir = output_dir

        self.config_gen = TPE(configspace=config_space)

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.exp_output = dict()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        # Estimate the size of the hyperparameter space.
        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(10000))) * 0.75)
            self.config_num_threshold = _threshold
        self.logger.debug('The maximum trial number in HPO is: %d' % self.config_num_threshold)
        self.maximum_config_num = min(600, self.config_num_threshold)
        self.early_stopped_flag = False
        self.eval_dict = {}

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
                                    'hyperspace or maximum configuration number met: %d!' % self.maximum_config_num)
                break
            _config = self.config_gen.get_config()[0]
            try:
                _perf = self.evaluator(_config)
                _status = SUCCESS
            except:
                _perf = np.inf
                _status = FAILDED
            self.config_gen.new_result(_config, _perf, 1)
            if _status == SUCCESS:
                self.exp_output[time.time()] = (_config, _perf)
                self.configs.append(_config)
                self.perfs.append(-_perf)

        if hasattr(self.evaluator, 'fe_config'):
            fe_config = self.evaluator.fe_config
        else:
            fe_config = None
        self.eval_dict = {(fe_config, self.configs[i]): -self.perfs[i] for i in range(len(self.perfs))}
        incumbent_idx = np.argsort(self.perfs)[-1]
        self.incumbent_config = self.configs[incumbent_idx]
        self.incumbent_perf = self.perfs[incumbent_idx]
        iteration_cost = time.time() - _start_time
        # incumbent_perf: the large the better
        return self.incumbent_perf, iteration_cost, self.incumbent_config
