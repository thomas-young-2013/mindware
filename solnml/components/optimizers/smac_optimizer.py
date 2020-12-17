import time
import numpy as np
from litebo.facade.bo_facade import BayesianOptimization as BO
from litebo.utils.constants import SUCCESS
from solnml.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT


class SMACOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, output_dir='./',
                 inner_iter_num_per_iter=1, seed=1, n_jobs=1):
        super().__init__(evaluator, config_space, name, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.inner_iter_num_per_iter = inner_iter_num_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.output_dir = output_dir
        self.optimizer = BO(objective_function=self.evaluator,
                            config_space=config_space,
                            max_runs=int(1e10),
                            task_id=None,
                            time_limit_per_trial=self.per_run_time_limit,
                            rng=np.random.RandomState(self.seed))

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
            _threshold = int(len(set(self.config_space.sample_configuration(5000))))
            self.config_num_threshold = _threshold
        self.logger.debug('The maximum trial number in HPO is: %d' % self.config_num_threshold)
        self.maximum_config_num = min(600, self.config_num_threshold)
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

    def iterate(self, budget=MAX_INT):
        _start_time = time.time()

        if len(self.configs) == 0 and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
            print('initial hpo trial num is set to %d' % inner_iter_num)
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        for _ in range(inner_iter_num):
            if len(self.configs) >= self.maximum_config_num:
                self.early_stopped_flag = True
                self.logger.warning('Already explored 70 percentage of the '
                                    'hyperspace or maximum configuration number met: %d!' % self.maximum_config_num)
                break
            if time.time() - _start_time > budget:
                self.logger.warning('Time limit exceeded!')
                break
            _config, _status, _perf, _ = self.optimizer.iterate()
            if _status == SUCCESS:
                self.exp_output[time.time()] = (_config, _perf)
                self.configs.append(_config)
                self.perfs.append(-_perf)
                self.combine_tmp_config_path()

        runhistory = self.optimizer.get_history()
        if self.name == 'hpo':
            if hasattr(self.evaluator, 'fe_config'):
                fe_config = self.evaluator.fe_config
            else:
                fe_config = None
            self.eval_dict = {(fe_config, hpo_config): [-score, time.time()] for hpo_config, score in
                              runhistory.data.items()}
        else:
            if hasattr(self.evaluator, 'hpo_config'):
                hpo_config = self.evaluator.hpo_config
            else:
                hpo_config = None
            self.eval_dict = {(fe_config, hpo_config): [-score, time.time()] for fe_config, score in
                              runhistory.data.items()}

        if len(runhistory.get_incumbents()) > 0:
            self.incumbent_config, self.incumbent_perf = runhistory.get_incumbents()[0]
            self.incumbent_perf = -self.incumbent_perf
        iteration_cost = time.time() - _start_time
        # incumbent_perf: the large the better
        return self.incumbent_perf, iteration_cost, self.incumbent_config
