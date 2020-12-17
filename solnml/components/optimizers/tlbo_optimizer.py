import time
import numpy as np
import os
import pickle as pkl
import re
from collections import OrderedDict

from litebo.utils.constants import SUCCESS
from litebo.optimizer.smbo import SMBO
from solnml.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT

cur_dir = os.path.dirname(__file__)
source_dir = os.path.join('%s', '..', 'transfer_learning', 'tlbo', 'runhistory') % cur_dir


class TlboOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, surrogate_type='tlbo_rgpe_prf',
                 metric='bal_acc', time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, output_dir='./',
                 inner_iter_num_per_iter=1, seed=1, n_jobs=1):
        super().__init__(evaluator, config_space, name, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.inner_iter_num_per_iter = inner_iter_num_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.output_dir = output_dir

        # TODO: leave target out
        if hasattr(evaluator, 'estimator_id'):
            estimator_id = evaluator.estimator_id
        else:
            raise ValueError
        runhistory_dir = os.path.join(source_dir, 'hpo2', '%s_%s_%s') % ('hpo', metric, estimator_id)
        dataset_names = get_datasets(runhistory_dir, estimator_id, metric)
        source_data = load_runhistory(runhistory_dir, dataset_names, estimator_id, metric)

        self.optimizer = SMBO(self.evaluator, config_space,
                              history_bo_data=source_data,
                              surrogate_type=surrogate_type,
                              max_runs=int(1e10),
                              time_limit_per_trial=self.per_run_time_limit,
                              logging_dir=output_dir)

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

    def iterate(self, budget=MAX_INT):
        _start_time = time.time()
        for _ in range(self.inner_iter_num_per_iter):
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

        self.incumbent_config, self.incumbent_perf = runhistory.get_incumbents()[0]
        self.incumbent_perf = -self.incumbent_perf
        iteration_cost = time.time() - _start_time
        # incumbent_perf: the large the better
        return self.incumbent_perf, iteration_cost, self.incumbent_config


def get_metafeature_vector(metafeature_dict):
    sorted_keys = sorted(metafeature_dict.keys())
    return np.array([metafeature_dict[key] for key in sorted_keys])


def get_datasets(runhistory_dir, estimator_id, metric, task_id='hpo'):
    _datasets = list()
    pattern = r'(.*)-%s-%s-%s.pkl' % (estimator_id, metric, task_id)
    for filename in os.listdir(runhistory_dir):
        result = re.search(pattern, filename, re.M | re.I)
        if result is not None:
            _datasets.append(result.group(1))
    return _datasets


def load_runhistory(runhistory_dir, dataset_names, estimator_id, metric, task_id='hpo'):
    metafeature_file = os.path.join(source_dir, 'metafeature.pkl')
    with open(metafeature_file, 'rb') as f:
        metafeature_dict = pkl.load(f)

    for dataset in metafeature_dict.keys():
        vec = get_metafeature_vector(metafeature_dict[dataset])
        metafeature_dict[dataset] = vec

    runhistory = list()
    for dataset in dataset_names:
        _filename = '%s-%s-%s-%s.pkl' % (dataset, estimator_id, metric, task_id)
        with open(os.path.join(runhistory_dir, _filename), 'rb') as f:
            data = pkl.load(f)
        runhistory.append(OrderedDict(data))
    return runhistory
