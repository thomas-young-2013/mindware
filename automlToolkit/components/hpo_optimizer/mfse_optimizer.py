import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold

from automlToolkit.components.hpo_optimizer.base_optimizer import BaseHPOptimizer
from automlToolkit.components.computation.parallel_func import ParallelExecutor
from automlToolkit.components.hpo_optimizer.utils.acquisition import EI
from automlToolkit.components.hpo_optimizer.utils.acq_optimizer import RandomSampling
from automlToolkit.components.hpo_optimizer.utils.prob_rf import RandomForestWithInstances
from automlToolkit.components.hpo_optimizer.utils.prob_rf_cluster import WeightedRandomForestCluster
from automlToolkit.components.hpo_optimizer.utils.funcs import get_types, minmax_normalization
from automlToolkit.components.hpo_optimizer.utils.config_space_utils import convert_configurations_to_array, \
    sample_configurations, expand_configurations


class MfseOptimizer(BaseHPOptimizer):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', trials_per_iter=1, seed=1,
                 R=81, eta=3, n_workers=1):
        super().__init__(evaluator, config_space, seed)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.trials_per_iter = trials_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.config_space = config_space
        self.n_workers = n_workers

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        self.incumbent_configs = []
        self.incumbent_perfs = []

        # Parameters in Hyperband framework.
        self.restart_needed = True
        self.R = R
        self.eta = eta
        self.seed = seed
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))
        self.inner_iter_id = 0

        # Parameters in MFSE-HB.
        self.weight_update_id = 0
        self.iterate_r = []
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

        types, bounds = get_types(config_space)
        self.num_config = len(bounds)
        init_weight = [0.]
        init_weight.extend([1. / self.s_max] * self.s_max)
        self.weighted_surrogate = WeightedRandomForestCluster(types, bounds, self.s_max,
                                                              self.eta, init_weight, 'gpoe')
        self.weight_changed_cnt = 0
        self.hist_weights = list()
        self.executor = ParallelExecutor(self.evaluator, n_worker=n_workers)
        # TODO: need to improve with lite-bo.
        self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
                                                     config_space,
                                                     n_samples=max(500, 50 * self.num_config),
                                                     rng=np.random.RandomState(seed))

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
        inc_idx = np.argmin(np.array(self.incumbent_perf))

        self.incumbent_perf = 1 - self.incumbent_perfs[inc_idx]
        self.incumbent_config = self.incumbent_configs[inc_idx]
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def _iterate(self, s, skip_last=0):
        if self.weight_update_id > self.s_max:
            self.update_weight()
        self.weight_update_id += 1

        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-s))

        # Choose a batch of configurations in different mechanisms.
        start_time = time.time()
        T = self.fetch_candidate_configurations(n)
        time_elapsed = time.time() - start_time
        self.logger.info("Choosing next configurations took %.2f sec." % time_elapsed)

        for i in range((s + 1) - int(skip_last)):  # changed from s + 1

            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations

            n_configs = n * self.eta ** (-i)
            n_resource = r * self.eta ** i

            self.logger.info("MFSE: %d configurations x size %f each" %
                             (int(n_configs), float(n_resource / self.R)))

            val_losses = self.executor.parallel_execute(T, subsample_ratio=float(n_resource / self.R))

            self.target_x[int(n_resource)].extend(T)
            self.target_y[int(n_resource)].extend(val_losses)

            if int(n_resource) == self.R:
                self.incumbent_configs.extend(T)
                self.incumbent_perfs.extend(val_losses)

            # Select a number of best configurations for the next loop.
            # Filter out early stops, if any.
            indices = np.argsort(val_losses)
            if len(T) >= self.eta:
                T = [T[i] for i in indices]
                reduced_num = int(n_configs / self.eta)
                T = T[0:reduced_num]
            else:
                T = [T[indices[0]]]
        for item in self.iterate_r[self.iterate_r.index(r):]:
            # NORMALIZE Objective value: MinMax linear normalization
            normalized_y = minmax_normalization(self.target_y[item])
            self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                          np.array(normalized_y, dtype=np.float64), r=item)

    def fetch_candidate_configurations(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        config_cnt = 0
        config_candidates = list()
        total_sample_cnt = 0

        while config_cnt < num_config and total_sample_cnt < 3 * num_config:
            incumbent = dict()
            max_r = self.iterate_r[-1]
            best_index = np.argmin(self.target_y[max_r])
            incumbent['config'] = self.target_x[max_r][best_index]
            approximate_obj = self.weighted_surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
            incumbent['obj'] = approximate_obj

            self.weighted_acquisition_func.update(model=self.weighted_surrogate, eta=incumbent)
            _config = self.weighted_acq_optimizer.maximize(batch_size=1)[0]

            if _config not in config_candidates:
                config_candidates.append(_config)
                config_cnt += 1
            total_sample_cnt += 1

        if config_cnt < num_config:
            config_candidates = expand_configurations(config_candidates, self.config_space, num_config)
        return config_candidates

    def update_weight(self):
        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.weighted_surrogate.surrogate_r
        K = len(r_list)
        p = 3

        if len(test_y) >= 3:
            # Get previous weights
            preserving_order_p = list()
            preserving_order_nums = list()
            for i, r in enumerate(r_list):
                fold_num = 5
                if i != K - 1:
                    mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                    tmp_y = np.reshape(mean, -1)
                    preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                    preserving_order_p.append(preorder_num / pair_num)
                    preserving_order_nums.append(preorder_num)
                else:
                    if len(test_y) < 2 * fold_num:
                        preserving_order_p.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.config_space)
                            _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            _surrogate.train(train_configs, train_y)
                            pred, _ = _surrogate.predict(valid_configs)
                            cv_pred[valid_idx] = pred.reshape(-1)
                        preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
            trans_order_weight = np.array(preserving_order_p)
            power_sum = np.sum(np.power(trans_order_weight, p))
            new_weights = np.power(trans_order_weight, p) / power_sum
        else:
            old_weights = list()
            for i, r in enumerate(r_list):
                _weight = self.weighted_surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info(' %d-th Updating weights: %s' % (self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(r_list):
            self.weighted_surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num
