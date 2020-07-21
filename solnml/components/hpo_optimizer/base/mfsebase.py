import time
import numpy as np
import random as rd
from math import log, ceil

from solnml.utils.constant import MAX_INT
from solnml.utils.logging_utils import get_logger
from solnml.components.hpo_optimizer.base.prob_rf import RandomForestWithInstances
from solnml.components.hpo_optimizer.base.config_space_utils import sample_configurations
from solnml.components.hpo_optimizer.base.acquisition import EI
from solnml.components.hpo_optimizer.base.acq_optimizer import RandomSampling
from solnml.components.hpo_optimizer.base.prob_rf_cluster import WeightedRandomForestCluster
from solnml.components.hpo_optimizer.base.funcs import get_types, std_normalization
from solnml.components.hpo_optimizer.base.config_space_utils import convert_configurations_to_array
from solnml.components.computation.parallel_process import ParallelProcessEvaluator


class MfseBase(object):
    def __init__(self, eval_func, config_space,
                 seed=1, R=81, eta=3, n_jobs=1):
        self.eval_func = eval_func
        self.config_space = config_space
        self.n_workers = n_jobs

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        self.incumbent_configs = list()
        self.incumbent_perfs = list()
        self.evaluation_stats = dict()
        self.evaluation_stats['timestamps'] = list()
        self.evaluation_stats['val_scores'] = list()
        self.global_start_time = time.time()
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

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
        self.exp_output = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        types, bounds = get_types(self.config_space)
        self.num_config = len(bounds)
        init_weight = [1. / self.s_max] * self.s_max + [0.]
        self.weighted_surrogate = WeightedRandomForestCluster(types, bounds, self.s_max,
                                                              self.eta, init_weight, 'gpoe')
        self.weight_changed_cnt = 0
        self.hist_weights = list()

        self.weighted_acquisition_func = EI(model=self.weighted_surrogate)
        self.weighted_acq_optimizer = RandomSampling(self.weighted_acquisition_func,
                                                     self.config_space,
                                                     n_samples=2000,
                                                     rng=np.random.RandomState(seed))
        self.eval_dict = dict()

    def _iterate(self, s, budget=MAX_INT, skip_last=0):
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
        self.logger.info("Choosing next batch of configurations took %.2f sec." % time_elapsed)

        with ParallelProcessEvaluator(self.eval_func, n_worker=self.n_workers) as executor:
            for i in range((s + 1) - int(skip_last)):  # changed from s + 1
                if time.time() >= budget + start_time:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_resource = r * self.eta ** i

                self.logger.info("MFSE: %d configurations x size %d / %d each" %
                                 (int(n_configs), n_resource, self.R))

                if self.n_workers > 1:
                    val_losses = executor.parallel_execute(T, resource_ratio=float(n_resource / self.R),
                                                           eta=self.eta,
                                                           first_iter=(i == 0))
                    for _id, _val_loss in enumerate(val_losses):
                        if np.isfinite(_val_loss):
                            self.target_x[int(n_resource)].append(T[_id])
                            self.target_y[int(n_resource)].append(_val_loss)
                            self.evaluation_stats['timestamps'].append(time.time() - self.global_start_time)
                            self.evaluation_stats['val_scores'].append(_val_loss)
                else:
                    val_losses = list()
                    for config in T:
                        val_loss = self.eval_func(config, resource_ratio=float(n_resource / self.R),
                                                  eta=self.eta, first_iter=(i == 0))
                        val_losses.append(val_loss)
                        if np.isfinite(val_loss):
                            self.target_x[int(n_resource)].append(config)
                            self.target_y[int(n_resource)].append(val_loss)
                            self.evaluation_stats['timestamps'].append(time.time() - self.global_start_time)
                            self.evaluation_stats['val_scores'].append(val_loss)

                self.exp_output[time.time()] = (int(n_resource), T, val_losses)

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
            if len(self.target_y[item]) == 0:
                continue
            normalized_y = std_normalization(self.target_y[item])
            self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                          np.array(normalized_y, dtype=np.float64), r=item)

    def fetch_candidate_configurations(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            return sample_configurations(self.config_space, num_config)

        incumbent = dict()
        max_r = self.iterate_r[-1]
        # The lower, the better.
        best_index = np.argmin(self.target_y[max_r])
        incumbent['config'] = self.target_x[max_r][best_index]
        approximate_obj = self.weighted_surrogate.predict(convert_configurations_to_array([incumbent['config']]))[0]
        incumbent['obj'] = approximate_obj
        self.weighted_acquisition_func.update(model=self.weighted_surrogate, eta=incumbent)

        config_candidates = self.weighted_acq_optimizer.maximize(batch_size=num_config)
        p_threshold = 0.3
        n_acq = self.eta * self.eta

        if num_config <= n_acq:
            return config_candidates

        candidates = config_candidates[: n_acq]
        idx_acq = n_acq
        for _id in range(num_config - n_acq):
            if rd.random() < p_threshold:
                config = sample_configurations(self.config_space, 1)[0]
            else:
                config = config_candidates[idx_acq]
                idx_acq += 1
            candidates.append(config)
        return candidates

    def update_weight(self):
        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        if len(incumbent_configs) <= 3:
            return
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.weighted_surrogate.surrogate_r
        K = len(r_list)
        if len(test_y) >= 3:
            # # p-norm
            # # Get previous weights
            # preserving_order_p = list()
            # preserving_order_nums = list()
            # for i, r in enumerate(r_list):
            #     fold_num = 5
            #     if i != K - 1:
            #         mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
            #         tmp_y = np.reshape(mean, -1)
            #         preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
            #         preserving_order_p.append(preorder_num / pair_num)
            #         preserving_order_nums.append(preorder_num)
            #     else:
            #         if len(test_y) < 2 * fold_num:
            #             preserving_order_p.append(0)
            #         else:
            #             # 5-fold cross validation.
            #             kfold = KFold(n_splits=fold_num)
            #             cv_pred = np.array([0] * len(test_y))
            #             for train_idx, valid_idx in kfold.split(test_x):
            #                 train_configs, train_y = test_x[train_idx], test_y[train_idx]
            #                 valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
            #                 types, bounds = get_types(self.config_space)
            #                 _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
            #                 _surrogate.train(train_configs, train_y)
            #                 pred, _ = _surrogate.predict(valid_configs)
            #                 cv_pred[valid_idx] = pred.reshape(-1)
            #             preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
            #             preserving_order_p.append(preorder_num / pair_num)
            #             preserving_order_nums.append(preorder_num)
            # p = 3
            # trans_order_weight = np.array(preserving_order_p)
            # power_sum = np.sum(np.power(trans_order_weight, p))
            # new_weights = np.power(trans_order_weight, p) / power_sum

            # sample
            n_sampling = 100
            argmin_cnt = [0] * K
            predictive_mu, predictive_std = list(), list()
            n_fold = 5
            n_instance = len(test_y)
            ranking_loss_hist = list()
            for i, r in enumerate(r_list):
                if i != K - 1:
                    _mean, _var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                    predictive_mu.append(_mean)
                    predictive_std.append(np.sqrt(_var))
                else:
                    fold_num = n_instance // n_fold
                    target_mu, target_std = list(), list()
                    for i in range(n_fold):
                        instance_indexs = list(range(n_instance))
                        bound = (n_instance - i * fold_num) if i == (n_fold - 1) else fold_num
                        start_id = i * fold_num
                        del instance_indexs[start_id: start_id + bound]
                        types, bounds = get_types(self.config_space)
                        _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                        _surrogate.train(test_x[instance_indexs, :], test_y[instance_indexs])
                        _mu, _var = _surrogate.predict(test_x[start_id: start_id + bound])
                        target_mu.extend(_mu.flatten())
                        target_std.extend(np.sqrt(_var).flatten())
                    predictive_mu.append(target_mu)
                    predictive_std.append(target_std)

            for _ in range(n_sampling):
                ranking_loss_list = list()
                for i, r in enumerate(r_list):
                    sampled_y = np.random.normal(predictive_mu[i], predictive_std[i])
                    rank_loss = 0
                    for i in range(len(test_y)):
                        for j in range(len(test_y)):
                            if (test_y[i] < test_y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                rank_loss += 1
                    ranking_loss_list.append(rank_loss)

                ranking_loss_hist.append(ranking_loss_list)
                argmin_id = np.argmin(ranking_loss_list)
                argmin_cnt[argmin_id] += 1

            new_weights = np.array(argmin_cnt) / n_sampling

        else:
            old_weights = list()
            for i, r in enumerate(r_list):
                _weight = self.weighted_surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info('Model weights[%d]: %s' % (self.weight_changed_cnt, str(new_weights)))
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
