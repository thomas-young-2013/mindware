import os
import time
import numpy as np
import pickle as pkl
from scipy.stats import norm
from typing import List
from sklearn.metrics import accuracy_score
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.components.feature_engineering.transformation_graph import DataNode, TransformationGraph
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.components.evaluators.base_evaluator import fetch_predict_estimator
from automlToolkit.utils.logging_utils import setup_logger, get_logger
from automlToolkit.components.utils.constants import CLS_TASKS
from automlToolkit.components.ensemble import EnsembleBuilder


class FirstLayerBandit(object):
    def __init__(self, task_type, trial_num,
                 classifier_ids: List[str], data: DataNode,
                 metric='acc',
                 ensemble_method='ensemble_selection',
                 ensemble_size=10,
                 per_run_time_limit=300, output_dir=None,
                 dataset_name='default_dataset',
                 tmp_directory='logs',
                 eval_type='holdout',
                 share_feature=False,
                 logging_config=None,
                 opt_algo='rb',
                 fe_algo='tree_based',
                 n_jobs=1,
                 seed=1):
        """
        :param classifier_ids: subset of {'adaboost','bernoulli_nb','decision_tree','extra_trees','gaussian_nb','gradient_boosting',
        'gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','multinomial_nb','passive_aggressive','qda',
        'random_forest','sgd'}
        """
        self.timestamp = time.time()
        self.task_type = task_type
        self.metric = get_metric(metric)
        self.original_data = data.copy_()
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.trial_num = trial_num
        self.n_jobs = n_jobs
        self.alpha = 6
        self.B = 0.01
        self.seed = seed
        self.shared_mode = share_feature
        self.output_dir = output_dir
        np.random.seed(self.seed)

        # Best configuration.
        self.optimal_algo_id = None
        self.nbest_algo_ids = None
        self.best_lower_bounds = None

        # Set up backend.
        self.dataset_name = dataset_name
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        logger_name = "%s-%s" % (__class__.__name__, self.dataset_name)
        self.logger = self._get_logger(logger_name)

        # Bandit settings.
        self.incumbent_perf = -1.
        self.arms = classifier_ids
        self.include_algorithms = classifier_ids
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()
        self.fe_datanodes = dict()
        self.eval_type = eval_type
        self.fe_algo = fe_algo
        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            self.fe_datanodes[arm] = list()
            self.sub_bandits[arm] = SecondLayerBandit(
                self.task_type, arm, self.original_data,
                metric=self.metric,
                output_dir=output_dir,
                per_run_time_limit=per_run_time_limit,
                share_fe=self.shared_mode,
                seed=self.seed,
                eval_type=eval_type,
                dataset_id=dataset_name,
                n_jobs=self.n_jobs,
                fe_algo=fe_algo,
                mth=opt_algo,
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

    def optimize(self, strategy='explore_first'):
        if strategy == 'explore_first':
            self.optimize_explore_first()
        elif strategy == 'exp3':
            self.optimize_exp3()
        elif strategy == 'sw_ucb':
            self.optimize_sw_ucb()
        elif strategy == 'discounted_ucb':
            self.optimize_discounted_ucb()
        elif strategy == 'sw_ts':
            self.optimize_sw_ts()
        else:
            raise ValueError('Unsupported optimization method: %s!' % strategy)
        self.stats = self.fetch_ensemble_members()
        if self.ensemble_method is not None:
            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=self.stats,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=self.original_data)

        # Fit the best model
        ### Local_inc or inc ###

        best_algo_id = None
        best_perf = float("-INF")
        for algo_id in self.include_algorithms:
            if self.sub_bandits[algo_id].incumbent_perf > best_perf:
                best_perf = self.sub_bandits[algo_id].incumbent_perf
                best_algo_id = algo_id

        if self.fe_algo == 'tree_based':
            self.best_data_node = self.sub_bandits[best_algo_id].inc['fe']
        else:
            self.best_data_node = self.stats[best_algo_id]['train_data_list'][0]

        self.fe_optimizer = self.sub_bandits[best_algo_id].optimizer['fe']
        best_config = self.sub_bandits[best_algo_id].inc['hpo']
        best_estimator = fetch_predict_estimator(self.task_type, best_config, self.best_data_node.data[0],
                                                 self.best_data_node.data[1])
        with open(os.path.join(self.output_dir, '%s-best_model' % self.timestamp), 'wb') as f:
            pkl.dump(best_estimator, f)

    def _best_predict(self, test_data: DataNode):
        # Check the validity of feature engineering.
        _train_data = self.fe_optimizer.apply(self.original_data, self.best_data_node, phase='train')
        # assert _train_data == self.best_data_node
        test_data_node = self.fe_optimizer.apply(test_data, self.best_data_node)
        with open(os.path.join(self.output_dir, '%s-best_model' % self.timestamp), 'rb') as f:
            estimator = pkl.load(f)
        return estimator.predict(test_data_node.data[0])

    def _es_predict(self, test_data: DataNode):
        if self.ensemble_method is not None:
            if self.es is None:
                raise AttributeError("AutoML is not fitted!")
        pred = self.es.predict(test_data, self.sub_bandits)
        if self.task_type in CLS_TASKS:
            return np.argmax(pred, axis=-1)
        else:
            return pred

    def _predict(self, test_data: DataNode):
        if self.ensemble_method is not None:
            if self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data, self.sub_bandits)
        else:
            test_data_node = self.fe_optimizer.apply(test_data, self.best_data_node)
            with open(os.path.join(self.output_dir, '%s-best_model' % self.timestamp), 'rb') as f:
                estimator = pkl.load(f)
            if self.task_type in CLS_TASKS:
                return estimator.predict_proba(test_data_node.data[0])
            else:
                return estimator.predict(test_data_node.data[0])

    def predict_proba(self, test_data: DataNode):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data)

    def predict(self, test_data: DataNode):
        if self.task_type in CLS_TASKS:
            pred = self._predict(test_data)
            return np.argmax(pred, axis=-1)
        else:
            return self._predict(test_data)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            self.logger.info('Metric is set to accuracy_score by default!')
            metric_func = accuracy_score
        y_pred = self.predict(test_data)
        return metric_func(test_data.data[1], y_pred)

    def optimize_sw_ts(self):
        K = len(self.arms)
        C = 4
        # Initialize the parameters.
        params = [0, 1] * K
        arm_cnts = np.zeros(K)

        for iter_id in range(1, 1 + self.trial_num):
            if iter_id <= C * K:
                arm_idx = (iter_id - 1) % K
                _arm = self.arms[arm_idx]
            else:
                samples = list()
                for _id in range(K):
                    idx = 2 * _id
                    sample = norm.rvs(loc=params[idx], scale=params[idx + 1])
                    sample = params[idx] if sample < params[idx] else sample
                    samples.append(sample)
                arm_idx = np.argmax(samples)
                _arm = self.arms[arm_idx]

                l1 = '\nIn the %d-th iteration: ' % iter_id
                l1 += '\nmu: %s' % str([val for idx, val in enumerate(params) if idx % 2 == 0])
                l1 += '\nstd: %s' % str([val for idx, val in enumerate(params) if idx % 2 == 1])
                l1 += '\nI_t: %s\n' % str(samples)
                self.logger.info(l1)

            arm_cnts[arm_idx] += 1
            self.logger.info('PULLING %s in %d-th round' % (_arm, iter_id))
            reward = self.sub_bandits[_arm].play_once()

            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))

            # Update parameters in Thompson Sampling.
            for _id in range(K):
                _rewards = self.rewards[self.arms[_id]][-C:]
                _mu = np.mean(_rewards)
                _std = np.std(_rewards)
                idx = 2 * _id
                params[idx], params[idx + 1] = _mu, _std

    def optimize_sw_ucb(self):
        # Initialize the parameters.
        K = len(self.arms)
        N_t = np.zeros(K)
        X_t = np.zeros(K)
        c_t = np.zeros(K)
        gamma = 0.9
        tau = 2 * K
        B = 0.1
        epsilon = 0.1
        action_ids = list()

        for iter_id in range(1, 1 + self.trial_num):
            if iter_id <= K:
                arm_idx = iter_id - 1
                _arm = self.arms[arm_idx]
            else:
                # Choose the arm according to SW-UCB.
                sw = np.max([0, iter_id - tau + 1])
                _action_ids, _rewards = action_ids[sw:], self.final_rewards[sw:]
                _It = np.zeros(K)
                for id in range(K):
                    past_rewards = [item for idx, item in zip(_action_ids, _rewards) if idx == id]
                    X_sum = 0. if len(past_rewards) == 0 else np.sum(past_rewards)
                    X_t[id] = 1. / N_t[id] * X_sum
                    c = np.log(np.min([iter_id, tau]))
                    c_t[id] = B * np.sqrt(epsilon * c / N_t[id])
                    _It[id] = X_t[id] + c_t[id]
                arm_idx = np.argmax(_It)
                _arm = self.arms[arm_idx]

                l1 = '\nIn the %d-th iteration: ' % iter_id
                l1 += '\nX_t: %s' % str(X_t)
                l1 += '\nc_t: %s' % str(c_t)
                l1 += '\nI_t: %s\n' % str(_It)
                self.logger.info(l1)

            action_ids.append(arm_idx)
            self.logger.info('PULLING %s in %d-th round' % (_arm, iter_id))
            reward = self.sub_bandits[_arm].play_once()

            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))

            # Update N_t.
            for id in range(K):
                N_t[id] = N_t[id] * gamma
                if id == arm_idx:
                    N_t[id] += 1

        result = list()
        for _arm in self.arms:
            val = 0. if len(self.rewards[_arm]) == 0 else np.max(self.rewards[_arm])
            result.append(val)
        self.optimal_algo_id = self.arms[np.argmax(result)]
        _best_perf = np.max(result)

        threshold = 0.96
        idxs = np.argsort(-np.array(result))[:3]
        _algo_ids = [self.arms[idx] for idx in idxs]
        self.nbest_algo_ids = list()
        for _idx, _arm in zip(idxs, _algo_ids):
            if result[_idx] >= threshold * _best_perf:
                self.nbest_algo_ids.append(_arm)

        return self.rewards

    def optimize_discounted_ucb(self):
        # Initialize the parameters.
        K = len(self.arms)
        N_t = np.zeros(K)
        X_t = np.zeros(K)
        X_ac = np.zeros(K)
        c_t = np.zeros(K)
        gamma = 0.95
        # 0.1 0.1
        B = self.B
        epsilon = 1.

        for iter_id in range(1, 1 + self.trial_num):
            if iter_id <= K:
                arm_idx = iter_id - 1
                _arm = self.arms[arm_idx]
            else:
                # Choose the arm according to D-UCB.
                _It = np.zeros(K)
                n_t = np.sum(N_t)
                for id in range(K):
                    X_t[id] = 1. / N_t[id] * X_ac[id]
                    c_t[id] = 2 * B * np.sqrt(epsilon * np.log(n_t) / N_t[id])
                    _It[id] = X_t[id] + c_t[id]
                arm_idx = np.argmax(_It)
                _arm = self.arms[arm_idx]

                l1 = '\nIn the %d-th iteration: ' % iter_id
                l1 += '\nX_t: %s' % str(X_t)
                l1 += '\nc_t: %s' % str(c_t)
                l1 += '\nI_t: %s\n' % str(_It)
                self.logger.info(l1)

            self.logger.info('PULLING %s in %d-th round' % (_arm, iter_id))
            reward = self.sub_bandits[_arm].play_once()

            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))

            # Update N_t.
            for id in range(K):
                N_t[id] *= gamma
                X_ac[id] *= gamma
                if id == arm_idx:
                    N_t[id] += 1
                    X_ac[id] += reward

    def optimize_exp3(self):
        # Initialize the parameters.
        K = len(self.arms)
        p_distri = np.ones(K) / K
        estimated_cumulative_loss = np.zeros(K)

        for iter_id in range(1, 1 + self.trial_num):
            eta = np.sqrt(np.log(K) / (iter_id * K))
            # Draw an arm according to p distribution.
            arm_idx = np.random.choice(K, 1, p=p_distri)[0]
            _arm = self.arms[arm_idx]

            self.logger.info('PULLING %s in %d-th round' % (_arm, iter_id))
            reward = self.sub_bandits[_arm].play_once()
            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))

            loss = 1. - reward
            estimated_loss = loss / p_distri[arm_idx]
            estimated_cumulative_loss[arm_idx] += estimated_loss
            # Update the probability distribution over arms.
            tmp_weights = np.exp(-eta * estimated_cumulative_loss)
            p_distri = tmp_weights / np.sum(tmp_weights)
        return self.rewards

    def optimize_explore_first(self):
        # Initialize the parameters.
        arm_num = len(self.arms)
        arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)
        _iter_id = 0
        assert arm_num * self.alpha <= self.trial_num

        while _iter_id < self.trial_num:
            if _iter_id < arm_num * self.alpha:
                _arm = self.arms[_iter_id % arm_num]
                self.logger.info('PULLING %s in %d-th round' % (_arm, _iter_id))
                reward = self.sub_bandits[_arm].play_once()

                self.rewards[_arm].append(reward)
                self.action_sequence.append(_arm)
                self.final_rewards.append(reward)
                self.time_records.append(time.time() - self.start_time)
                if reward > self.incumbent_perf:
                    self.incumbent_perf = reward
                    self.optimal_algo_id = _arm
                self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))
                _iter_id += 1
            else:
                # Pull each arm in the candidate once.
                for _arm in arm_candidate:
                    self.logger.info('PULLING %s in %d-th round' % (_arm, _iter_id))
                    reward = self.sub_bandits[_arm].play_once()
                    self.rewards[_arm].append(reward)
                    self.action_sequence.append(_arm)
                    self.final_rewards.append(reward)
                    self.time_records.append(time.time() - self.start_time)

                    self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))
                    _iter_id += 1

            if _iter_id >= arm_num * self.alpha:
                # Update the upper/lower bound estimation.
                upper_bounds, lower_bounds = list(), list()
                for _arm in arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha]) / self.alpha
                    upper_bound = np.min([1.0, rewards[-1] + slope * (self.trial_num - _iter_id)])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])
                    self.best_lower_bounds[self.arms.index(_arm)] = rewards[-1]

                # Reject the sub-optimal arms.
                n = len(arm_candidate)
                flags = [False] * n
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if upper_bounds[i] < lower_bounds[j]:
                                flags[i] = True

                if np.sum(flags) == n:
                    self.logger.error('Removing all the arms simultaneously!')
                self.logger.info('Candidates : %s' % ','.join(arm_candidate))
                self.logger.info('Upper bound: %s' % ','.join(['%.4f' % val for val in upper_bounds]))
                self.logger.info('Lower bound: %s' % ','.join(['%.4f' % val for val in lower_bounds]))
                self.logger.info('Remove Arms: %s' % [item for idx, item in enumerate(arm_candidate) if flags[idx]])

                # Update the arm_candidates.
                arm_candidate = [item for index, item in enumerate(arm_candidate) if not flags[index]]

            if _iter_id >= self.trial_num - 1:
                _lower_bounds = self.best_lower_bounds.copy()
                algo_idx = np.argmax(_lower_bounds)
                self.optimal_algo_id = self.arms[algo_idx]
                _best_perf = _lower_bounds[algo_idx]

                threshold = 0.96
                idxs = np.argsort(-_lower_bounds)[:3]
                _algo_ids = [self.arms[idx] for idx in idxs]
                self.nbest_algo_ids = list()
                for _idx, _arm in zip(idxs, _algo_ids):
                    if _lower_bounds[_idx] >= threshold * _best_perf:
                        self.nbest_algo_ids.append(_arm)
                assert len(self.nbest_algo_ids) > 0

                self.logger.info('=' * 50)
                self.logger.info('Best_algo_perf:    %s' % str(_best_perf))
                self.logger.info('Best_algo_id:      %s' % str(self.optimal_algo_id))
                self.logger.info('Arm candidates:    %s' % str(self.arms))
                self.logger.info('Best_lower_bounds: %s' % str(self.best_lower_bounds))
                self.logger.info('Nbest_algo_ids   : %s' % str(self.nbest_algo_ids))
                self.logger.info('=' * 50)

        return self.final_rewards

    def _get_logger(self, name):
        logger_name = 'AutomlToolkit_%s' % name
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def __del__(self):
        for _arm in self.arms:
            del self.sub_bandits[_arm].optimizer

    def fetch_ensemble_members(self, threshold=0.95):
        stats = dict()
        stats['include_algorithms'] = self.include_algorithms
        stats['split_seed'] = self.seed
        best_perf = float('-INF')
        for algo_id in self.nbest_algo_ids:
            best_perf = max(best_perf, self.sub_bandits[algo_id].incumbent_perf)
        for algo_id in self.nbest_algo_ids:
            data = dict()
            fe_optimizer = self.sub_bandits[algo_id].optimizer['fe']
            hpo_optimizer = self.sub_bandits[algo_id].optimizer['hpo']

            if self.fe_algo == 'bo':
                data_candidates = fe_optimizer.fetch_nodes(10)
                train_data_candidates = list()
                # Check the dimensions.
                labels = self.original_data.data[1]
                for tmp_data in data_candidates:
                    equal_flag = (tmp_data.data[1] == labels)
                    if not isinstance(equal_flag, bool):
                        assert equal_flag.all()
                        train_data_candidates.append(tmp_data)
                # TODO: what about empty train_data_candidates.
            else:
                train_data_candidates = self.sub_bandits[algo_id].local_hist['fe']
            # for _feature_set in fe_optimizer.features_hist:
            #     if _feature_set not in train_data_candidates:
            #         train_data_candidates.append(_feature_set)

            # Remove duplicates.
            train_data_list = list()
            for item in train_data_candidates:
                if item not in train_data_list:
                    train_data_list.append(item)

            data['train_data_list'] = train_data_list
            print(algo_id, len(train_data_list))

            # Build hyperparameter configuration candidates.
            configs = hpo_optimizer.configs
            perfs = hpo_optimizer.perfs
            best_configs = self.sub_bandits[algo_id].local_hist['hpo']
            best_configs = list(set(best_configs))
            if self.metric._sign > 0:
                threshold = best_perf * threshold
            else:
                threshold = best_perf / threshold

            for idx in np.argsort(-np.array(perfs)):
                if perfs[idx] >= threshold and configs[idx] not in best_configs:
                    best_configs.append(configs[idx])
                # TODO: self.ensemble_size/len is not a good option.
                config_num = 5
                if len(best_configs) >= config_num:
                    break
            data['configurations'] = best_configs

            stats[algo_id] = data
        return stats
