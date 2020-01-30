import os
import time
import numpy as np
from scipy.stats import norm
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from autosklearn.constants import *
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from automlToolkit.components.evaluator import Evaluator
from automlToolkit.components.feature_engineering.transformation_graph import DataNode, TransformationGraph
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.utils.logging_utils import setup_logger, get_logger
from automlToolkit.components.evaluator import get_estimator
from automlToolkit.utils.metalearning import get_meta_learning_configs, get_trans_from_str


class FirstLayerBandit(object):
    def __init__(self, trial_num, classifier_ids: List[str], data: DataNode,
                 per_run_time_limit=300, output_dir=None,
                 dataset_name='default_dataset',
                 tmp_directory='logs',
                 eval_type='cv',
                 share_feature=False,
                 meta_configs=0,
                 n_jobs=1,
                 logging_config=None,
                 seed=1):
        """
        :param classifier_ids: subset of {'adaboost','bernoulli_nb','decision_tree','extra_trees','gaussian_nb','gradient_boosting',
        'gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','multinomial_nb','passive_aggressive','qda',
        'random_forest','sgd'}
        """
        self.original_data = data.copy_()
        self.trial_num = trial_num
        self.n_jobs = n_jobs
        self.alpha = 4
        self.B = 0.01
        self.seed = seed
        self.shared_mode = share_feature
        np.random.seed(self.seed)

        self.dataset_name = dataset_name
        self.meta_configs = None
        self.meta_success = 0
        if meta_configs is not None and isinstance(meta_configs, int) and meta_configs > 0:
            if len(set(self.original_data.data[1])) == 2:
                self.meta_configs = get_meta_learning_configs(self.original_data.data[0],
                                                              self.original_data.data[1],
                                                              BINARY_CLASSIFICATION,
                                                              metric='accuracy',
                                                              num_cfgs=meta_configs)
            else:
                self.meta_configs = get_meta_learning_configs(self.original_data.data[0],
                                                              self.original_data.data[1],
                                                              MULTICLASS_CLASSIFICATION,
                                                              metric='accuracy',
                                                              num_cfgs=meta_configs)

        # Best configuration.
        self.optimal_algo_id = None
        self.nbest_algo_ids = None
        self.best_lower_bounds = None

        # Set up backend.
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        logger_name = "%s-%s" % (__class__.__name__, self.dataset_name)
        self.logger = self._get_logger(logger_name)

        # Bandit settings.
        self.incumbent_perf = -1.
        self.arms = classifier_ids
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()
        self.fe_datanodes = dict()
        self.eval_type = eval_type

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            self.fe_datanodes[arm] = list()
            self.sub_bandits[arm] = SecondLayerBandit(
                arm, self.original_data, output_dir=output_dir,
                per_run_time_limit=per_run_time_limit,
                share_fe=self.shared_mode,
                seed=self.seed,
                eval_type=eval_type,
                dataset_id=dataset_name,
                n_jobs=self.n_jobs
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

    def update_global_datanodes(self, arm):
        self.fe_datanodes[arm] = self.sub_bandits[arm].fetch_local_incumbents()

    def apply_metalearning_fe(self, optimizer, configs):
        rescaler_id = None
        rescaler_dict = {}
        preprocessor_id = None
        preprocessor_dict = {}
        for key in configs:
            key_str = key.split(":")
            # TODO: Imputation, Encoding and Balancing is required in fe_pipeline
            if key_str[0] == 'rescaling':
                if key_str[1] == '__choice__':
                    rescaler_id = configs[key]
                else:
                    rescaler_dict[key_str[2]] = configs[key]
            elif key_str[0] == 'preprocessor':
                if key_str[1] == '__choice__':
                    preprocessor_id = configs[key]
                else:
                    preprocessor_dict[key_str[2]] = configs[key]

        preprocessor_tran = get_trans_from_str(preprocessor_id)(**preprocessor_dict)
        rescaler_tran = get_trans_from_str(rescaler_id)(**rescaler_dict)
        rescale_node = rescaler_tran.operate(optimizer.root_node)
        depth = 2
        if rescaler_tran.type != 0:
            rescale_node.depth = depth
            depth += 1
            rescale_node.trans_hist.append(rescaler_tran.type)
            rescale_node.score = 0
            optimizer.temporary_nodes.append(rescale_node)
            optimizer.graph.add_node(rescale_node)
            # Avoid self-loop.
            if rescaler_tran.type != 0 and optimizer.root_node.node_id != rescale_node.node_id:
                optimizer.graph.add_trans_in_graph(optimizer.root_node, rescale_node, rescaler_tran)

        preprocess_node = preprocessor_tran.operate(rescale_node)
        if preprocessor_tran.type != 0:
            preprocess_node.depth = depth
            preprocess_node.trans_hist.append(preprocessor_tran.type)
            preprocess_node.score = 0
            optimizer.temporary_nodes.append(preprocess_node)
            optimizer.graph.add_node(preprocess_node)
            # Avoid self-loop.
            if preprocessor_tran.type != 0 and rescale_node.node_id != preprocess_node.node_id:
                optimizer.graph.add_trans_in_graph(rescale_node, preprocess_node, preprocessor_tran)

        return preprocess_node

    def evaluate_metalearning_configs(self):
        meta_list = []
        for config in self.meta_configs:
            try:
                config = config.get_dictionary()
                # print(config)
                arm = None
                cs = ConfigurationSpace()
                for key in config:
                    key_str = key.split(":")
                    if key_str[0] == 'classifier':
                        if key_str[1] == '__choice__':
                            arm = config[key]
                            cs.add_hyperparameter(UnParametrizedHyperparameter("estimator", config[key]))
                        else:
                            cs.add_hyperparameter(UnParametrizedHyperparameter(key_str[2], config[key]))

                if arm in self.arms:
                    transformed_node = self.apply_metalearning_fe(self.sub_bandits[arm].optimizer['fe'], config)
                    default_config = cs.sample_configuration(1)
                    hpo_evaluator = Evaluator(None,
                                              data_node=transformed_node, name='hpo',
                                              resampling_strategy=self.eval_type,
                                              seed=self.seed)
                    if arm not in meta_list:
                        meta_list.append(arm)
                        self.sub_bandits[arm].default_config = default_config

                    start_time = time.time()
                    score = 1 - hpo_evaluator(default_config)
                    self.sub_bandits[arm].collect_iter_stats('hpo',
                                                             (score, time.time() - start_time, default_config))
                    self.sub_bandits[arm].inc['fe'] = transformed_node
                    transformed_node.score = score
                    self.final_rewards.append(score)
                    self.action_sequence.append(arm)
                    self.meta_success += 1
            except Exception as e:
                print(e)

    def optimize(self, strategy='explore_first'):
        if self.meta_configs is not None:
            self.evaluate_metalearning_configs()
            self.trial_num -= self.meta_success

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

    def fetch_ensemble_members(self, test_data: DataNode = None, mode=True):
        stats = dict()
        if mode:
            stats['split_seed'] = self.seed
            for algo_id in self.nbest_algo_ids:
                data = dict()
                inc = self.sub_bandits[algo_id].inc
                fe_optimizer = self.sub_bandits[algo_id].optimizer['fe']
                hpo_optimizer = self.sub_bandits[algo_id].optimizer['hpo']
                if test_data is not None:
                    test_data_node = fe_optimizer.apply(test_data, inc['fe'])
                    data['test_dataset'] = test_data_node

                data['train_dataset'] = inc['fe']
                data['configurations'] = hpo_optimizer.configs
                data['performance'] = hpo_optimizer.perfs
                inc_source = self.sub_bandits[algo_id].incumbent_source
                data['inc_source'] = inc_source

                stats[algo_id] = data
        else:
            stats = dict()
            stats['split_seed'] = self.seed
            for algo_id in self.nbest_algo_ids:
                data = dict()
                inc = self.sub_bandits[algo_id].inc
                local_inc = self.sub_bandits[algo_id].local_inc
                fe_optimizer = self.sub_bandits[algo_id].optimizer['fe']
                hpo_optimizer = self.sub_bandits[algo_id].optimizer['hpo']

                train_data_candidates = [inc['fe'], local_inc['fe'], self.sub_bandits[algo_id].original_data]
                for _feature_set in fe_optimizer.features_hist:
                    if _feature_set not in train_data_candidates:
                        train_data_candidates.append(_feature_set)

                train_data_list, test_data_list = list(), list()
                for item in train_data_candidates:
                    if item not in train_data_list:
                        train_data_list.append(item)
                        test_data_node = fe_optimizer.apply(test_data, item)
                        test_data_list.append(test_data_node)

                data['test_data_list'] = test_data_list
                data['train_data_list'] = train_data_list
                print(algo_id, len(train_data_list), len(test_data_list))

                configs = hpo_optimizer.configs
                perfs = hpo_optimizer.perfs
                best_configs = [self.sub_bandits[algo_id].default_config, inc['hpo'], local_inc['hpo']]
                best_configs = list(set(best_configs))
                n_best = 40
                threshold = np.max(self.best_lower_bounds) * 0.
                for idx in np.argsort(-np.array(perfs)):
                    if perfs[idx] >= threshold and configs[idx] not in best_configs:
                        best_configs.append(configs[idx])
                    if len(best_configs) >= n_best:
                        break
                data['configurations'] = best_configs

                stats[algo_id] = data
        return stats

    def predict(self, test_data: DataNode, phase='test'):
        assert phase in ['test', 'validation']
        best_arm = self.optimal_algo_id
        sub_bandit = self.sub_bandits[best_arm]
        fe_optimizer = sub_bandit.optimizer['fe']

        # print('Test data shape', test_data.data[0].shape)
        train_data_node = sub_bandit.inc['fe']
        test_data_node = fe_optimizer.apply(test_data, sub_bandit.inc['fe'])
        config = sub_bandit.inc['hpo']

        # Just test.
        _train_data = fe_optimizer.apply(self.original_data, sub_bandit.inc['fe'])
        assert train_data_node == _train_data

        if phase == 'validation':
            _, tmp_test_data = self.train_valid_split(self.original_data)
            assert tmp_test_data == test_data
            train_data_node, tmp_valid_data = self.train_valid_split(train_data_node)
            _test_data_node = fe_optimizer.apply(tmp_test_data, sub_bandit.inc['fe'])
            assert _test_data_node == test_data_node

            # print(train_data_node.shape, test_data_node.shape)
            # print(tmp_valid_data.shape)
            # print(test_data_node.data[0] == tmp_valid_data.data[0])
            # print(test_data_node.data[1] == tmp_valid_data.data[1])
            #
            # print(sum(test_data_node.data[0] == tmp_valid_data.data[0]))
            # print(sum(test_data_node.data[1] == tmp_valid_data.data[1]))

            assert test_data_node == tmp_valid_data

        # Build the ML estimator.
        _, estimator = get_estimator(config)
        X_train, y_train = train_data_node.data
        X_test, y_test = test_data_node.data
        self.logger.info('X_train/test shapes: %s, %s' % (str(X_train.shape), str(X_test.shape)))

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(test_data_node.data[0])
        if phase == 'validation':
            print('=' * 50)
            print(accuracy_score(y_pred, y_test))
            print('=' * 50)
        return y_pred

    def train_valid_split(self, node: DataNode):
        X, y = node.copy_().data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
        train_data = DataNode(data=[X_train, y_train], feature_type=node.feature_types.copy())
        valid_data = DataNode(data=[X_val, y_val], feature_type=node.feature_types.copy())
        return train_data, valid_data

    def validate(self, metric_func=None):
        if metric_func is None:
            metric_func = accuracy_score
        _, valid_data = self.train_valid_split(self.original_data)
        y_pred = self.predict(valid_data, phase='validation')
        return metric_func(valid_data.data[1], y_pred)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            from sklearn.metrics.classification import accuracy_score
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

                if self.shared_mode:
                    self.update_global_datanodes(_arm)

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

                    if self.shared_mode:
                        self.update_global_datanodes(_arm)

                    self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))
                    _iter_id += 1

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

            # Sync the features data nodes.
            if self.shared_mode and _iter_id >= arm_num * self.alpha \
                    and _iter_id % 2 == 0 and len(arm_candidate) > 1:
                self.logger.info('Start to SYNC features among all arms!')
                data_nodes = list()
                for _arm in arm_candidate:
                    data_nodes.extend(self.fe_datanodes[_arm])
                # Sample #beam_size-1 nodes.
                beam_size = self.sub_bandits[arm_candidate[0]].optimizer['fe'].beam_width
                # TODO: how to generate the global nodes.
                global_nodes = TransformationGraph.sort_nodes_by_score(data_nodes)[:beam_size - 1]
                for _arm in arm_candidate:
                    self.sub_bandits[_arm].sync_global_incumbents(global_nodes)

        return self.final_rewards

    def _get_logger(self, name):
        logger_name = 'AutomlToolkit_%s' % name
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)
