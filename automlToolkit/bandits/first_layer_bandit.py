import os
import time
import numpy as np
from scipy.stats import norm
from typing import List
from sklearn.model_selection import train_test_split
from automlToolkit.components.feature_engineering.transformation_graph import DataNode, TransformationGraph
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.utils.logging_utils import setup_logger, get_logger
from automlToolkit.components.evaluator import get_estimator


class FirstLayerBandit(object):
    def __init__(self, trial_num, classifier_ids: List[str], data: DataNode,
                 per_run_time_limit=300, output_dir=None,
                 dataset_name='default_dataset_name',
                 tmp_directory='logs',
                 eval_type='cv',
                 share_feature=False, logging_config=None, seed=1):
        self.original_data = data.copy_()
        self.trial_num = trial_num
        self.alpha = 4
        self.B = 0.01
        self.seed = seed
        self.shared_mode = share_feature
        np.random.seed(self.seed)

        self.dataset_name = dataset_name

        # Best configuration.
        self.optimal_algo_id = None

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

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            self.fe_datanodes[arm] = list()
            self.sub_bandits[arm] = SecondLayerBandit(
                arm, data, output_dir=output_dir,
                per_run_time_limit=per_run_time_limit,
                share_fe=self.shared_mode,
                seed=self.seed,
                eval_type=eval_type,
                dataset_id=dataset_name
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

    def update_global_datanodes(self, arm):
        self.fe_datanodes[arm] = self.sub_bandits[arm].fetch_local_incumbents()

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

    def fetch_ensemble_members(self, test_data: DataNode = None):
        stats = dict()
        stats['split_seed'] = self.seed
        for algo_id in self.arms:
            data = dict()
            fe_optimizer = self.sub_bandits[algo_id].optimizer['fe']
            hpo_optimizer = self.sub_bandits[algo_id].optimizer['hpo']
            if test_data is not None:
                try:
                    test_data_node = fe_optimizer.apply(test_data)
                    data['test_dataset'] = test_data_node
                except Exception as e:
                    self.logger.error('CRITICAL ERROR!')
                    self.logger.error(str(e))
                    data['test_dataset'] = test_data
                    return None

            data['train_dataset'] = fe_optimizer.incumbent
            data['configurations'] = hpo_optimizer.configs
            data['performance'] = hpo_optimizer.perfs
            inc_source = self.sub_bandits[algo_id].incumbent_source
            if inc_source == 'hpo':
                data['train_dataset'] = self.original_data
                if test_data is not None:
                    data['test_dataset'] = test_data
            data['inc_source'] = inc_source

            stats[algo_id] = data
        return stats

    def predict(self, test_data: DataNode, phase='test'):
        assert phase in ['test', 'validation']
        best_arm = self.optimal_algo_id
        sub_bandit = self.sub_bandits[best_arm]
        # Get the best features found.
        fe_optimizer = sub_bandit.optimizer['fe']
        # Get the best configuration found.
        hpo_optimizer = sub_bandit.optimizer['hpo']

        # Build the ML estimator.
        inc_source = sub_bandit.incumbent_source
        if inc_source == 'hpo':
            test_data_node = test_data
            train_data_node = self.original_data
            config = hpo_optimizer.incumbent_config
        elif inc_source == 'fe':
            test_data_node = fe_optimizer.apply(test_data)
            train_data_node = fe_optimizer.incumbent
            config = sub_bandit.config_space.get_default_configuration()
        else:
            test_data_node = fe_optimizer.apply(test_data)
            train_data_node = fe_optimizer.incumbent
            config = hpo_optimizer.incumbent_config

        _, estimator = get_estimator(config)
        X_train, y_train = train_data_node.data
        X_test, _ = test_data_node.data
        print(X_train.shape, X_test.shape)

        if phase == 'validation':
            X, y = train_data_node.data
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=self.seed, stratify=y)

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(test_data_node.data[0])
        return y_pred

    def validate(self, metric_func=None):
        if metric_func is None:
            from sklearn.metrics.classification import accuracy_score
            metric_func = accuracy_score

        X, y = self.original_data.data
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y)
        valid_data = DataNode(data=[X_val, y_val], feature_type=self.original_data.feature_types.copy())

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
            if iter_id <= C*K:
                arm_idx = (iter_id-1) % K
                _arm = self.arms[arm_idx]
            else:
                samples = list()
                for _id in range(K):
                    idx = 2 * _id
                    sample = norm.rvs(loc=params[idx], scale=params[idx+1])
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
                params[idx], params[idx+1] = _mu, _std

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
                arm_idx = iter_id-1
                _arm = self.arms[arm_idx]
            else:
                # Choose the arm according to SW-UCB.
                sw = np.max([0, iter_id - tau + 1])
                _action_ids, _rewards = action_ids[sw:], self.final_rewards[sw:]
                _It = np.zeros(K)
                for id in range(K):
                    past_rewards = [item for idx, item in zip(_action_ids, _rewards) if idx == id]
                    X_sum = 0. if len(past_rewards) == 0 else np.sum(past_rewards)
                    X_t[id] = 1./N_t[id] * X_sum
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
                arm_idx = iter_id-1
                _arm = self.arms[arm_idx]
            else:
                # Choose the arm according to D-UCB.
                _It = np.zeros(K)
                n_t = np.sum(N_t)
                for id in range(K):
                    X_t[id] = 1./N_t[id] * X_ac[id]
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
            eta = np.sqrt(np.log(K) / (iter_id*K))
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
            tmp_weights = np.exp(-eta*estimated_cumulative_loss)
            p_distri = tmp_weights/np.sum(tmp_weights)
        return self.rewards

    def optimize_explore_first(self):
        # Initialize the parameters.
        arm_num = len(self.arms)
        arm_candidate = self.arms.copy()
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
                    slope = (rewards[-1] - rewards[-self.alpha])/self.alpha
                    upper_bound = np.min([1.0, rewards[-1] + slope*(self.trial_num - _iter_id)])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])

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

                if n == 1:
                    self.optimal_algo_id = arm_candidate[0]
                elif n > 1:
                    algo_idx = np.argmax(upper_bounds)
                    self.optimal_algo_id = arm_candidate[algo_idx]
                else:
                    raise ValueError('The size of candidate set is zero!')

                # Update the arm_candidates.
                arm_candidate = [item for index, item in enumerate(arm_candidate) if not flags[index]]
            
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
