import os
import time
import numpy as np
from typing import List
from automlToolkit.components.feature_engineering.transformation_graph import DataNode, TransformationGraph
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.utils.logging_utils import setup_logger, get_logger


class FirstLayerBandit(object):
    def __init__(self, trial_num, classifier_ids: List[str], data: DataNode,
                 per_run_time_limit=300, output_dir=None,
                 dataset_name='default_dataset_name',
                 tmp_directory='logs',
                 share_feature=False, logging_config=None, seed=1):
        self.original_data = data
        self.trial_num = trial_num
        self.alpha = 4
        self.seed = seed
        self.shared_mode = share_feature
        np.random.seed(self.seed)

        self.dataset_name = dataset_name

        # Set up backend.
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        logger_name = "%s-%s" % (__class__.__name__, self.dataset_name)
        self.logger = self._get_logger(logger_name)
        
        # Bandit settings.
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
        else:
            raise ValueError('Unsupported optimization method: %s!' % strategy)

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
        logger_name = 'AutomlToolkit_%d_%s' % (self.seed, name)
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)
