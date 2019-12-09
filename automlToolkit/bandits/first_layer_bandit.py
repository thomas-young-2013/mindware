import os
import time
import numpy as np
from typing import List
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.utils.logging_utils import setup_logger, get_logger


class FirstLayerBandit(object):
    def __init__(self, trial_num, classifier_ids: List[str], data: DataNode,
                 per_run_time_limit=300, output_dir=None,
                 dataset_name='default_dataset_name',
                 tmp_directory='logs', logging_config=None, seed=1):
        self.original_data = data
        self.trial_num = trial_num
        self.alpha = 4
        self.seed = seed
        self.shared_mode = False
        np.random.seed(self.seed)

        self.dataset_name = dataset_name

        # Set up backend.
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        self.logger = self._get_logger(self.dataset_name)

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
                per_run_time_limit=per_run_time_limit, seed=self.seed
            )

        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

    def update_global_datanodes(self, arm):
        self.fe_datanodes[arm] = self.sub_bandits[arm].fetch_local_incumbents()

    def optimize(self):
        arm_num = len(self.arms)
        arm_candidate = self.arms.copy()
        while self.pull_cnt < self.trial_num:

            if self.pull_cnt < arm_num * self.alpha:
                _arm = self.arms[self.pull_cnt % arm_num]
                self.logger.info('Pulling %s in %d-th round' % (_arm, self.pull_cnt))
                reward = self.sub_bandits[_arm].play_once()
                self.rewards[_arm].append(reward)
                self.action_sequence.append(_arm)
                self.final_rewards.append(reward)
                if self.shared_mode:
                    self.update_global_datanodes(_arm)
                self.time_records.append(time.time() - self.start_time)
                self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))
            else:
                # Pull each arm in the candidate once.
                for _arm in arm_candidate:
                    self.logger.info('Pulling %s in %d-th round' % (_arm, self.pull_cnt))
                    reward = self.sub_bandits[_arm].play_once()
                    self.rewards[_arm].append(reward)
                    self.action_sequence.append(_arm)
                    self.final_rewards.append(reward)
                    if self.shared_mode:
                        self.update_global_datanodes(_arm)
                    self.time_records.append(time.time() - self.start_time)
                    self.logger.info('Rewards for pulling %s = %.4f' % (_arm, reward))

                # Update the upper/lower bound estimation.
                upper_bounds, lower_bounds = list(), list()
                for _arm in arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha])/self.alpha
                    upper_bound = rewards[-1] + slope*(self.trial_num - self.pull_cnt)
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

                self.logger.info('Remove Arms: %s' %
                                 [item for idx, item in enumerate(arm_candidate) if flags[idx]])
                # Update the arm_candidates.
                arm_candidate = [item for index, item in enumerate(arm_candidate) if not flags[index]]

            # Sync the features data nodes.
            if self.shared_mode:
                data_nodes = list()
                scores = list()
                for _arm in arm_candidate:
                    data_nodes.extend(self.fe_datanodes[_arm])
                    scores.extend([node.score for node in self.fe_datanodes[_arm]])
                # Sample #beam_size-1 nodes.
                beam_size = self.sub_bandits[arm_candidate[0]].optimizer['fe'].beam_width
                # TODO: how to generate the global nodes.
                idxs = np.argsort(-np.asarray(scores))[: beam_size - 1]
                global_nodes = [data_nodes[idx] for idx in idxs]
                for _arm in arm_candidate:
                    self.sub_bandits[_arm].sync_global_incumbents(global_nodes)

            self.pull_cnt += 1

        return self.final_rewards

    def _get_logger(self, name):
        logger_name = 'AutomlToolkit_%d_%s' % (self.seed, name)
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)
