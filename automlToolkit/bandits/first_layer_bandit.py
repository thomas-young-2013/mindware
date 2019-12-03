import time
import numpy as np
from typing import List
from automlToolkit.utils.logging_utils import get_logger
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit


class FirstLayerBandit(object):
    def __init__(self, trial_num, classifier_ids: List[str], data: DataNode, seed=1):
        self.original_data = data
        self.seed = seed
        self.alpha = 3
        self.trial_num = trial_num
        self.logger = get_logger(__class__.__name__)
        np.random.seed(self.seed)

        # Bandit settings.
        self.arms = classifier_ids
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            self.sub_bandits[arm] = SecondLayerBandit(arm, data, seed=self.seed)

        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

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

            self.pull_cnt += 1

        return self.final_rewards
