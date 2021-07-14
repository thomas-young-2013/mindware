import time
import numpy as np
from copy import deepcopy
from ConfigSpace import ConfigurationSpace, Constant
from mindware.utils.constant import MAX_INT
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.blocks.abstract_block import AbstractBlock


class ConditioningBlock(AbstractBlock):
    def __init__(self, node_list, node_index,
                 task_type, timestamp,
                 fe_config_space: ConfigurationSpace,
                 cash_config_space: ConfigurationSpace,
                 data: DataNode,
                 fixed_config=None,
                 time_limit=None,
                 trial_num=0,
                 metric='acc',
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 per_run_time_limit=300,
                 output_dir="logs",
                 dataset_name='default_dataset',
                 eval_type='holdout',
                 resampling_params=None,
                 n_jobs=1,
                 seed=1):
        """
        :param classifier_ids: subset of {'adaboost','bernoulli_nb','decision_tree','extra_trees','gaussian_nb','gradient_boosting',
        'gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','multinomial_nb','passive_aggressive','qda',
        'random_forest','sgd'}
        """
        super(ConditioningBlock, self).__init__(node_list, node_index, task_type, timestamp,
                                                fe_config_space, cash_config_space, data,
                                                fixed_config=fixed_config,
                                                time_limit=time_limit,
                                                trial_num=trial_num,
                                                metric=metric,
                                                ensemble_method=ensemble_method,
                                                ensemble_size=ensemble_size,
                                                per_run_time_limit=per_run_time_limit,
                                                output_dir=output_dir,
                                                dataset_name=dataset_name,
                                                eval_type=eval_type,
                                                resampling_params=resampling_params,
                                                n_jobs=n_jobs,
                                                seed=seed)

        # Best configuration.
        self.optimal_arm = None
        self.best_lower_bounds = None

        # Bandit settings.
        self.alpha = 4
        self.arms = list(cash_config_space.get_hyperparameter('algorithm').choices)
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()

        self.arm_cost_stats = dict()
        for _arm in self.arms:
            self.arm_cost_stats[_arm] = list()

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()

            hps = cash_config_space.get_hyperparameters()
            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', arm))
            for hp in hps:
                if hp.name.split(':')[0] == arm:
                    cs.add_hyperparameter(hp)

            # Add active conditions
            conds = cash_config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass

            # Add active forbidden clauses
            forbids = cash_config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass

            from mindware.blocks.block_utils import get_node_type
            child_type = get_node_type(node_list, node_index + 1)
            self.sub_bandits[arm] = child_type(
                node_list, node_index + 1, task_type, timestamp,
                deepcopy(fe_config_space), deepcopy(cs), data.copy_(),
                fixed_config=fixed_config,
                time_limit=time_limit,
                metric=metric,
                ensemble_method=ensemble_method,
                ensemble_size=ensemble_size,
                per_run_time_limit=per_run_time_limit,
                output_dir=output_dir,
                dataset_name=dataset_name,
                eval_type=eval_type,
                resampling_params=resampling_params,
                n_jobs=n_jobs,
                seed=seed
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

        # Initialize the parameters.
        self.pull_cnt = 0
        self.pick_id = 0
        self.update_cnt = 0
        arm_num = len(self.arms)
        self.optimal_algo_id = None
        self.arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)
        _iter_id = 0
        if self.time_limit is None:
            if arm_num * self.alpha > self.trial_num:
                raise ValueError('Trial number should be larger than %d.' % (arm_num * self.alpha))
        else:
            self.trial_num = MAX_INT

    def iterate(self, trial_num=10):
        # Search for an arm that is not early-stopped.
        while self.sub_bandits[self.arm_candidate[self.pick_id]].early_stop_flag and \
                self.pick_id < len(self.arm_candidate):
            self.pick_id += 1

        if self.pick_id < len(self.arm_candidate):
            # Pull the arm.
            arm_to_pull = self.arm_candidate[self.pick_id]
            self.logger.info('Optimize %s in the %d-th iteration' % (arm_to_pull, self.pull_cnt))
            _start_time = time.time()
            reward = self.sub_bandits[arm_to_pull].iterate(trial_num=trial_num)

            # Update results after each iteration
            self.arm_cost_stats[arm_to_pull].append(time.time() - _start_time)
            if reward > self.incumbent_perf:
                self.optimal_algo_id = arm_to_pull
                self.incumbent_perf = reward
                self.incumbent = self.sub_bandits[arm_to_pull].incumbent
            self.eval_dict.update(self.sub_bandits[arm_to_pull].eval_dict)
            self.rewards[arm_to_pull].append(reward)
            self.action_sequence.append(arm_to_pull)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            # self.logger.info('The best performance found for %s is %.4f' % (arm_to_pull, reward))
            self.pull_cnt += 1
            self.pick_id += 1

            # Logger output
            scores = list()
            for _arm in self.arms:
                scores.append(self.sub_bandits[_arm].incumbent_perf)
            scores = np.array(scores)
            self.logger.info('=' * 50)
            self.logger.info('Node index: %s' % str(self.node_index))
            self.logger.info('Best_algo_perf:  %s' % str(self.incumbent_perf))
            self.logger.info('Best_algo_id:    %s' % str(self.optimal_algo_id))
            self.logger.info('Arm candidates:  %s' % str(self.arms))
            self.logger.info('Best val scores: %s' % str(list(scores)))
            self.logger.info('=' * 50)

        # Eliminate arms after pulling each arm a few times.
        if self.pick_id == len(self.arm_candidate):
            self.update_cnt += 1
            self.pick_id = 0
            # Update the arms until pulling each arm for at least alpha times.
            if self.update_cnt >= self.alpha:
                # Update the upper/lower bound estimation.
                budget_left = max(self.time_limit - (time.time() - self.start_time), 0)
                avg_cost = np.array([np.mean(self.arm_cost_stats[_arm]) for _arm in self.arm_candidate]).mean()
                steps = int(budget_left / avg_cost)
                upper_bounds, lower_bounds = list(), list()

                for _arm in self.arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha]) / self.alpha
                    if self.time_limit is None:
                        steps = self.trial_num - self.pull_cnt
                    upper_bound = np.min([1.0, rewards[-1] + slope * steps])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])
                    self.best_lower_bounds[self.arms.index(_arm)] = rewards[-1]

                # Reject the sub-optimal arms.
                n = len(self.arm_candidate)
                flags = [False] * n
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if upper_bounds[i] < lower_bounds[j]:
                                flags[i] = True

                if np.sum(flags) == n:
                    self.logger.error('Removing all the arms simultaneously!')

                self.logger.info('=' * 50)
                self.logger.info('Node index: %s' % str(self.node_index))
                self.logger.info('Candidates  : %s' % ','.join(self.arm_candidate))
                self.logger.info('Upper bound : %s' % ','.join(['%.4f' % val for val in upper_bounds]))
                self.logger.info('Lower bound : %s' % ','.join(['%.4f' % val for val in lower_bounds]))
                self.logger.info(
                    'Arms removed: %s' % [item for idx, item in enumerate(self.arm_candidate) if flags[idx]])
                self.logger.info('=' * 50)

                # Update arm_candidates.
                self.arm_candidate = [item for index, item in enumerate(self.arm_candidate) if not flags[index]]

        # Update stop flag
        self.early_stop_flag = True
        self.timeout_flag = False
        for _arm in self.arm_candidate:
            if not self.sub_bandits[_arm].early_stop_flag:
                self.early_stop_flag = False
        if self.early_stop_flag:
            self.logger.info(
                "Maximum configuration number met for each arm candidate in conditioning block %s!" % self.node_index)
        for _arm in self.arm_candidate:
            if self.sub_bandits[_arm].timeout_flag:
                self.timeout_flag = True

        return self.incumbent_perf
