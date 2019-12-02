import numpy as np
from automlToolkit.components.evaluator import Evaluator
from automlToolkit.utils.logging_utils import get_logger
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer


class SecondLayerBandit(object):
    def __init__(self, classifier_id: str, data: DataNode, seed=1):
        self.classifier_id = classifier_id
        self.original_data = data
        self.seed = seed
        self.sliding_window_size = 3
        self.logger = get_logger(__class__.__name__)
        np.random.seed(self.seed)

        # Bandit settings.
        self.arms = ['fe', 'hpo']
        self.rewards = dict()
        self.optimizer = dict()
        self.evaluation_cost = dict()
        self.inc = dict()
        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()

        from autosklearn.pipeline.components.classification import _classifiers
        clf_class = _classifiers[classifier_id]
        cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", classifier_id)
        cs.add_hyperparameter(model)

        # Build the HPO component.
        hpo_evaluator = Evaluator(cs.get_default_configuration(), data_node=data, name='hpo', seed=self.seed)
        self.optimizer['hpo'] = SMACOptimizer(hpo_evaluator, cs, trials_per_iter=10, seed=self.seed)

        # Build the Feature Engineering component.
        fe_evaluator = Evaluator(cs.get_default_configuration(), name='fe', seed=self.seed)
        self.optimizer['fe'] = EvaluationBasedOptimizer(self.original_data, fe_evaluator, self.seed)

    def collect_iter_stats(self, _arm, results):
        score, iter_cost, config = results
        self.rewards[_arm].append(score)
        self.evaluation_cost[_arm].append(iter_cost)
        self.inc[_arm] = config

    def optimize(self):
        if self.pull_cnt == 0:
            # First pull each arm #sliding_window_size times.
            for _ in range(self.sliding_window_size):
                for _arm in self.arms:
                    self.logger.info('Pulling arm: %s' % _arm)
                    results = self.optimizer[_arm].iterate()
                    self.collect_iter_stats(_arm, results)
                    self.pull_cnt += 1
                    self.action_sequence.append(_arm)
            self.final_rewards.append(self.optimizer['fe'].baseline_score)
        else:
            imp_values = list()
            for _arm in self.arms:
                increasing_rewards = list()
                for _reward in self.rewards[_arm]:
                    if len(increasing_rewards) == 0:
                        increasing_rewards.append(_reward)
                    else:
                        inc_reward = increasing_rewards[-1] if _reward <= increasing_rewards[-1] else _reward
                        increasing_rewards.append(inc_reward)

                impv = list()
                for idx in range(1, len(increasing_rewards)):
                    impv.append(increasing_rewards[idx] - increasing_rewards[idx - 1])
                imp_values.append(np.mean(impv[-self.sliding_window_size:]))

            self.logger.info('Imp values: %s' % imp_values)
            if np.sum(imp_values) == 0:
                # If ties, break randomly.
                arm_picked = np.random.choice(self.arms, 1)[0]
            else:
                arm_picked = self.arms[np.argmax(imp_values)]
            self.action_sequence.append(arm_picked)
            self.logger.info('Pulling arm: %s' % arm_picked)
            results = self.optimizer[arm_picked].iterate()
            self.collect_iter_stats(arm_picked, results)
            self.pull_cnt += 1

    def play_once(self):
        self.optimize()
        _perf = Evaluator(self.inc['hpo'], data_node=self.inc['fe'], seed=self.seed)(self.inc['hpo'])
        self.final_rewards.append(_perf)
        return _perf
