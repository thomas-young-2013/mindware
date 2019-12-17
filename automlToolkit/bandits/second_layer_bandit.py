import typing
import numpy as np
from automlToolkit.components.evaluator import Evaluator
from automlToolkit.utils.logging_utils import get_logger
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.utils.functions import get_increasing_sequence


class SecondLayerBandit(object):
    def __init__(self, classifier_id: str, data: DataNode,
                 share_fe=False, output_dir='logs',
                 per_run_time_limit=150, seed=1,
                 mth='rb', sw_size=3):
        self.per_run_time_limit = per_run_time_limit
        self.classifier_id = classifier_id
        self.original_data = data
        self.share_fe = share_fe
        self.output_dir = output_dir
        self.mth = mth
        self.seed = seed
        self.sliding_window_size = sw_size
        self.logger = get_logger('%s=>%s' % (__class__.__name__, classifier_id))
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
        self.incumbent_perf = -1.

        from autosklearn.pipeline.components.classification import _classifiers
        clf_class = _classifiers[classifier_id]
        cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", classifier_id)
        cs.add_hyperparameter(model)
        self.config_space = cs

        # Build the Feature Engineering component.
        fe_evaluator = Evaluator(cs.get_default_configuration(), name='fe', seed=self.seed)
        self.optimizer['fe'] = EvaluationBasedOptimizer(
            self.original_data, fe_evaluator,
            classifier_id, per_run_time_limit, self.seed,
            shared_mode=self.share_fe
        )
        self.inc['fe'] = data

        # Build the HPO component.
        trials_per_iter = len(self.optimizer['fe'].trans_types)
        hpo_evaluator = Evaluator(cs.get_default_configuration(), data_node=data, name='hpo', seed=self.seed)
        self.optimizer['hpo'] = SMACOptimizer(
            hpo_evaluator, cs, output_dir=output_dir, per_run_time_limit=per_run_time_limit,
            trials_per_iter=trials_per_iter // 2, seed=self.seed)
        self.inc['hpo'] = cs.get_default_configuration()

    def collect_iter_stats(self, _arm, results):
        self.logger.info('After %d-th pulling, results: %s' % (self.pull_cnt, results))
        score, iter_cost, config = results
        if score is None:
            score = 0.0
        self.rewards[_arm].append(score)
        self.evaluation_cost[_arm].append(iter_cost)
        self.inc[_arm] = config
        self.incumbent_perf = max(score, self.incumbent_perf)

        if _arm == 'fe':
            _num_iter = self.optimizer['fe'].evaluation_num_last_iteration // 2
            self.optimizer['hpo'].trials_per_iter = max(_num_iter, 1)

    def optimize(self):
        # First pull each arm #sliding_window_size times.
        if self.pull_cnt < len(self.arms) * self.sliding_window_size:
            _arm = self.arms[self.pull_cnt % 2]
            self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))
            results = self.optimizer[_arm].iterate()
            self.collect_iter_stats(_arm, results)
            self.pull_cnt += 1
            self.action_sequence.append(_arm)
            if _arm == 'fe' and len(self.final_rewards) == 0:
                self.final_rewards.append(self.optimizer['fe'].baseline_score)
        else:
            imp_values = list()
            for _arm in self.arms:
                increasing_rewards = get_increasing_sequence(self.rewards[_arm])

                impv = list()
                for idx in range(1, len(increasing_rewards)):
                    impv.append(increasing_rewards[idx] - increasing_rewards[idx - 1])
                imp_values.append(np.mean(impv[-self.sliding_window_size:]))

            self.logger.debug('Imp values: %s' % imp_values)
            if np.sum(imp_values) == 0:
                # Break ties randomly.
                arm_picked = np.random.choice(self.arms, 1)[0]
            else:
                arm_picked = self.arms[np.argmax(imp_values)]
            self.action_sequence.append(arm_picked)

            self.logger.info('Pulling arm: %s for %s at %d-th round' % (arm_picked, self.classifier_id, self.pull_cnt))
            results = self.optimizer[arm_picked].iterate()
            self.collect_iter_stats(arm_picked, results)
            self.pull_cnt += 1

    def optimize_alternatedly(self):
        _arm = self.arms[self.pull_cnt % 2]
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))
        if _arm == 'fe':
            # Build the Feature Engineering component.
            fe_evaluator = Evaluator(self.inc['hpo'], name='fe', seed=self.seed)
            self.optimizer[_arm] = EvaluationBasedOptimizer(
                self.inc['fe'], fe_evaluator,
                self.classifier_id, self.per_run_time_limit, self.seed,
                shared_mode=self.share_fe
            )
        else:
            trials_per_iter = self.optimizer['fe'].evaluation_num_last_iteration
            hpo_evaluator = Evaluator(self.config_space.get_default_configuration(),
                                      data_node=self.inc['fe'],
                                      name='hpo',
                                      seed=self.seed)
            self.optimizer[_arm] = SMACOptimizer(
                hpo_evaluator, self.config_space, output_dir=self.output_dir,
                per_run_time_limit=self.per_run_time_limit,
                trials_per_iter=trials_per_iter // 2, seed=self.seed
            )

        results = self.optimizer[_arm].iterate()
        if _arm == 'fe' and len(self.final_rewards) == 0:
            self.final_rewards.append(self.optimizer['fe'].baseline_score)
        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def play_once(self):
        if self.mth == 'rb':
            self.optimize()
            _perf = Evaluator(self.inc['hpo'], data_node=self.inc['fe'],
                              name='fe', seed=self.seed)(self.inc['hpo'])
            if _perf is None:
                _perf = 0.0
            self.incumbent_perf = max(_perf, self.incumbent_perf)
        elif self.mth == 'alter':
            self.optimize_alternatedly()
        else:
            raise ValueError('Invalid method: %s' % self.mth)

        self.final_rewards.append(self.incumbent_perf)
        return self.incumbent_perf

    def fetch_local_incumbents(self):
        return self.optimizer['fe'].local_datanodes

    def sync_global_incumbents(self, global_nodes: typing.List[DataNode]):
        fe_optimizer = self.optimizer['fe']
        fe_optimizer.global_datanodes = []
        for node in global_nodes:
            _node = node.copy_()
            _node.depth = node.depth
            fe_optimizer.global_datanodes.append(_node)
        fe_optimizer.refresh_beam_set()
