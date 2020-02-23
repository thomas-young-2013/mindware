import typing
import numpy as np
from timeout_decorator import timeout
from automlToolkit.components.evaluators.evaluator import Evaluator
from automlToolkit.utils.logging_utils import get_logger
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.hpo_optimizer.psmac_optimizer import PSMACOptimizer
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.utils.functions import get_increasing_sequence


class SecondLayerBandit(object):
    def __init__(self, classifier_id: str, data: DataNode,
                 share_fe=False, output_dir='logs',
                 per_run_time_limit=120,
                 per_run_mem_limit=5120,
                 eval_type='cv', dataset_id='default',
                 mth='rb', sw_size=3, strategy='avg',
                 n_jobs=1, seed=1):
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.classifier_id = classifier_id
        self.evaluation_type = eval_type
        self.original_data = data.copy_()
        self.share_fe = share_fe
        self.output_dir = output_dir
        self.mth = mth
        self.strategy = strategy
        self.seed = seed
        self.sliding_window_size = sw_size
        self.logger = get_logger('%s:%s-%d=>%s' % (__class__.__name__, dataset_id, seed, classifier_id))
        np.random.seed(self.seed)

        # Bandit settings.
        self.arms = ['fe', 'hpo']
        self.rewards = dict()
        self.optimizer = dict()
        self.evaluation_cost = dict()
        self.inc = dict()
        self.local_inc = dict()
        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()
        self.incumbent_perf = -1.
        self.incumbent_source = None
        self.update_flag = dict()
        self.imp_rewards = dict()
        for arm in self.arms:
            self.update_flag[arm] = True
            self.imp_rewards[arm] = list()

        from autosklearn.pipeline.components.classification import _classifiers
        clf_class = _classifiers[classifier_id]
        cs = clf_class.get_hyperparameter_search_space()
        model = UnParametrizedHyperparameter("estimator", classifier_id)
        cs.add_hyperparameter(model)
        self.config_space = cs
        self.default_config = cs.get_default_configuration()
        self.config_space.seed(self.seed)

        # Build the Feature Engineering component.
        fe_evaluator = Evaluator(self.default_config,
                                 name='fe', resampling_strategy=self.evaluation_type,
                                 seed=self.seed)
        self.optimizer['fe'] = EvaluationBasedOptimizer(
                'classification',
                self.original_data, fe_evaluator,
                classifier_id, per_run_time_limit, per_run_mem_limit, self.seed,
                shared_mode=self.share_fe, n_jobs=n_jobs)
        self.inc['fe'], self.local_inc['fe'] = self.original_data, self.original_data

        # Build the HPO component.
        trials_per_iter = len(self.optimizer['fe'].trans_types)
        hpo_evaluator = Evaluator(self.default_config,
                                  data_node=self.original_data, name='hpo',
                                  resampling_strategy=self.evaluation_type,
                                  seed=self.seed)
        if n_jobs == 1:
            self.optimizer['hpo'] = SMACOptimizer(
                hpo_evaluator, cs, output_dir=output_dir, per_run_time_limit=per_run_time_limit,
                trials_per_iter=trials_per_iter // 2, seed=self.seed)
        else:
            self.optimizer['hpo'] = PSMACOptimizer(
                hpo_evaluator, cs, output_dir=output_dir, per_run_time_limit=per_run_time_limit,
                trials_per_iter=trials_per_iter // 2, seed=self.seed,
                n_jobs=n_jobs
            )
        self.inc['hpo'], self.local_inc['hpo'] = self.default_config, self.default_config

    def collect_iter_stats(self, _arm, results):
        if _arm == 'fe' and len(self.final_rewards) == 0:
            self.incumbent_perf = self.optimizer['fe'].baseline_score
            self.final_rewards.append(self.incumbent_perf)

        self.logger.info('After %d-th pulling, results: %s' % (self.pull_cnt, results))
        score, iter_cost, config = results

        if score is None:
            score = 0.0

        self.rewards[_arm].append(score)
        self.evaluation_cost[_arm].append(iter_cost)
        self.local_inc[_arm] = config
        if score > self.incumbent_perf:
            self.inc[_arm] = config
            if _arm == 'fe':
                self.inc['hpo'] = self.default_config
            else:
                self.inc['fe'] = self.original_data
            self.incumbent_perf = score

        for arm_id in self.arms:
            self.update_flag[arm_id] = False

        if len(self.final_rewards) > 0 and self.final_rewards[-1] < self.incumbent_perf:
            arm_id = 'fe' if _arm == 'hpo' else 'hpo'
            self.update_flag[arm_id] = True

        if _arm == 'fe':
            _num_iter = self.optimizer['fe'].evaluation_num_last_iteration // 2
            self.optimizer['hpo'].trials_per_iter = max(_num_iter, 1)

        if self.mth == 'alter' and self.strategy == 'rb':
            if len(self.final_rewards) > 0:
                imp = self.incumbent_perf - self.final_rewards[-1]
            else:
                imp = self.incumbent_perf - self.optimizer['fe'].baseline_score
            assert imp >= 0.
            self.imp_rewards[_arm].append(imp)

    def prepare_optimizer(self, _arm):
        if _arm == 'fe':
            if self.update_flag[_arm] is True:
                # Build the Feature Engineering component.
                fe_evaluator = Evaluator(self.inc['hpo'], name='fe', resampling_strategy=self.evaluation_type,
                                         seed=self.seed)
                self.optimizer[_arm] = EvaluationBasedOptimizer(
                    'classification',
                    self.inc['fe'], fe_evaluator,
                    self.classifier_id, self.per_run_time_limit, self.per_run_mem_limit, self.seed,
                    shared_mode=self.share_fe
                )
            else:
                self.logger.info('No improvement on HPO, so use the old FE optimizer!')
        else:
            if self.update_flag[_arm] is True:
                trials_per_iter = self.optimizer['fe'].evaluation_num_last_iteration
                hpo_evaluator = Evaluator(self.config_space.get_default_configuration(),
                                          data_node=self.inc['fe'],
                                          name='hpo',
                                          resampling_strategy=self.evaluation_type,
                                          seed=self.seed)
                self.optimizer[_arm] = SMACOptimizer(
                    hpo_evaluator, self.config_space, output_dir=self.output_dir,
                    per_run_time_limit=self.per_run_time_limit,
                    trials_per_iter=trials_per_iter // 2, seed=self.seed
                )
            else:
                self.logger.info('No improvement on FE, so use the old HPO optimizer!')

    def optimize(self):
        # First pull each arm #sliding_window_size times.
        if self.pull_cnt < len(self.arms) * self.sliding_window_size:
            _arm = self.arms[self.pull_cnt % 2]
            self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))
            results = self.optimizer[_arm].iterate()
            self.collect_iter_stats(_arm, results)
            self.pull_cnt += 1
            self.action_sequence.append(_arm)
        else:
            imp_values = list()
            for _arm in self.arms:
                increasing_rewards = get_increasing_sequence(self.rewards[_arm])

                impv = list()
                for idx in range(1, len(increasing_rewards)):
                    impv.append(increasing_rewards[idx] - increasing_rewards[idx - 1])
                imp_values.append(np.mean(impv[-self.sliding_window_size:]))

            self.logger.debug('Imp values: %s' % imp_values)
            if imp_values[0] == imp_values[1]:
                # Break ties randomly.
                # arm_picked = np.random.choice(self.arms, 1)[0]
                arm_picked = 'fe' if self.action_sequence[-1] == 'hpo' else 'hpo'
            else:
                arm_picked = self.arms[np.argmax(imp_values)]
            self.action_sequence.append(arm_picked)

            self.logger.info('Pulling arm: %s for %s at %d-th round' % (arm_picked, self.classifier_id, self.pull_cnt))
            results = self.optimizer[arm_picked].iterate()
            self.collect_iter_stats(arm_picked, results)
            self.pull_cnt += 1

    def choose_arm(self):
        # First pull each arm #sliding_window_size times.
        if self.pull_cnt < len(self.arms) * self.sliding_window_size:
            arm_picked = self.arms[self.pull_cnt % 2]
            # if _arm == 'fe' and len(self.final_rewards) == 0:
            #     self.final_rewards.append(self.optimizer['fe'].baseline_score)
        else:
            imp_values = list()
            for _arm in self.arms:
                if self.mth == 'rb':
                    increasing_rewards = get_increasing_sequence(self.rewards[_arm])
                    impv = list()
                    for idx in range(1, len(increasing_rewards)):
                        impv.append(increasing_rewards[idx] - increasing_rewards[idx - 1])
                    imp_values.append(np.mean(impv[-self.sliding_window_size:]))
                elif self.mth == 'alter' and self.strategy == 'rb':
                    imp_values.append(np.mean(self.imp_rewards[_arm][-self.sliding_window_size:]))
                else:
                    raise ValueError('Invalid parameters!')

            self.logger.info('Imp values: %s' % imp_values)
            if imp_values[0] == imp_values[1]:
                # Break ties randomly.
                self.logger.info('Same Imp values: %s' % imp_values)
                arm_picked = 'fe' if self.action_sequence[-1] == 'hpo' else 'hpo'
                # arm_picked = np.random.choice(self.arms, 1)[0]
            else:
                self.logger.info('Different Imp values: %s' % imp_values)
                arm_picked = self.arms[np.argmax(imp_values)]
        return arm_picked

    def optimize_alternatedly(self):
        # First choose one arm.
        if self.strategy == 'avg':
            _arm = self.arms[self.pull_cnt % 2]
        else:
            _arm = self.choose_arm()
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))

        self.prepare_optimizer(_arm)

        # Execute one iteration.
        results = self.optimizer[_arm].iterate()

        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def play_once(self):
        if self.mth == 'rb':
            self.optimize()
            _perf = None

            @timeout(300)
            def evaluate():
                perf = Evaluator(
                    self.local_inc['hpo'], data_node=self.local_inc['fe'],
                    name='fe', resampling_strategy=self.evaluation_type,
                    seed=self.seed)(self.local_inc['hpo'])
                return perf

            try:
                _perf = evaluate()

            except Exception as e:
                self.logger.error(str(e))
            if _perf is None:
                _perf = 0.0
            if _perf > self.incumbent_perf:
                self.inc['hpo'] = self.local_inc['hpo']
                self.inc['fe'] = self.local_inc['fe']
                self.incumbent_perf = _perf
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
