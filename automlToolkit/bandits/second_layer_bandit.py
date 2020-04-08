import typing
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from automlToolkit.components.evaluators.evaluator import Evaluator
from automlToolkit.utils.logging_utils import get_logger
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from automlToolkit.components.hpo_optimizer.smac_optimizer import SMACOptimizer
from automlToolkit.components.hpo_optimizer.psmac_optimizer import PSMACOptimizer
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.components.evaluators.evaluator import fetch_predict_estimator
from automlToolkit.utils.decorators import time_limit, TimeoutException
from automlToolkit.utils.functions import get_increasing_sequence


class SecondLayerBandit(object):
    def __init__(self, classifier_id: str, data: DataNode,
                 share_fe=False, output_dir='logs',
                 per_run_time_limit=120,
                 per_run_mem_limit=5120,
                 dataset_id='default',
                 eval_type='holdout',
                 mth='rb', sw_size=3,
                 n_jobs=1, seed=1,
                 enable_intersection=True,
                 number_of_unit_resource=2):
        self.number_of_unit_resource = number_of_unit_resource
        # One unit of resource, that's, the number of trials per iteration.
        self.one_unit_of_resource = 5
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.classifier_id = classifier_id
        self.evaluation_type = eval_type
        self.original_data = data.copy_()
        self.share_fe = share_fe
        self.output_dir = output_dir
        self.mth = mth
        self.seed = seed
        self.sliding_window_size = sw_size
        self.logger = get_logger('%s:%s-%d=>%s' % (
            __class__.__name__, dataset_id, seed, classifier_id))
        np.random.seed(self.seed)

        # Bandit settings.
        self.arms = ['fe', 'hpo']
        self.rewards = dict()
        self.optimizer = dict()
        self.evaluation_cost = dict()
        self.update_flag = dict()
        # Global incumbent.
        self.inc = dict()
        self.local_inc = dict()
        for arm in self.arms:
            self.rewards[arm] = list()
            self.update_flag[arm] = False
            self.evaluation_cost[arm] = list()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()
        self.incumbent_perf = -1.
        self.early_stopped_flag = False
        self.enable_intersection = enable_intersection

        # Set hyperparameter space.
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
        # trials_per_iter = max(len(self.optimizer['fe'].trans_types), 20)
        trials_per_iter = self.one_unit_of_resource * self.number_of_unit_resource
        hpo_evaluator = Evaluator(self.default_config,
                                  data_node=self.original_data, name='hpo',
                                  resampling_strategy=self.evaluation_type,
                                  seed=self.seed)
        if n_jobs == 1:
            self.optimizer['hpo'] = SMACOptimizer(
                hpo_evaluator, cs, output_dir=output_dir, per_run_time_limit=per_run_time_limit,
                trials_per_iter=trials_per_iter, seed=self.seed)
        else:
            self.optimizer['hpo'] = PSMACOptimizer(
                hpo_evaluator, cs, output_dir=output_dir, per_run_time_limit=per_run_time_limit,
                trials_per_iter=trials_per_iter, seed=self.seed,
                n_jobs=n_jobs
            )
        self.inc['hpo'], self.local_inc['hpo'] = self.default_config, self.default_config

    def collect_iter_stats(self, _arm, results):
        for arm_id in self.arms:
            self.update_flag[arm_id] = False

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

        # Update global incumbent from FE and HPO.
        if score > self.incumbent_perf and np.isfinite(score):
            self.inc[_arm] = config
            if _arm == 'fe':
                self.inc['hpo'] = self.default_config
            else:
                if self.mth not in ['alter_hpo', 'rb_hpo']:
                    self.inc['fe'] = self.original_data
                else:
                    self.inc['fe'] = self.local_inc['fe']

            self.incumbent_perf = score

            arm_id = 'fe' if _arm == 'hpo' else 'hpo'
            self.update_flag[arm_id] = True

            if self.mth in ['rb_hpo', 'alter_hpo'] and _arm == 'fe':
                self.prepare_optimizer(arm_id)
            if self.mth == 'alter_p':
                self.prepare_optimizer(arm_id)

    def optimize_rb(self):
        # First pull each arm #sliding_window_size times.
        if self.pull_cnt < len(self.arms) * self.sliding_window_size:
            arm_picked = self.arms[self.pull_cnt % 2]
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

        # Early stopping scenario.
        if self.optimizer[arm_picked].early_stopped_flag is True:
            arm_picked = 'hpo' if arm_picked == 'fe' else 'fe'
            if self.optimizer[arm_picked].early_stopped_flag is True:
                self.early_stopped_flag = True
                return

        self.action_sequence.append(arm_picked)
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (arm_picked, self.classifier_id, self.pull_cnt))
        results = self.optimizer[arm_picked].iterate()
        self.collect_iter_stats(arm_picked, results)
        self.pull_cnt += 1

    def optimize_alternatedly(self):
        # First choose one arm.
        _arm = self.arms[self.pull_cnt % 2]
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))

        # Execute one iteration.
        results = self.optimizer[_arm].iterate()

        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def optimize_one_component(self, mth):
        _arm = 'hpo' if mth == 'hpo_only' else 'fe'
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (_arm, self.classifier_id, self.pull_cnt))

        # Execute one iteration.
        results = self.optimizer[_arm].iterate()
        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def evaluate_joint_solution(self):
        # Update join incumbent from FE and HPO.
        _perf = None
        try:
            with time_limit(600):
                _perf = Evaluator(
                    self.local_inc['hpo'], data_node=self.local_inc['fe'],
                    name='fe', resampling_strategy=self.evaluation_type,
                    seed=self.seed)(self.local_inc['hpo'])
        except Exception as e:
            self.logger.error(str(e))
        # Update INC.
        if _perf is not None and _perf > self.incumbent_perf and np.isfinite(_perf):
            self.inc['hpo'] = self.local_inc['hpo']
            self.inc['fe'] = self.local_inc['fe']
            self.incumbent_perf = _perf

    def play_once(self):
        if self.early_stopped_flag:
            return self.incumbent_perf

        if self.mth in ['rb', 'rb_hpo']:
            self.optimize_rb()
            self.evaluate_joint_solution()
        elif self.mth in ['alter', 'alter_p', 'alter_hpo']:
            self.optimize_alternatedly()
            self.evaluate_joint_solution()
        elif self.mth in ['fe_only', 'hpo_only']:
            self.optimize_one_component(self.mth)
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

    def predict_proba(self, X_test, is_weighted=False):
        """
            weight source: ...
            model 1: local_inc['fe'], default_hpo
            model 2: default_fe, local_inc['hpo']
            model 3: local_inc['fe'], local_inc['hpo']
        :param X_test:
        :param is_weighted:
        :return:
        """
        X_train_ori, y_train_ori = self.original_data.data
        X_train_inc, y_train_inc = self.local_inc['fe'].data

        model1_clf = fetch_predict_estimator(self.default_config, X_train_inc, y_train_inc)
        model2_clf = fetch_predict_estimator(self.local_inc['hpo'], X_train_ori, y_train_ori)
        model3_clf = fetch_predict_estimator(self.local_inc['hpo'], X_train_inc, y_train_inc)
        model4_clf = fetch_predict_estimator(self.default_config, X_train_ori, y_train_ori)

        if is_weighted:
            # Based on performance on the validation set
            # TODO: Save the results so that the models will not be trained again
            from sklearn.model_selection import train_test_split
            from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection
            from autosklearn.metrics import balanced_accuracy
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=1)
            X, y = X_train_ori.copy(), y_train_ori.copy()
            _X, _y = X_train_inc.copy(), y_train_inc.copy()
            for train_index, test_index in sss.split(X, y):
                X_train, X_val, y_train, y_val = X[train_index], X[test_index], y[train_index], y[test_index]
                _X_train, _X_val, _y_train, _y_val = _X[train_index], _X[test_index], _y[train_index], _y[test_index]

            assert (y_val == _y_val).all()
            model1_clf_temp = fetch_predict_estimator(self.default_config, _X_train, _y_train)
            model2_clf_temp = fetch_predict_estimator(self.local_inc['hpo'], X_train, y_train)
            model3_clf_temp = fetch_predict_estimator(self.local_inc['hpo'], _X_train, _y_train)
            model4_clf_temp = fetch_predict_estimator(self.default_config, X_train, y_train)
            pred1 = model1_clf_temp.predict_proba(_X_val)
            pred2 = model2_clf_temp.predict_proba(X_val)
            pred3 = model3_clf_temp.predict_proba(_X_val)
            pred4 = model4_clf_temp.predict_proba(X_val)

            # Ensemble size is a hyperparameter
            es = EnsembleSelection(ensemble_size=20, task_type=1, metric=balanced_accuracy,
                                   random_state=np.random.RandomState(self.seed))
            es.fit([pred1, pred2, pred3, pred4], y_val, None)
            weights = es.weights_
            print("weights " + str(weights))

        # Make sure that the estimator has "predict_proba"
        _test_node = DataNode(data=[X_test, None], feature_type=self.original_data.feature_types.copy())
        _X_test = self.optimizer['fe'].apply(_test_node, self.local_inc['fe']).data[0]
        pred1 = model1_clf.predict_proba(_X_test)
        pred2 = model2_clf.predict_proba(X_test)
        pred3 = model3_clf.predict_proba(_X_test)
        pred4 = model4_clf.predict_proba(X_test)

        if is_weighted:
            final_pred = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3 + weights[3] * pred4
        else:
            final_pred = (pred1 + pred2 + pred3 + pred4) / 4

        return final_pred

    def predict(self, X_test, is_weighted=False):
        proba_pred = self.predict_proba(X_test, is_weighted)
        return np.argmax(proba_pred, axis=-1)

    def prepare_optimizer(self, _arm):
        if _arm == 'fe':
            # Build the Feature Engineering component.
            fe_evaluator = Evaluator(self.inc['hpo'], name='fe', resampling_strategy=self.evaluation_type,
                                     seed=self.seed)
            self.optimizer[_arm] = EvaluationBasedOptimizer(
                'classification', self.inc['fe'], fe_evaluator,
                self.classifier_id, self.per_run_time_limit, self.per_run_mem_limit,
                self.seed, shared_mode=self.share_fe
            )
        else:
            # trials_per_iter = self.optimizer['fe'].evaluation_num_last_iteration // 2
            # trials_per_iter = max(20, trials_per_iter)
            trials_per_iter = self.one_unit_of_resource * self.number_of_unit_resource
            hpo_evaluator = Evaluator(self.config_space.get_default_configuration(),
                                      resampling_strategy=self.evaluation_type,
                                      data_node=self.inc['fe'],
                                      seed=self.seed,
                                      name='hpo')
            self.optimizer[_arm] = SMACOptimizer(
                hpo_evaluator, self.config_space, output_dir=self.output_dir,
                per_run_time_limit=self.per_run_time_limit,
                trials_per_iter=trials_per_iter, seed=self.seed
            )

        self.logger.info('='*30)
        self.logger.info('UPDATE OPTIMIZER: %s' % _arm)
        self.logger.info('='*30)
