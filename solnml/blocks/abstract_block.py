import os
import time
import pickle as pkl
import numpy as np
from ConfigSpace import ConfigurationSpace
from solnml.components.metrics.metric import get_metric
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.feature_engineering.parse import construct_node, parse_config
from solnml.components.ensemble.ensemble_bulider import EnsembleBuilder
from solnml.components.evaluators.base_evaluator import load_combined_transformer_estimator, fetch_predict_estimator, \
    CombinedTopKModelSaver
from solnml.components.utils.constants import CLS_TASKS
from solnml.utils.functions import is_imbalanced_dataset
from solnml.utils.logging_utils import get_logger


class AbstractBlock(object):
    def __init__(self, node_list, node_index,
                 task_type, timestamp,
                 fe_config_space: ConfigurationSpace,
                 cash_config_space: ConfigurationSpace,
                 data: DataNode,
                 fixed_config=None,
                 trial_num=0,
                 time_limit=None,
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
        # Tree setting
        self.node_list = node_list
        self.node_index = node_index

        # Set up backend.
        self.dataset_name = dataset_name
        self.trial_num = trial_num
        self.time_limit = time_limit
        self.per_run_time_limit = per_run_time_limit
        self.start_time = time.time()
        self.logger = get_logger('Soln-ml: %s' % dataset_name)

        # Basic settings.
        self.eval_type = eval_type
        self.resampling_params = resampling_params
        self.task_type = task_type
        self.timestamp = timestamp
        self.fe_config_space = fe_config_space
        self.cash_config_space = cash_config_space
        self.fixed_config = fixed_config
        self.original_data = data.copy_()
        self.metric = get_metric(metric)
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.seed = seed
        self.output_dir = output_dir

        self.early_stop_flag = False
        self.timeout_flag = False
        self.incumbent_perf = -float("INF")
        self.incumbent = None
        self.eval_dict = dict()

        if self.task_type in CLS_TASKS:
            self.if_imbal = is_imbalanced_dataset(self.original_data)
        else:
            self.if_imbal = False

        self.es = None

    def refit(self):
        if self.ensemble_method is not None:
            self.logger.info('Start to refit all the well-performed models!')
            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)
            for algo_id in stats.keys():
                model_to_eval = stats[algo_id]
                for idx, (config, perf, path) in enumerate(model_to_eval):
                    data_node, op_list = parse_config(self.original_data.copy_(), config, record=True,
                                                      if_imbal=self.if_imbal)
                    algo_id = config['algorithm']
                    estimator = fetch_predict_estimator(self.task_type, algo_id, config,
                                                        data_node.data[0], data_node.data[1],
                                                        weight_balance=data_node.enable_balance,
                                                        data_balance=data_node.data_balance)
                    with open(path, 'wb')as f:
                        pkl.dump([op_list, estimator], f)

            self.fit_ensemble()
        else:
            self.logger.info('Start to refit the best model!')
            model_path = os.path.join(self.output_dir, '%s_%s.pkl' % (
                self.timestamp, CombinedTopKModelSaver.get_configuration_id(self.incumbent)))
            config = self.incumbent.copy()
            data_node, op_list = parse_config(self.original_data.copy_(), config, record=True,
                                              if_imbal=self.if_imbal)
            algo_id = config['algorithm']
            estimator = fetch_predict_estimator(self.task_type, algo_id, config,
                                                data_node.data[0], data_node.data[1],
                                                weight_balance=data_node.enable_balance,
                                                data_balance=data_node.data_balance)
            with open(model_path, 'wb')as f:
                pkl.dump([op_list, estimator], f)

    def fit_ensemble(self):
        if self.ensemble_method is not None:
            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      data_node=self.original_data,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=self.original_data)

    def predict(self, test_data: DataNode):
        if self.task_type in CLS_TASKS:
            pred = self._predict(test_data)
            return np.argmax(pred, axis=-1)
        else:
            return self._predict(test_data)

    def _predict(self, test_data: DataNode):
        if self.ensemble_method is not None:
            if self.es is None and self.eval_type == 'cv':
                raise AttributeError("Please call refit() for cross-validation!")
            elif self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data)
        else:
            try:
                best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.incumbent,
                                                                              self.timestamp)
            except Exception as e:
                if self.eval_type == 'cv':
                    raise AttributeError("Please call refit() for cross-validation!")
                else:
                    raise e
            test_data_node = test_data.copy_()
            test_data_node = construct_node(test_data_node, best_op_list)

            if self.task_type in CLS_TASKS:
                return estimator.predict_proba(test_data_node.data[0])
            else:
                return estimator.predict(test_data_node.data[0])

    def predict_proba(self, test_data: DataNode):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            raise ValueError('metric_func is not defined!')
        y_pred = self.predict(test_data)
        return metric_func(test_data.data[1], y_pred)

    def iterate(self, trial_num=10):
        raise NotImplementedError()
