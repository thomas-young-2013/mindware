import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from solnml.components.fe_optimizers import Optimizer
from solnml.components.fe_optimizers.parse import parse_config

from solnml.components.feature_engineering.transformations import _imb_balancer, _bal_balancer, _preprocessor, _rescaler
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.evaluators.base_evaluator import _BaseEvaluator
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.feature_engineering import TRANS_CANDIDATES
from litebo.facade.bo_facade import BayesianOptimization as BO


class AnotherBayesianOptimizationOptimizer(Optimizer):
    def __init__(self, task_type, input_data: DataNode, evaluator: _BaseEvaluator,
                 model_id: str, time_limit_per_trans: int,
                 mem_limit_per_trans: int,
                 seed: int, n_jobs=1,
                 number_of_unit_resource=1,
                 time_budget=600, algo='smac'):
        super().__init__(str(__class__.__name__), task_type, input_data, seed)
        self.number_of_unit_resource = number_of_unit_resource
        self.iter_num_per_unit_resource = 5
        self.time_limit_per_trans = time_limit_per_trans
        self.mem_limit_per_trans = mem_limit_per_trans
        self.time_budget = time_budget
        self.evaluator = evaluator
        self.model_id = model_id

        self.incumbent_score = -np.inf
        self.fetch_incumbent = None
        self.baseline_score = -np.inf
        self.start_time = time.time()
        self.hp_config = None
        self.seed = seed
        self.n_jobs = n_jobs

        self.node_dict = dict()

        self.early_stopped_flag = False
        self.is_finished = False
        self.iteration_id = 0

        self.evaluator.parse_needed = True
        # Prepare the hyperparameter space.
        self.hyperparameter_space = get_task_hyperparameter_space(task_type=task_type, estimator_id=model_id,
                                                                  optimizer=algo)
        if algo == 'smac':
            self.incumbent_config = self.hyperparameter_space.get_default_configuration()
        else:
            self.incumbent_config = None
        self.optimizer = BO(objective_function=self.evaluator,
                            config_space=self.hyperparameter_space,
                            max_runs=int(1e10),
                            task_id=self.model_id,
                            time_limit_per_trial=self.time_limit_per_trans,
                            rng=np.random.RandomState(self.seed))
        self.eval_dict = {}

    def optimize(self):
        """
            Interface enables user to use this FE optimizer only.
        :return:
        """
        self.is_finished = False
        while not self.is_finished:
            self.logger.debug('=' * 50)
            self.logger.debug('Start the ITERATION: %d' % self.iteration_id)
            self.logger.debug('=' * 50)
            self.iterate()
            if self.start_time + self.time_budget < time.time():
                self.is_finished = True
        return self.incumbent

    def iterate(self):
        result = None
        for _ in range(self.number_of_unit_resource):
            _start_time = time.time()
            for _ in range(self.iter_num_per_unit_resource):
                _, status, _, info = self.optimizer.iterate()
                if status == 1:
                    print(info)

            runhistory = self.optimizer.get_history()
            self.eval_dict = {(fe_config, self.evaluator.hpo_config): [-score, time.time()] for fe_config, score in
                              runhistory.data.items()}
            self.incumbent_config, iter_incumbent_score = runhistory.get_incumbents()[0]
            self.incumbent_score = -iter_incumbent_score  # Maximize
            iteration_cost = time.time() - _start_time

            # incumbent_score: the large the better
            result = (self.incumbent_score, iteration_cost, self.incumbent_config)
        return result

    def fetch_nodes(self, n=5):
        runhistory = self.optimizer.get_history()
        hist_dict = runhistory.data
        default_config = self.hyperparameter_space.get_default_configuration()

        if len(hist_dict) > 0:
            min_list = sorted(hist_dict.items(), key=lambda item: item[1])
        else:
            min_list = [(default_config, None)]

        if len(min_list) < 50:
            min_n = list(min_list[:n])
        else:
            amplification_rate = 3
            chosen_idxs = np.arange(n) * amplification_rate
            min_n = [min_list[idx] for idx in chosen_idxs]

        # Filter out.
        if len(hist_dict) > 0:
            default_perf = hist_dict[default_config]
            min_n = list(filter(lambda x: x[1] < default_perf, min_n))

        if default_config not in [x[0] for x in min_n]:
            # Get default configuration.
            min_n.append((default_config, runhistory.data[default_config]))

        if self.incumbent_config not in [x[0] for x in min_n]:
            min_n.append((self.incumbent_config, runhistory.data[self.incumbent_config]))

        node_list = []
        for i, config in enumerate(min_n):
            if config[0] in self.node_dict:
                node_list.append(self.node_dict[config[0]][0])
                continue

            try:
                node, tran_list = parse_config(self.root_node, config[0], record=True)
                node.config = config[0]
                if node.data[0].shape[1] == 0:
                    continue
                if self.fetch_incumbent is None:
                    self.fetch_incumbent = node  # Update incumbent node
                node_list.append(node)
                self.node_dict[config[0]] = [node, tran_list]
            except:
                print("Re-parse failed on config %s" % str(config[0]))
        return node_list

    def fetch_nodes_by_config(self, config_list):
        node_list = []
        self.logger.info("Re-parse %d configs" % len(config_list))
        for i, config in enumerate(config_list):
            try:
                if config is None:
                    config = self.hyperparameter_space.get_default_configuration()

                if config in self.node_dict:
                    node_list.append(self.node_dict[config][0])
                    continue

                node, tran_list = parse_config(self.root_node, config, record=True)
                node.config = config
                if node.data[0].shape[1] == 0:
                    continue
                if self.fetch_incumbent is None:
                    # Ensure the config_list is sorted by performance
                    self.fetch_incumbent = node  # Update incumbent node
                node_list.append(node)
                self.node_dict[config] = [node, tran_list]
                assert None not in self.node_dict
            except Exception as e:
                node_list.append(None)
                print("Re-parse failed on config %s" % str(config))
        return node_list

    def apply(self, data_node: DataNode, ref_node: DataNode, phase='test'):
        input_node = data_node.copy_()
        fe_config = ref_node.config
        if ref_node is None:
            return data_node
        elif fe_config in self.node_dict:
            fe_trans_list = self.node_dict[fe_config][1]
            for i, tran in enumerate(fe_trans_list):
                if phase == 'test' and i == 0:  # Disable balancer
                    continue
                if tran is not None:
                    input_node = tran.operate(input_node)
        else:
            self.logger.info("Ref node config:")
            self.logger.info(str(fe_config))
            self.logger.info("Node history in optimizer:")
            self.logger.info(str(self.node_dict))
            raise ValueError("Ref node not in history!")
        return input_node


def get_task_hyperparameter_space(task_type, estimator_id, optimizer='smac'):
    """
        Fetch the underlying hyperparameter space for feature engineering.
        Pipeline Space:
            1. balancer: weight_balancer,
                         data_balancer.
            2. scaler: normalizer, scaler, quantile.
            3. preprocessor
    :return: hyper space.
    """
    if task_type in CLS_TASKS:
        trans_types = TRANS_CANDIDATES['classification'].copy()
    else:
        trans_types = TRANS_CANDIDATES['regression'].copy()

    # Avoid transformations, which would take too long
    # Combinations of non-linear models with feature learning.
    # feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
    if task_type in CLS_TASKS:
        classifier_set = ["adaboost", "decision_tree", "extra_trees",
                          "gradient_boosting", "k_nearest_neighbors",
                          "libsvm_svc", "random_forest", "gaussian_nb",
                          "decision_tree", "lightgbm"]

        if estimator_id in classifier_set:
            for tran_id in [12, 13, 15]:
                if tran_id in trans_types:
                    trans_types.remove(tran_id)

        # if self.model_id == 'random_forest':
        #     if 18 in self.trans_types:
        #         self.trans_types.remove(18)
        #
        # if self.model_id == 'liblinear_svc':
        #     if 7 in self.trans_types:
        #         self.trans_types.remove(7)
        #
        # if self.model_id == 'extra_trees':
        #     if 35 in self.trans_types:
        #         self.trans_types.remove(35)

    preprocessor_dict = _get_configuration_space(_preprocessor, trans_types, optimizer=optimizer)
    rescaler_dict = _get_configuration_space(_rescaler, trans_types, optimizer=optimizer)
    if task_type in CLS_TASKS:
        _balancer = _bal_balancer
        balancer_dict = _get_configuration_space(_balancer, optimizer=optimizer)
    else:
        balancer_dict = None
    cs = _build_hierachical_configspace(preprocessor_dict, rescaler_dict, balancer_dict, optimizer=optimizer)
    return cs


def _get_configuration_space(builtin_transformers, trans_type=None, optimizer='smac'):
    config_dict = dict()
    for tran_key in builtin_transformers:
        tran = builtin_transformers[tran_key]
        tran_id = tran().type
        if trans_type is None or tran_id in trans_type:
            try:
                sub_configuration_space = builtin_transformers[tran_key].get_hyperparameter_search_space(
                    optimizer=optimizer)
                config_dict[tran_key] = sub_configuration_space
            except:
                if optimizer == 'smac':
                    config_dict[tran_key] = ConfigurationSpace()
                elif optimizer == 'tpe':
                    config_dict[tran_key] = {}
    return config_dict


def _add_hierachical_configspace(cs, config, parent_name):
    config_cand = list(config.keys())
    config_cand.append("empty")
    config_option = CategoricalHyperparameter(parent_name, config_cand,
                                              default_value=config_cand[-1])
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        if config_item == 'empty':
            sub_configuration_space = ConfigurationSpace()
        else:
            sub_configuration_space = config[config_item]
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)


def _build_hierachical_configspace(pre_config, res_config, bal_config=None, optimizer='smac'):
    if optimizer == 'smac':
        cs = ConfigurationSpace()
        if bal_config is not None:
            _add_hierachical_configspace(cs, bal_config, "balancer")
        _add_hierachical_configspace(cs, pre_config, "preprocessor")
        _add_hierachical_configspace(cs, res_config, "rescaler")
        return cs
    elif optimizer == 'tpe':
        from hyperopt import hp
        space = {}

        def dict2hi(dictionary):
            hi_list = list()
            for key in dictionary:
                hi_list.append((key, dictionary[key]))
            return hi_list

        space['generator'] = hp.choice('generator', dict2hi(pre_config))
        if bal_config is not None:
            space['balancer'] = hp.choice('balancer', dict2hi(bal_config))
        space['rescaler'] = hp.choice('rescaler', dict2hi(res_config))
        return space
