import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from automlToolkit.components.fe_optimizers import Optimizer
from automlToolkit.components.feature_engineering.transformations import _preprocessor1, _preprocessor2, _balancer, \
    _generator, _selector, _rescaler
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.evaluators.base_evaluator import _BaseEvaluator
from automlToolkit.components.utils.constants import SUCCESS, ERROR, TIMEOUT, CLS_TASKS
from automlToolkit.components.feature_engineering import TRANS_CANDIDATES
from litebo.facade.bo_facade import BayesianOptimization as BO


class BayesianOptimizationOptimizer(Optimizer):
    def __init__(self, task_type, input_data: DataNode, evaluator: _BaseEvaluator,
                 model_id: str, time_limit_per_trans: int,
                 mem_limit_per_trans: int,
                 seed: int, n_jobs=1,
                 number_of_unit_resource=1,
                 time_budget=600):
        super().__init__(str(__class__.__name__), task_type, input_data, seed)
        self.number_of_unit_resource = number_of_unit_resource
        self.iter_num_per_unit_resource = 10
        self.time_limit_per_trans = time_limit_per_trans
        self.mem_limit_per_trans = mem_limit_per_trans
        self.time_budget = time_budget
        self.evaluator = evaluator
        self.model_id = model_id

        self.incumbent_config = None
        self.incumbent_score = -np.inf
        self.baseline_score = -np.inf
        self.start_time = time.time()
        self.hp_config = None
        self.seed = seed

        self.node_dict = dict()

        self.early_stopped_flag = False
        self.is_finished = False
        self.iteration_id = 0

        self.evaluator.parse_needed = True
        # Prepare the hyperparameter space.
        self.hyperparameter_space = self._get_task_hyperparameter_space()
        self.optimizer = BO(objective_function=self.evaluate_function,
                            config_space=self.hyperparameter_space,
                            max_runs=int(1e10),
                            task_id=self.model_id,
                            rng=np.random.RandomState(self.seed))

    def evaluate_function(self, config):
        """
            The config is the configuration that specifies the FE pipeline.
        :param config:
        :return: the evaluation score.
        """
        input_node = self.root_node
        output_node = self._parse(input_node, config)
        return 1 - self.evaluator(self.hp_config, data_node=output_node, name='fe')

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
            result = self._iterate()
        return result

    def _iterate(self):
        _start_time = time.time()
        for _ in range(self.iter_num_per_unit_resource):
            _, status, _, info = self.optimizer.iterate()
            if status == 1:
                print(info)

        runhistory = self.optimizer.get_history()
        self.incumbent_config, iter_incumbent_score = runhistory.get_incumbents()[0]
        iter_incumbent_score = 1 - iter_incumbent_score
        iteration_cost = time.time() - _start_time
        if iter_incumbent_score > self.incumbent_score:
            self.incumbent_score = iter_incumbent_score
            self.incumbent = self._parse(self.root_node, self.incumbent_config)

        return self.incumbent_score, iteration_cost, self.incumbent

    def _parse(self, data_node: DataNode, config, record=False, skip_balance=False):
        """
            Transform the data node based on the pipeline specified by configuration.
        :param data_node:
        :param config:
        :param record:
        :return: the resulting data node.
        """
        # Remove the indicator in config_dict.
        config_dict = config.get_dictionary().copy()
        pre1_id = config_dict['preprocessor1']
        config_dict.pop('preprocessor1')
        pre2_id = config_dict['preprocessor2']
        config_dict.pop('preprocessor2')
        if skip_balance:
            bal_id = 'empty'
        else:
            if 'balancer' in config_dict:
                bal_id = config_dict['balancer']
                config_dict.pop('balancer')
            else:
                bal_id = 'empty'
        gen_id = config_dict['generator']
        config_dict.pop('generator')
        res_id = config_dict['rescaler']
        config_dict.pop('rescaler')
        sel_id = config_dict['selector']
        config_dict.pop('selector')

        def tran_operate(id, tran_set, config, node):
            if id != "empty":
                _config = {}
                for key in config:
                    if id in key:
                        config_name = key.split(':')[1]
                        _config[config_name] = config[key]
                tran = tran_set[id](**_config)
                output_node = tran.operate(node)
                return output_node, tran
            return node, None

        _node = data_node.copy_()
        # Preprocessor1
        _node, pre1_tran = tran_operate(pre1_id, _preprocessor1, config_dict, _node)

        # Preprocessor2
        _node, pre2_tran = tran_operate(pre2_id, _preprocessor2, config_dict, _node)

        # Balancer
        _node, bal_tran = tran_operate(bal_id, _balancer, config_dict, _node)

        # Rescaler
        _node, res_tran = tran_operate(res_id, _rescaler, config_dict, _node)

        # Generator
        _node, gen_tran = tran_operate(gen_id, _generator, config_dict, _node)

        # Selector
        _node, sel_tran = tran_operate(sel_id, _selector, config_dict, _node)

        if record:
            return _node, [pre1_tran, pre2_tran, bal_tran, res_tran, gen_tran, sel_tran]
        return _node

    def _get_task_hyperparameter_space(self):
        """
            Fetch the underlying hyperparameter space for feature engineering.
            Pipeline Space:
                1. preprocess: continous_discretizer
                2. preprocess: discrete_categorizer,
                3. balancer: weight_balancer,
                             data_balancer.
                4. scaler: normalizer, scaler, quantile.
                5. generator: all generators.
                6. selector: all selectors.
        :return: hyper space.
        """
        if self.task_type in CLS_TASKS:
            self.trans_types = TRANS_CANDIDATES['classification'].copy()
        else:
            self.trans_types = TRANS_CANDIDATES['regression'].copy()

        # Avoid transformations, which would take too long
        # Combinations of non-linear models with feature learning.
        # feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
        if self.task_type in CLS_TASKS:
            classifier_set = ["adaboost", "decision_tree", "extra_trees",
                              "gradient_boosting", "k_nearest_neighbors",
                              "libsvm_svc", "random_forest", "gaussian_nb",
                              "decision_tree", "lightgbm"]

            if self.model_id in classifier_set:
                for tran_id in [12, 13, 15]:
                    if tran_id in self.trans_types:
                        self.trans_types.remove(tran_id)

        generator_dict = self._get_configuration_space(_generator, self.trans_types)
        rescaler_dict = self._get_configuration_space(_rescaler, self.trans_types)
        selector_dict = self._get_configuration_space(_selector, self.trans_types)
        preprocessor1_dict = self._get_configuration_space(_preprocessor1)
        preprocessor2_dict = self._get_configuration_space(_preprocessor2)
        if self.task_type in CLS_TASKS:
            balancer_dict = self._get_configuration_space(_balancer)
        else:
            balancer_dict = None
        cs = self._build_hierachical_configspace(preprocessor1_dict, preprocessor2_dict,
                                                 generator_dict, rescaler_dict,
                                                 selector_dict, balancer_dict)
        return cs

    def _get_configuration_space(self, builtin_transformers, trans_type=None):
        config_dict = dict()
        for tran_key in builtin_transformers:
            tran = builtin_transformers[tran_key]
            tran_id = tran().type
            if trans_type is None or tran_id in trans_type:
                try:
                    sub_configuration_space = builtin_transformers[tran_key].get_hyperparameter_search_space()
                    config_dict[tran_key] = sub_configuration_space
                except:
                    config_dict[tran_key] = ConfigurationSpace()
        return config_dict

    def _build_hierachical_configspace(self, pre_config1, pre_config2, gen_config, res_config, sel_config,
                                       bal_config=None):
        cs = ConfigurationSpace()
        self._add_hierachical_configspace(cs, pre_config1, "preprocessor1")
        self._add_hierachical_configspace(cs, pre_config2, "preprocessor2")
        if bal_config is not None:
            self._add_hierachical_configspace(cs, bal_config, "balancer")
        self._add_hierachical_configspace(cs, gen_config, "generator")
        self._add_hierachical_configspace(cs, res_config, "rescaler")
        self._add_hierachical_configspace(cs, sel_config, "selector")
        return cs

    def _add_hierachical_configspace(self, cs, config, parent_name):
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

    def fetch_nodes(self, n=10):
        runhistory = self.optimizer.get_history()
        hist_dict = runhistory.data
        min_list = sorted(hist_dict.items(), key=lambda item: item[1])
        min_n = list(min_list[:n])
        # Get default configuration.
        default_config = self.hyperparameter_space.get_default_configuration()
        min_n.append((default_config, runhistory.data[default_config]))
        min_n.append((self.incumbent_config, runhistory.data[self.incumbent_config]))

        node_list = []
        for i, config in enumerate(min_n):
            try:
                node, tran_list = self._parse(self.root_node, config[0], record=True)
                if i == 0:
                    self.incumbent = node  # Update incumbent node
                node_list.append(node)
                self.node_dict[len(self.node_dict)] = [node, tran_list]
            except:
                print("Re-parse failed on config %s" % str(config[0]))
        return node_list

    def apply(self, data_node: DataNode, ref_node: DataNode, phase='test'):
        input_node = data_node.copy_()
        if_fit = False
        for _, value in self.node_dict.items():
            if ref_node == value[0]:
                if_fit = True
                for i, tran in enumerate(value[1]):
                    if phase == 'test' and i == 2:  # Disable balancer
                        continue
                    if tran is not None:
                        input_node = tran.operate(input_node)
                break
        if not if_fit:
            raise ValueError("Ref node not in history!")
        return input_node
