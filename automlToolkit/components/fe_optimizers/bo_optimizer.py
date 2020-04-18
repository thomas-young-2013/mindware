import time
import numpy as np
from automlToolkit.components.fe_optimizers import Optimizer
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
                 number_of_unit_resource=2):
        super().__init__(str(__class__.__name__), task_type, input_data, seed)
        self.number_of_unit_resource = number_of_unit_resource
        self.iter_num_per_unit_resource = 10
        self.time_limit_per_trans = time_limit_per_trans
        self.mem_limit_per_trans = mem_limit_per_trans
        self.evaluator = evaluator
        self.model_id = model_id
        self.incumbent_score = -np.inf
        self.start_time = time.time()
        self.hp_config = None
        self.seed = seed

        self.is_finished = False
        self.iteration_id = 0

        self.evaluator.parse_needed = True
        # Prepare the hyperparameter space.
        self.hyperparameter_space = self._get_task_hyperparameter_space()

        self.optimizer = BO(objective_function=self.evaluate_function,
                            configspace=self.hyperparameter_space,
                            max_runs=int(1e10),
                            task_id=self.model_id,
                            rng=np.random.RandomState(self.seed))

    def evaluate_function(self, config):
        input_node = self.root_node
        outout_node = self._parse(input_node, config)
        return self.evaluator(datanode=outout_node, name='fe')

    def optimize(self):
        while not self.is_finished:
            self.logger.debug('=' * 50)
            self.logger.debug('Start the ITERATION: %d' % self.iteration_id)
            self.logger.debug('=' * 50)
            self.iterate()
        return self.incumbent

    def iterate(self):
        result = None
        for _ in range(self.number_of_unit_resource):
            result = self._iterate()
        return result

    def _iterate(self):
        _start_time = time.time()
        for _ in range(self.iter_num_per_unit_resource):
            self.optimizer.iterate()

        runhistory = self.optimizer.get_history()
        self.incumbent, self.incumbent_score = runhistory.get_incumbents()[0]
        self.incumbent_score = 1 - self.incumbent_score
        iteration_cost = time.time() - _start_time
        return self.incumbent_score, iteration_cost, self.incumbent

    def _parse(self, data_node: DataNode, config):
        return None

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
        if trans_set is None:
            if self.task_type in CLS_TASKS:
                self.trans_types = TRANS_CANDIDATES['classification']
            else:
                self.trans_types = TRANS_CANDIDATES['regression']
        else:
            self.trans_types = trans_set

        # Avoid transformations, which would take too long
        # Combinations of non-linear models with feature learning.
        # feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
        if self.task_type in CLS_TASKS:
            classifier_set = ["adaboost", "decision_tree", "extra_trees",
                              "gradient_boosting", "k_nearest_neighbors",
                              "libsvm_svc", "random_forest", "gaussian_nb", "decision_tree"]

            if self.model_id in classifier_set:
                for tran_id in [12, 13, 15]:
                    if tran_id in self.trans_types:
                        self.trans_types.remove(tran_id)
        # TODO: for regression task, the trans types should be elaborated.
        pass
