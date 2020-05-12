from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import CLS_TASKS, REG_TASKS
from solnml.components.ensemble import ensemble_list
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.models.classification import _classifiers
from solnml.components.models.regression import _regressors
from solnml.components.models.imbalanced_classification import _imb_classifiers
from solnml.utils.functions import is_unbalanced_dataset
from solnml.components.feature_engineering.transformations.preprocessor.to_balanced import DataBalancer
from solnml.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from solnml.bandits.first_layer_bandit import FirstLayerBandit

classification_algorithms = _classifiers.keys()
imb_classication_algorithms = _imb_classifiers.keys()
regression_algorithms = _regressors.keys()


class AutoML(object):
    def __init__(self, time_limit=300,
                 amount_of_resource=None,
                 task_type=None,
                 metric='bal_acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 enable_meta_algorithm_selection=True,
                 per_run_time_limit=150,
                 ensemble_size=50,
                 evaluation='holdout',
                 output_dir="/tmp/",
                 random_state=1,
                 n_jobs=1):
        self.metric_id = metric
        self.metric = get_metric(self.metric_id)
        self.time_limit = time_limit
        self.seed = random_state
        self.amount_of_resource = amount_of_resource
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.output_dir = output_dir
        self.evaluation_type = evaluation
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.solver = None

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type in CLS_TASKS:
                self.include_algorithms = list(classification_algorithms)
            elif task_type in REG_TASKS:
                self.include_algorithms = list(regression_algorithms)
            else:
                raise ValueError("Unknown task type %s" % task_type)
        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def fit(self, train_data: DataNode, dataset_id=None):
        """
        this function includes this following two procedures.
            1. tune each algorithm's hyperparameters.
            2. engineer each algorithm's features automatically.
        :param train_data:
        :return:
        """
        if self.enable_meta_algorithm_selection:
            try:
                alad = AlgorithmAdvisor(task_type=self.task_type, n_algorithm=9,
                                        metric=self.metric_id)
                n_algo = 5
                model_candidates = alad.fetch_algorithm_set(train_data, dataset_id=dataset_id)
                include_models = list()
                for algo in model_candidates:
                    if algo in self.include_algorithms and len(include_models) < n_algo:
                        include_models.append(algo)
                self.include_algorithms = include_models
                print('Algorithms recommended:', self.include_algorithms)
            except Exception as e:
                print(e)

        # Check whether this dataset is balanced or not.
        if self.task_type in CLS_TASKS and is_unbalanced_dataset(train_data):
            # self.include_algorithms = imb_classication_algorithms
            train_data = DataBalancer().operate(train_data)
        if self.amount_of_resource is None:
            trial_num = len(self.include_algorithms) * 30
        else:
            trial_num = self.amount_of_resource

        self.solver = FirstLayerBandit(self.task_type, trial_num,
                                       self.include_algorithms, train_data,
                                       output_dir='logs',
                                       per_run_time_limit=self.per_run_time_limit,
                                       dataset_name=dataset_id,
                                       ensemble_size=self.ensemble_size,
                                       inner_opt_algorithm='fixed',
                                       metric=self.metric,
                                       fe_algo='bo',
                                       seed=self.seed,
                                       time_limit=self.time_limit,
                                       eval_type=self.evaluation_type)
        self.solver.optimize()

    def refit(self):
        self.solver.refit()

    def predict_proba(self, test_data: DataNode):
        return self.solver.predict_proba(test_data)

    def predict(self, test_data: DataNode):
        return self.solver.predict(test_data)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        return metric_func(self, test_data, test_data.data[1])
