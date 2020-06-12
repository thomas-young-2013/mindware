import os
from solnml.utils.logging_utils import setup_logger, get_logger
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import CLS_TASKS, REG_TASKS, IMG_CLS
from solnml.components.ensemble import ensemble_list
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.models.classification import _classifiers
from solnml.components.models.regression import _regressors
from solnml.components.models.imbalanced_classification import _imb_classifiers
from solnml.components.models.img_classification import _classifiers as _img_classifiers
from solnml.components.meta_learning.algorithm_recomendation.algorithm_advisor import AlgorithmAdvisor
from solnml.bandits.first_layer_bandit import FirstLayerBandit

classification_algorithms = _classifiers.keys()
imb_classication_algorithms = _imb_classifiers.keys()
regression_algorithms = _regressors.keys()
img_classification_algorithms = _img_classifiers.keys()


class AutoML(object):
    def __init__(self, time_limit=300,
                 dataset_name='default_name',
                 amount_of_resource=None,
                 task_type=None,
                 metric='bal_acc',
                 include_algorithms=None,
                 ensemble_method='ensemble_selection',
                 enable_meta_algorithm_selection=True,
                 enable_fe=True,
                 per_run_time_limit=150,
                 ensemble_size=50,
                 evaluation='holdout',
                 output_dir="logs",
                 logging_config=None,
                 random_state=1,
                 n_jobs=1):
        self.metric_id = metric
        self.metric = get_metric(self.metric_id)

        self.dataset_name = dataset_name
        self.time_limit = time_limit
        self.seed = random_state
        self.per_run_time_limit = per_run_time_limit
        self.output_dir = output_dir
        self.logging_config = logging_config
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(self.dataset_name)

        self.evaluation_type = evaluation
        self.amount_of_resource = amount_of_resource
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.enable_fe = enable_fe
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.solver = None

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type in CLS_TASKS:
                if task_type == IMG_CLS:
                    self.include_algorithms = list(img_classification_algorithms)
                else:
                    self.include_algorithms = list(classification_algorithms)
            elif task_type in REG_TASKS:
                self.include_algorithms = list(regression_algorithms)
            else:
                raise ValueError("Unknown task type %s" % task_type)
        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def _get_logger(self, name):
        logger_name = 'SolnML-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

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
                self.logger.info('Executing meta-learning based algorithm recommendation!')
                self.logger.info('Algorithms recommended: %s' % ','.join(self.include_algorithms))
            except Exception as e:
                self.logger.error("Meta-learning failed!")

        # Check whether this dataset is balanced or not.
        # if self.task_type in CLS_TASKS and is_unbalanced_dataset(train_data):
        #     # self.include_algorithms = imb_classication_algorithms
        #     self.logger.info('Input dataset is imbalanced!')
        #     train_data = DataBalancer().operate(train_data)
        if self.amount_of_resource is None:
            trial_num = len(self.include_algorithms) * 30
        else:
            trial_num = self.amount_of_resource

        self.solver = FirstLayerBandit(self.task_type, trial_num,
                                       self.include_algorithms, train_data,
                                       per_run_time_limit=self.per_run_time_limit,
                                       dataset_name=self.dataset_name,
                                       ensemble_method=self.ensemble_method,
                                       ensemble_size=self.ensemble_size,
                                       inner_opt_algorithm='fixed',
                                       metric=self.metric,
                                       enable_fe=self.enable_fe,
                                       fe_algo='bo',
                                       seed=self.seed,
                                       time_limit=self.time_limit,
                                       eval_type=self.evaluation_type,
                                       output_dir=self.output_dir)
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

    def get_ens_model_info(self):
        if self.ensemble_method is not None:
            return self.solver.es.get_ens_model_info()
        else:
            return None
