import os
import abc
from automlToolkit.components.utils.constants import *
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.feature_engineering.transformations.preprocessor.imputer import ImputationTransformation
from automlToolkit.components.feature_engineering.transformations.preprocessor.onehot_encoder import \
    OneHotTransformation
from automlToolkit.components.feature_engineering.transformations.selector.variance_selector import VarianceSelector
from automlToolkit.components.fe_optimizers import EvaluationBasedOptimizer
from automlToolkit.components.metrics.metric import get_metric
from automlToolkit.utils.logging_utils import setup_logger, get_logger


class FEPipeline(object, metaclass=abc.ABCMeta):
    """
    This controls the whole pipeline for feature engineering.
    """

    def __init__(self, task_type=CLASSIFICATION,
                 optimizer_type='eval_base',
                 metric=None,
                 trans_set=None,
                 time_budget=None,
                 maximum_evaluation_num=None,
                 time_limit_per_trans=600,
                 mem_limit_per_trans=1024,
                 fe_enabled=True, evaluator=None, debug=False, seed=1,
                 tmp_directory='logs', logging_config=None, model_id=None,
                 task_id='Default'):
        self.fe_enabled = fe_enabled
        self.trans_set = trans_set
        self.maximum_evaluation_num = maximum_evaluation_num
        self.time_budget = time_budget
        self.time_limit_per_trans = time_limit_per_trans
        self.mem_limit_per_trans = mem_limit_per_trans
        self.optimizer_type = optimizer_type
        self.evaluator = evaluator
        self.optimizer = None

        self.metric = get_metric(metric)
        self.task_type = task_type
        self.task_id = task_id
        self.model_id = model_id
        self._seed = seed
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config
        self._logger = self._get_logger(task_id)

        # Set up backend.
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)

        # For data preprocessing.
        self.uninformative_columns, self.uninformative_idx = list(), list()
        self.variance_selector = None
        self.onehot_encoder = None

    def remove_uninf_cols(self, input_node: DataNode, train_phase=True):
        raw_dataframe = input_node.data[0]
        types = input_node.feature_types
        if train_phase:
            # Remove the uninformative columns.
            uninformative_columns, uninformative_idx = list(), list()
            for idx, column in enumerate(list(raw_dataframe)):
                if raw_dataframe[column].isnull().values.all():
                    uninformative_columns.append(column)
                    uninformative_idx.append(idx)
                    continue
                if types[idx] == CATEGORICAL:
                    num_sample = input_node.data[0].shape[0]
                    num_unique = len(set(input_node.data[0][column]))
                    if num_unique >= int(0.8 * num_sample):
                        uninformative_columns.append(column)
                        uninformative_idx.append(idx)
            self.uninformative_columns, self.uninformative_idx = uninformative_columns, uninformative_idx

        input_node.feature_types = [types[idx] for idx in range(len(types)) if idx not in self.uninformative_idx]
        raw_dataframe = raw_dataframe.drop(self.uninformative_columns, axis=1)
        input_node.data[0] = raw_dataframe
        return input_node

    def impute_cols(self, input_node: DataNode):
        raw_dataframe = input_node.data[0]
        feat_types = input_node.feature_types
        need_imputation = raw_dataframe.isnull().values.any()
        if need_imputation:
            for idx, column in enumerate(list(raw_dataframe)):
                if raw_dataframe[column].isnull().values.any():
                    feature_type = feat_types[idx]
                    if feature_type in [CATEGORICAL, ORDINAL]:
                        imputer = ImputationTransformation('most_frequent')
                        input_node = imputer.operate(input_node, [idx])
                    else:
                        imputer = ImputationTransformation('median')
                        input_node = imputer.operate(input_node, [idx])
        return input_node

    def one_hot(self, input_node: DataNode):
        # One-hot encoding TO categorical features.
        categorical_fields = [idx for idx, type in enumerate(input_node.feature_types) if type == CATEGORICAL]
        if len(categorical_fields) > 0:
            if self.onehot_encoder is None:
                self.onehot_encoder = OneHotTransformation()
            input_node = self.onehot_encoder.operate(input_node, categorical_fields)
        return input_node

    def remove_cols_with_same_values(self, input_node: DataNode):
        if self.variance_selector is None:
            self.variance_selector = VarianceSelector()
        input_node = self.variance_selector.operate(input_node)
        return input_node

    def encode_label(self, input_node):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        X, y = input_node.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            y = LabelEncoder().fit_transform(y)
        input_node.data = (X, y)
        return input_node

    def preprocess(self, input_node: DataNode, train_phase=True):
        print('=' * 20)
        print(input_node.shape)
        input_node = self.remove_uninf_cols(input_node, train_phase)
        print(input_node.shape)
        input_node = self.impute_cols(input_node)
        print(input_node.shape)
        input_node = self.one_hot(input_node)
        print(input_node.shape)
        input_node = self.remove_cols_with_same_values(input_node)
        print(input_node.shape)
        print('=' * 20)
        if self.task_type is None or self.task_type in CLS_TASKS:
            # Label encoding.
            input_node = self.encode_label(input_node)
        return input_node

    def fit_transform(self, data_node: DataNode):
        if not self.fe_enabled:
            preprocessed_node = self.preprocess(data_node, train_phase=True)
            return preprocessed_node
        else:
            self.fit(data_node)
            return self.optimizer.get_incumbent()

    def fit(self, data_node: DataNode):
        preprocessed_node = self.preprocess(data_node, train_phase=True)
        print('After pre-processing, the shape is', preprocessed_node.shape)

        # TODO: dtype is object.
        if self.fe_enabled:
            if self.optimizer_type == 'eval_base':
                self.optimizer = EvaluationBasedOptimizer(
                    self.task_type, preprocessed_node, self.evaluator,
                    self.model_id, self.time_limit_per_trans,
                    self.mem_limit_per_trans, self._seed, trans_set=self.trans_set
                )
            else:
                raise ValueError('invalid optimizer type!')
            self.optimizer.time_budget = self.time_budget
            self.optimizer.maximum_evaluation_num = self.maximum_evaluation_num
            self.optimizer.optimize()
        return self

    def transform(self, test_data: DataNode):
        preprocessed_node = self.preprocess(test_data, train_phase=False)
        print('After pre-processing, the shape is', preprocessed_node.shape)
        if not self.fe_enabled:
            return preprocessed_node
        return self.optimizer.apply(preprocessed_node, self.optimizer.incumbent)

    # Conduct feature engineering iteratively.
    def iterate(self):
        raise NotImplementedError('ooops!')

    def _get_logger(self, name):
        import os
        logger_name = 'AutomlToolkit-%s-%d:%s' % (self.task_id, self._seed, name)
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)
