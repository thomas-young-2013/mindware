import abc
from typing import Optional
from automlToolkit.utils.data_manager import DataManager
from automlToolkit.components.utils.constants import *
from automlToolkit.components.feature_engineering.transformation_graph import DataNode
from automlToolkit.components.feature_engineering.transformation_graph import TransformationGraph
from automlToolkit.components.feature_engineering.transformations.preprocessor.imputer import ImputationTransformation
from automlToolkit.components.feature_engineering.transformations.preprocessor.onehot_encoder import OneHotTransformation
from automlToolkit.components.feature_engineering.transformations.selector.variance_selector import VarianceSelector
from automlToolkit.components.fe_optimizers.evaluation_based_optimizer import EvaluationBasedOptimizer
from automlToolkit.utils.logging_utils import setup_logger, get_logger


class FEPipeline(object, metaclass=abc.ABCMeta):
    """
    This controls the whole pipeline for feature engineering.
    """
    def __init__(self, optimizer_type='eval_base', time_budget=None,
                 maximum_evaluation_num=None,
                 fe_enabled=True, evaluator=None, debug=False, seed=1,
                 tmp_directory='./', logging_config=None):
        self.maximum_evaluation_num = maximum_evaluation_num
        self.time_budget = time_budget

        self.fe_enabled = fe_enabled
        self.debug_enabled = debug
        self.transgraph = TransformationGraph()
        self.optimizer_type = optimizer_type
        self.evaluator = evaluator

        self.root_node = None
        self.raw_range = None
        self.cleaned_node = None
        self.optimizer = None
        self._logger = None
        self._seed = seed
        self.tmp_directory = tmp_directory
        self.logging_config = logging_config

    def preprocess(self):
        # In imputation, the data in node is a DataFrame.
        raw_dataframe = self.root_node.data[0]
        types = self.root_node.feature_types
        input_node = self.root_node
        # print(raw_dataframe.shape)
        # print(types)

        # Filter the uninformative columns.
        uninformative_columns, uninformative_idx = list(), list()
        for idx, column in enumerate(list(raw_dataframe)):
            if raw_dataframe[column].isnull().values.all():
                uninformative_columns.append(column)
                uninformative_idx.append(idx)
                continue
            if types[idx] == CATEGORICAL:
                num_sample = input_node.data[0].shape[0]
                num_unique = len(set(input_node.data[0][column]))
                if num_unique >= int(0.8*num_sample):
                    uninformative_columns.append(column)
                    uninformative_idx.append(idx)

        input_node.feature_types = [types[idx] for idx in range(len(types)) if idx not in uninformative_idx]
        raw_dataframe = raw_dataframe.drop(uninformative_columns, axis=1)
        input_node.data[0] = raw_dataframe

        imputation_needed = raw_dataframe.isnull().values.any()
        if imputation_needed:
            for idx, column in enumerate(list(raw_dataframe)):
                if raw_dataframe[column].isnull().values.any():
                    feature_type = types[idx]
                    if feature_type in [CATEGORICAL, ORDINAL]:
                        imputer = ImputationTransformation('most_frequent')
                        input_node = imputer.operate(input_node, [idx])
                    else:
                        imputer = ImputationTransformation('median')
                        input_node = imputer.operate(input_node, [idx])
                    if self.debug_enabled:
                        print(input_node)
                        print(input_node.data)

        # Conduct one-hot encoding for categorical features.
        categorical_fields = [idx for idx, type in enumerate(input_node.feature_types) if type == CATEGORICAL]
        if len(categorical_fields) > 0:
            onehot_encoder = OneHotTransformation()
            input_node = onehot_encoder.operate(input_node, categorical_fields)

        # Removing columns with the same values.
        variance_selector = VarianceSelector()
        input_node = variance_selector.operate(input_node)

        # Label encoding.
        self.encode_label(input_node)
        self.cleaned_node = input_node

    def encode_label(self, input_node):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        X, y = input_node.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            y = LabelEncoder().fit_transform(y)
        input_node.data = (X, y)

    def as_node(self, datamanager: DataManager, test=False):
        data = [datamanager.train_X, datamanager.train_y]
        if test:
            data = [datamanager.test_X, datamanager.test_y]

        return DataNode(data, datamanager.feature_types)

    def fit_transform(self, datamanager: DataManager or DataNode):
        self.fit(datamanager)
        if self.optimizer is None:
            return self.cleaned_node
        return self.optimizer.get_incumbent()

    def _get_logger(self, name):
        import os
        logger_name = 'AutoFeatureEngine (%d):%s' % (self._seed, name)
        setup_logger(os.path.join(self.tmp_directory, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def fit(self, datamanager: DataManager, dataset_name: Optional[str] = 'dataset'):
        self._logger = self._get_logger(dataset_name)

        if isinstance(datamanager, DataManager):
            self.root_node = self.as_node(datamanager)
            self.preprocess()
        elif isinstance(datamanager, DataNode):
            self.cleaned_node = datamanager
        else:
            raise ValueError('Invalid object type!')

        print('After preprocessing', self.cleaned_node.data[0].shape)

        # TODO: dtype is object.
        if self.fe_enabled:
            if self.optimizer_type == 'eval_base':
                self.optimizer = EvaluationBasedOptimizer(self.cleaned_node, self.evaluator, self._seed)
            else:
                raise ValueError('invalid optimizer type!')

            self.optimizer.time_budget = self.time_budget
            self.optimizer.maximum_evaluation_num = self.maximum_evaluation_num
            self.optimizer.optimize()

        # Conduct feature engineering iteratively.
        return self

    def iterate(self):
        pass

    def transform(self, test_data: DataManager or DataNode):
        if isinstance(test_data, DataManager):
            self.root_node = self.as_node(test_data)
            self.preprocess()
        else:
            self.cleaned_node = test_data

        if not self.fe_enabled:
            return self.cleaned_node
        return self.optimizer.apply(self.cleaned_node)
