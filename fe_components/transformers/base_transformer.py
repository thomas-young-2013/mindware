import abc
import typing
import numpy as np
import pandas as pd

from fe_components.utils.utils import *
from fe_components.utils.constants import *
from fe_components.transformation_graph import DataNode


class Transformer(object, metaclass=abc.ABCMeta):
    """
    This is the parent class for all transformers.
    type specification:
        0: empty, no transformation.
        1: PRE transformations: imputer, one-hot.
        1: 4 scalers {func}.
        2: normalizer {norm}.
        3: discretizer {bins}, discrete_categorizer.
        4: 7 arithmetic transformers {func}, quantile_transformer.
        5: FS transformers: generic_univariate_selector, model_based_selector, rfe_selector, variance_selector.
        6: DR transformations: fast_ica_decomposer, feature_agglomeration_decomposer,
                               lda_decomposer, pca_decomposer, svd_decomposer.
        7: merger, random_trees_embedding.
        8: data_balancer.
        9: nystronem.
        10: polynomial_generator.

        0: empty.
        1: imputer.
        2: one-hot encoding.
        3: scaler.
        4: normalizer.
        5: quantile_transformer.
        6: generic_univariate_selector.
        7: model_based_selector.
        8: percentile_selector.
        9: variance_selector.
        10: fast_ica.
        11: feature_agg.
        12: kernel_pca.
        13: kitchen_sinks.
        14: lda_decomposer.
        15: nystronem.
        16: pca.
        17: polynomial.
        18: random_trees_embedding.
        19: svd_decomposer.
        20: data_balancer.
        ----
        21: arithmetic_operations.
        22: binary_operations.
        23: rfe.
        24: continous_discretizer.
        25: discrete_categorizer.
        ----
        26: merger.

    """
    def __init__(self, name, type, random_state=1):
        self.name = name
        self.type = type
        self.params = None
        self.model = None
        self._input_type = None
        self.output_type = None
        self.optional_params = None
        self.target_fields = None
        self._compound_mode = 'only_new'
        self.random_state = random_state
        self.sample_size = 2

    @property
    def compound_mode(self):
        return self._compound_mode

    @compound_mode.setter
    def compound_mode(self, mode):
        if mode not in ['only_new', 'concatenate', 'in_place']:
            raise ValueError('Invalid compound mode: %s!' % mode)
        self._compound_mode = mode

    @property
    def input_type(self):
        return self._input_type

    @input_type.setter
    def input_type(self, input_type: typing.List[str]):
        if not isinstance(input_type, list):
            input_type = [input_type]

        for type_ in input_type:
            if type_ not in FEATURE_TYPES:
                raise ValueError('Invalid feature type: %s!' % type_)
        self._input_type = input_type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Transformer):
            if self.name == other.name and self.type == other.type and self.input_type == other.input_type \
                    and self.output_type == other.output_type and self.params == other.params \
                    and self.model == other.model and self.compound_mode == other.compound_mode:
                return True
        return False

    @abc.abstractmethod
    def operate(self, data_nodes: DataNode or typing.List[DataNode],
                target_fields: None or typing.List):
        raise NotImplementedError()


def ease_trans(func):
    def dec(*args, **kwargs):
        param_name = 'target_fields'
        target_fields = None
        if len(args) == 3:
            trans, input, target_fields = args
            if type(target_fields) is list and len(target_fields) == 0:
                target_fields = None
        elif len(args) == 2:
            trans, input = args
            if param_name in kwargs and len(kwargs[param_name]) > 0:
                target_fields = kwargs[param_name]
        else:
            raise ValueError('The number of input is wrong!')

        if target_fields is None:
            target_fields = collect_fields(input.feature_types, trans.input_type)
        if len(target_fields) == 0:
            return input.copy_()

        X, y = input.data
        if isinstance(X, pd.DataFrame):
            X = X.values

        args = (trans, input, target_fields)
        _X = func(*args)
        if isinstance(trans.output_type, list):
            trans.output_type = trans.output_type[0]
        _types = [trans.output_type] * _X.shape[1]

        if trans.compound_mode == 'only_new':
            new_X = _X
            new_types = _types
        elif trans.compound_mode == 'concatenate':
            new_X = np.hstack((X, _X))
            new_types = input.feature_types.copy()
            new_types.extend(_types)
        else:
            assert _X.shape[1] == len(target_fields)
            new_X = X.copy()
            new_X[:, target_fields] = _X
            new_types = input.feature_types.copy()

        output_datanode = DataNode((new_X, y), new_types, input.task_type)
        trans.target_fields = target_fields
        return output_datanode

    return dec
