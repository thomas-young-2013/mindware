import numpy as np
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_nn import BaseODClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


# TODO: Remain to be modified
class RetinaNet(BaseODClassificationNeuralNetwork):

    def fit(self, dataset):
        from .nn_utils.retinanet import resnet101

        # TODO: Standardize inputs
        self.model = resnet101(num_classes=None)
        self.model.training = True
        self.model.to(self.device)
        super().fit(dataset)
        return self

    def predict(self, X):
        self.model.training = False
        return super().predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RetinaNet',
                'name': 'RetinaNet',
                'handles_regression': False,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}
