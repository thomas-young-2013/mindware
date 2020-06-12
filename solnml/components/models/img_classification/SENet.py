import numpy as np
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class SENetClassifier(BaseImgClassificationNeuralNetwork):
    def __init__(self, learning_rate, beta1, batch_size, epoch_num,
                 lr_decay, step_decay, random_state=None, device='cpu'):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.step_decay = step_decay
        self.random_state = random_state
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None

    def fit(self, X, y, sample_weight=None):
        from .nn_utils.senet import SENet, SEResNeXtBottleneck

        self.model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                           dropout_p=None, inplanes=64, input_3x3=False,
                           downsample_kernel_size=1, downsample_padding=0,
                           num_classes=len(set(y)), grayscale=True if X.shape[1] == 1 else False)

        self.model.to(self.device)
        super().fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SENet',
                'name': 'SENet Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            learning_rate = UniformFloatHyperparameter(
                "learning_rate", lower=1e-5, upper=1e-3, default_value=2e-4, log=True)
            beta1 = UniformFloatHyperparameter(
                "beta1", lower=0.5, upper=0.999, default_value=0.9, log=False)
            batch_size = CategoricalHyperparameter(
                "batch_size", [8, 16, 32], default_value=16)
            lr_decay = UnParametrizedHyperparameter("lr_decay", 0.8)
            step_decay = UnParametrizedHyperparameter("step_decay", 10)
            epoch_num = UnParametrizedHyperparameter("epoch_num", 100)
            cs.add_hyperparameters([learning_rate, beta1, batch_size, epoch_num, lr_decay, step_decay])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'learning_rate': hp.loguniform('senet_learning_rate', np.log(1e-5), np.log(1e-3)),
                     'batch_size': hp.choice('senet_batch_size', [8, 16, 32]),
                     'beta1': hp.loguniform('senet_beta1', 0.5, 0.999),
                     'epoch_num': 100,
                     'lr_decay': 10,
                     'step_decay': 10
                     }
            return space
