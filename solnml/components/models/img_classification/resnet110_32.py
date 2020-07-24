import numpy as np
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class ResNet110_32Classifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.resnet_32 import resnet110
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = resnet110(num_classes=len(dataset.train_dataset.classes))
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.resnet_32 import resnet110
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = resnet110(num_classes=len(dataset.train_dataset.classes))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ResNet110_32',
                'name': 'ResNet110_32 Classifier',
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
            optimizer = CategoricalHyperparameter('optimizer', ['SGD', 'Adam'], default_value='SGD')
            sgd_learning_rate = UniformFloatHyperparameter(
                "sgd_learning_rate", lower=1e-5, upper=1e-2, default_value=2e-3, log=True)
            sgd_momentum = UniformFloatHyperparameter(
                "sgd_momentum", lower=0, upper=0.9, default_value=0, log=False)
            adam_learning_rate = UniformFloatHyperparameter(
                "adam_learning_rate", lower=1e-5, upper=1e-3, default_value=2e-4, log=True)
            beta1 = UniformFloatHyperparameter(
                "beta1", lower=0.5, upper=0.999, default_value=0.9, log=False)
            batch_size = CategoricalHyperparameter(
                "batch_size", [16, 32], default_value=32)
            lr_decay = UnParametrizedHyperparameter("lr_decay", 0.8)
            step_decay = UnParametrizedHyperparameter("step_decay", 10)
            epoch_num = UnParametrizedHyperparameter("epoch_num", 150)
            cs.add_hyperparameters(
                [optimizer, sgd_learning_rate, adam_learning_rate, sgd_momentum, beta1, batch_size, epoch_num, lr_decay,
                 step_decay])
            sgd_lr_depends_on_sgd = EqualsCondition(sgd_learning_rate, optimizer, "SGD")
            adam_lr_depends_on_adam = EqualsCondition(adam_learning_rate, optimizer, "Adam")
            sgd_momentum_depends_on_sgd = EqualsCondition(sgd_momentum, optimizer, "SGD")
            cs.add_conditions([sgd_lr_depends_on_sgd, adam_lr_depends_on_adam, sgd_momentum_depends_on_sgd])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'batch_size': hp.choice('resnet110_32_batch_size', [16, 32]),
                     'optimizer': hp.choice('resnet110_32_optimizer',
                                            [(
                                                "SGD",
                                                {'sgd_learning_rate': hp.loguniform('resnet110_32_sgd_learning_rate',
                                                                                    np.log(1e-5), np.log(1e-2)),
                                                 'sgd_momentum': hp.uniform('resnet110_32_sgd_momentum', 0, 0.9)}),
                                                (
                                                    "Adam",
                                                    {'adam_learning_rate': hp.loguniform(
                                                        'resnet110_32_adam_learning_rate',
                                                        np.log(1e-5), np.log(1e-3)),
                                                        'beta1': hp.uniform('resnet110_32_beta1', 0.5, 0.999)})]),
                     'epoch_num': 100,
                     'lr_decay': 0.8,
                     'step_decay': 10
                     }
            return space
