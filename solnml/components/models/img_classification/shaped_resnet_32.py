import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from solnml.components.utils.configspace_utils import check_for_bool


class ShapedResNet_32Classifier(BaseImgClassificationNeuralNetwork):
    def __init__(self, depth, inplane, optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None,
                 grayscale=False, device='cpu', **kwargs):
        super(BaseImgClassificationNeuralNetwork, self).__init__()
        self.depth = depth
        self.inplane = inplane
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = epoch_num
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.nesterov = check_for_bool(nesterov)
        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1
        self.random_state = random_state
        self.grayscale = grayscale
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None
        self.load_path = None

        self.optimizer_ = None
        self.scheduler = None
        self.early_stop = None
        self.cur_epoch_num = 0

    def fit(self, dataset, **kwargs):
        from .nn_utils.resnet_32 import shaped_resnet
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = shaped_resnet(depth=self.depth, inplane=self.inplane,
                                   num_classes=len(dataset.train_dataset.classes))
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, config, dataset):
        from .nn_utils.resnet_32 import shaped_resnet
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        depth = config['depth']
        inplane = config['inplane']
        self.model = shaped_resnet(depth=depth, inplane=inplane, num_classes=len(dataset.classes))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ShapedResNet_32',
                'name': 'ShapedResNet_32 Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        depth = UniformIntegerHyperparameter('depth', 20, 122, default_value=20, q=6)
        inplane = UniformIntegerHyperparameter('inplane', 16, 128, default_value=32)
        optimizer = CategoricalHyperparameter('optimizer', ['SGD'], default_value='SGD')
        sgd_learning_rate = CategoricalHyperparameter(
            "sgd_learning_rate", [1e-3, 3e-3, 7e-3, 1e-2, 3e-2, 7e-2, 1e-1],
            default_value=1e-1)
        sgd_momentum = UniformFloatHyperparameter(
            "sgd_momentum", lower=0.5, upper=0.99, default_value=0.9, log=False)
        nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')

        batch_size = CategoricalHyperparameter(
            "batch_size", [32, 64, 128], default_value=32)
        lr_decay = CategoricalHyperparameter("lr_decay", [1e-2, 5e-2, 1e-1, 2e-1], default_value=1e-1)
        weight_decay = CategoricalHyperparameter("weight_decay", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                                                 default_value=1e-4)
        epoch_num = UnParametrizedHyperparameter("epoch_num", 150)
        cs.add_hyperparameters(
            [depth, inplane, optimizer, sgd_learning_rate, sgd_momentum, batch_size,
             epoch_num, lr_decay, weight_decay, nesterov])
        return cs
