import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from solnml.components.models.base_nn import BaseODClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from .nn_utils.yolov3_utils import *


class Yolov3(BaseODClassificationNeuralNetwork):
    def __init__(self, optimizer, batch_size, epoch_num, lr_decay, step_decay,
                 sgd_learning_rate=None, sgd_momentum=None, adam_learning_rate=None, beta1=None,
                 random_state=None, device='cpu'):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay
        self.step_decay = step_decay
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1
        self.random_state = random_state
        self.model = None
        self.device = torch.device(device)
        self.time_limit = None

    def fit(self, dataset, **kwargs):
        from .nn_utils.yolov3 import Darknet

        self.model = Darknet(num_class=len(dataset.classes), img_size=dataset.image_size)
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def score(self, dataset, metric, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        if isinstance(dataset, Dataset):
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
            img_size = dataset.image_size
        else:
            loader = DataLoader(dataset=dataset.val_dataset, batch_size=batch_size,
                                num_workers=4, collate_fn=dataset.val_dataset.collate_fn)
            img_size = dataset.val_dataset.image_size
            # else:
            #     loader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size,
            #                         sampler=dataset.val_sampler, num_workers=4,
            #                         collate_fn=dataset.train_dataset.collate_fn)
            #     img_size = dataset.train_dataset.image_size

        self.model.eval()
        labels = []
        sample_metrics = []
        with torch.no_grad():
            for batch_i, (_, batch_x, batch_y) in enumerate(loader):
                # Extract labels
                labels += batch_y[:, 1].tolist()
                # Rescale target
                batch_y[:, 2:] = xywh2xyxy(batch_y[:, 2:])
                batch_y[:, 2:] *= img_size

                batch_x = Variable(batch_x, requires_grad=False)

                with torch.no_grad():
                    outputs = self.model(batch_x.to(self.device))
                    outputs = non_max_suppression(outputs, conf_thres=0.001, nms_thres=0.5)

                sample_metrics += get_batch_statistics(outputs, batch_y, iou_threshold=0.5)

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        # Return mAP
        return AP.mean()

    def predict(self, dataset: Dataset, sampler=None, batch_size=None):
        if not self.model:
            raise ValueError("Model not fitted!")
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                            num_workers=4, collate_fn=dataset.collate_fn)
        self.model.to(self.device)
        self.model.eval()

        predictions = list()
        with torch.no_grad():
            for i, (_, batch_x) in enumerate(loader):
                outputs = self.model(batch_x.float().to(self.device))
                predictions.append(outputs.to('cpu').detach().numpy())
        return predictions

    def set_empty_model(self, dataset):
        from .nn_utils.yolov3 import Darknet

        self.model = Darknet(num_class=len(dataset.classes), img_size=dataset.image_size)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Yolov3',
                'name': 'Yolov3',
                'handles_regression': False,
                'handles_classification': False,
                'handles_multiclass': False,
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
                "sgd_learning_rate", lower=1e-4, upper=1e-2, default_value=2e-3, log=True)
            sgd_momentum = UniformFloatHyperparameter(
                "sgd_momentum", lower=0, upper=0.9, default_value=0, log=False)
            adam_learning_rate = UniformFloatHyperparameter(
                "adam_learning_rate", lower=1e-5, upper=1e-3, default_value=2e-4, log=True)
            beta1 = UniformFloatHyperparameter(
                "beta1", lower=0.5, upper=0.999, default_value=0.9, log=False)
            batch_size = CategoricalHyperparameter(
                "batch_size", [8, 16, 32], default_value=16)
            lr_decay = UnParametrizedHyperparameter("lr_decay", 0.8)
            step_decay = UnParametrizedHyperparameter("step_decay", 10)
            epoch_num = UnParametrizedHyperparameter("epoch_num", 200)
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
            space = {'batch_size': hp.choice('resnext_batch_size', [8, 16, 32]),
                     'optimizer': hp.choice('resnext_optimizer',
                                            [("SGD", {'sgd_learning_rate': hp.loguniform('resnext_sgd_learning_rate',
                                                                                         np.log(1e-4), np.log(1e-2)),
                                                      'sgd_momentum': hp.uniform('resnext_sgd_momentum', 0, 0.9)}),
                                             ("Adam", {'adam_learning_rate': hp.loguniform('resnext_adam_learning_rate',
                                                                                           np.log(1e-5), np.log(1e-3)),
                                                       'beta1': hp.uniform('resnext_beta1', 0.5, 0.999)})]),
                     'epoch_num': 200,
                     'lr_decay': 0.8,
                     'step_decay': 10
                     }
            return space
