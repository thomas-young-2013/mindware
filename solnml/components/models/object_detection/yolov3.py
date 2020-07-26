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
