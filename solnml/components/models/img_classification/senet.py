from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class SENetClassifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.senet import se_resnext101_32x4d
        self.model = se_resnext101_32x4d(num_classes=len(dataset.train_dataset.classes),
                                         grayscale=self.grayscale,
                                         pretrained='imagenet')

        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.senet import se_resnext101_32x4d
        self.model = se_resnext101_32x4d(num_classes=len(dataset.classes),
                                         grayscale=self.grayscale,
                                         pretrained='imagenet')

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

