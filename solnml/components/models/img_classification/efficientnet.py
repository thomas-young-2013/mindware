from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class EfficientNetClassifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.efficientnet import EfficientNet
        self.model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True,
                                                  num_classes=len(dataset.train_dataset.classes),
                                                  in_channels=1 if self.grayscale else 3)
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.efficientnet import EfficientNet
        self.model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True,
                                                  num_classes=len(dataset.classes),
                                                  in_channels=1 if self.grayscale else 3)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'EfficientNet',
                'name': 'EfficientNet Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}