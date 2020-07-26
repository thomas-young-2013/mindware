from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class DenseNet121Classifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.pytorch_zoo_model import densenet121
        if self.grayscale:
            raise ValueError("Models from pytorch-model zoo only support RGB inputs!")
        self.model = densenet121(num_classes=len(dataset.train_dataset.classes), pretrained='imagenet')
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.pytorch_zoo_model import densenet121
        if self.grayscale:
            raise ValueError("Models from pytorch-model zoo only support RGB inputs!")
        self.model = densenet121(num_classes=len(dataset.classes), pretrained=None)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DenseNet121',
                'name': 'DenseNet121 Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}