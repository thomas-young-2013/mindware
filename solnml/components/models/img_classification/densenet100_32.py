from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class DenseNet100_32Classifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.densenet_32 import densenet100bc
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = densenet100bc(num_classes=len(dataset.train_dataset.classes))
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.densenet_32 import densenet100bc
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = densenet100bc(num_classes=len(dataset.classes))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DenseNet100_32',
                'name': 'DenseNet100_32 Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}
