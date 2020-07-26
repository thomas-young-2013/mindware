from solnml.components.models.base_nn import BaseImgClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class DenseNet190_32Classifier(BaseImgClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.densenet_32 import densenet190bc
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = densenet190bc(num_classes=len(dataset.train_dataset.classes))
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.densenet_32 import densenet190bc
        if self.grayscale:
            raise ValueError("Only support RGB inputs!")
        self.model = densenet190bc(num_classes=len(dataset.classes))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DenseNet190_32',
                'name': 'DenseNet190_32 Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}