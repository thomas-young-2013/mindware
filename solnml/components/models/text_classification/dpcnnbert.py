from solnml.components.models.base_nn import BaseTextClassificationNeuralNetwork
from solnml.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class DPCNNBertClassifier(BaseTextClassificationNeuralNetwork):

    def fit(self, dataset, **kwargs):
        from .nn_utils.dpcnnbert import DPCNNModel
        if dataset.config_path is None:
            config_path = self.config
        else:
            config_path = dataset.config_path

        self.model = DPCNNModel.from_pretrained(config_path, num_class=len(dataset.classes))
        self.model.to(self.device)
        super().fit(dataset, **kwargs)
        return self

    def set_empty_model(self, dataset):
        from .nn_utils.dpcnnbert import DPCNNModel
        if dataset.config_path is None:
            config_path = self.config
        else:
            config_path = dataset.config_path

        self.model = DPCNNModel.from_pretrained(config_path, num_class=len(dataset.classes))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DPCNNBert',
                'name': 'DPCNNBert Text Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}
