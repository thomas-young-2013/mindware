from mindware.components.models.base_searcher import BaseSearcher
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.models.search.nas_utils.nb2_utils.nasbench2_ops import OPS, NUM_OPERATIONS

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

NUM_WORKERS = 10


class NASBench201Searcher(BaseSearcher):
    def __init__(self, arch_config, batch_size=128,
                 epoch_num=108, learning_rate=3e-4,
                 weight_decay=1e-4, lr_decay=1e-1,
                 random_state=None, grayscale=False,
                 device='cpu', **kwargs):
        super().__init__(arch_config=arch_config,
                         batch_size=batch_size,
                         epoch_num=epoch_num,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         lr_decay=lr_decay,
                         random_state=random_state,
                         grayscale=grayscale,
                         device=device,
                         **kwargs)

        self.space = 'nasbench201'

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NB201Searcher',
                'name': 'NASBench201Searcher',
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
        operations = list(OPS.keys())
        for i in range(NUM_OPERATIONS):
            cs.add_hyperparameter(
                CategoricalHyperparameter('op_%d' % i, choices=operations,
                                          default_value=operations[-1]))

        return cs
