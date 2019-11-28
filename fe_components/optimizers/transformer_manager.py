import abc
import typing
from fe_components.transformation_graph import DataNode
from fe_components.transformers import _transformers, _type_infos
from fe_components.utils.configspace_utils import sample_configurations


class TransformerManager(object, metaclass=abc.ABCMeta):
    def __init__(self, random_state=1):
        # Store the executed hyperparameter configurations for each transformer.
        self.hyper_configs = dict()
        self.random_state = random_state

    def get_transformations(self, node: DataNode, trans_types: typing.List, batch_size=3):
        """
        Collect a batch of transformations with different hyperparameters in each call.
        :return: a list of transformations.
        """
        feat_type = list(set(node.feature_types))
        if isinstance(feat_type, str):
            feat_type = [feat_type]

        trans_ids = list()
        for _type in feat_type:
            trans_ids.extend(_type_infos[_type])
        trans_ids = list(set(trans_ids))
        transformers = list()

        for id in trans_ids:
            if id not in self.hyper_configs:
                self.hyper_configs[id] = list()
            if _transformers[id]().type not in trans_types:
                continue

            transformer_class = _transformers[id]
            if not hasattr(transformer_class, 'get_hyperparameter_search_space'):
                transformers.append(transformer_class())
                continue

            config_space = transformer_class().get_hyperparameter_search_space()
            if len(config_space.get_hyperparameters()) == 0:
                transformers.append(transformer_class())
                continue

            sampled_configs = sample_configurations(
                config_space, batch_size, self.hyper_configs[id], seed=self.random_state)
            for config in sampled_configs:
                _transformer = transformer_class(**config.get_dictionary())
                transformers.append(_transformer)

            self.hyper_configs[id].extend(sampled_configs)

        return transformers
