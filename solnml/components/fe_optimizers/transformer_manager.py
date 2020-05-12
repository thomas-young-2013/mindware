import abc
import typing
from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.components.feature_engineering.transformations import _transformers, _type_infos
from solnml.components.utils.configspace_utils import sample_configurations


class TransformerManager(object, metaclass=abc.ABCMeta):
    def __init__(self, disable_hpo=False, random_state=1):
        # Store the executed hyperparameter configurations for each transformer.
        self.hyper_configs = dict()
        self.disable_hpo = disable_hpo
        self.random_state = random_state
        # Store the executed transformations on this node.
        self.node_trans_pairs = dict()
        # init the root node.
        self.node_trans_pairs[0] = list()

    def add_execution_record(self, node_id: int, trans_id: int):
        if node_id not in self.node_trans_pairs:
            self.node_trans_pairs[node_id] = list()
        if trans_id not in self.node_trans_pairs[node_id]:
            self.node_trans_pairs[node_id].append(trans_id)

    def get_transformations(self, node: DataNode, trans_types: typing.List,
                            batch_size: int = 1):
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
        node_id = node.node_id
        if node_id not in self.node_trans_pairs:
            self.node_trans_pairs[node_id] = list()

        for id in trans_ids:
            if id not in self.hyper_configs:
                self.hyper_configs[id] = list()

            trans_type = _transformers[id]().type
            if trans_type not in trans_types:
                continue

            # Avoid repeating the same transformation multiple times.
            if trans_type in node.trans_hist:
                continue

            transformer_class = _transformers[id]

            # For transformations without hyperparameters.
            if not hasattr(transformer_class, 'get_hyperparameter_search_space'):
                if trans_type not in self.node_trans_pairs[node_id]:
                    transformers.append(transformer_class())
                continue

            config_space = transformer_class().get_hyperparameter_search_space()
            if len(config_space.get_hyperparameters()) == 0 or self.disable_hpo:
                if trans_type not in self.node_trans_pairs[node_id]:
                    transformers.append(transformer_class())
                continue

            # For transformations with hyperparameters.
            sampled_configs = sample_configurations(
                config_space, batch_size, self.hyper_configs[id], seed=self.random_state)
            for config in sampled_configs:
                _transformer = transformer_class(**config.get_dictionary())
                transformers.append(_transformer)

            self.hyper_configs[id].extend(sampled_configs)

        return transformers
