import numpy as np
import warnings
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.text_util import build_embeddings_index, load_text_embeddings


class Text2VectorTransformation(Transformer):
    def __init__(self, method='weighted', alpha=1e-4):
        super().__init__("text2vector", 50)
        self.method = method
        self.alpha = alpha
        self.input_type = [TEXT]
        self.output_type = [TEXT_EMBEDDING]
        self.compound_mode = 'replace'
        self.embedding_dict = None

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data
        X_new = X[:, target_fields]
        if not self.embedding_dict:
            self.embedding_dict = build_embeddings_index()
        _X = None
        for i in range(X_new.shape[1]):
            emb_output = load_text_embeddings(X_new[:, i], self.embedding_dict, method=self.method, alpha=self.alpha)
            if _X is None:
                _X = emb_output.copy()
            else:
                _X = np.hstack((_X, emb_output))
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        method = CategoricalHyperparameter("method", ['average', 'weighted'], default_value='weighted')
        alpha = UniformFloatHyperparameter("alpha", 1e-5, 1e-3, log=True, default_value=1e-4)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([method, alpha])

        alpha_cond = EqualsCondition(alpha, method, 'weighted')
        cs.add_conditions([alpha_cond])

        return cs