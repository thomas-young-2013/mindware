import numpy as np
from solnml.components.feature_engineering.transformations.base_transformer import *


class EmptyTransformer(Transformer):
    def __init__(self):
        super().__init__("empty_transformer", 0)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'concatenate'

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, _ = input_datanode.data
        return np.zeros((X.shape[0], 0))
