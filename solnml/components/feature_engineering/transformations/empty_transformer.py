from solnml.components.feature_engineering.transformations.base_transformer import *


class Empty(Transformer):
    def __init__(self):
        super().__init__("empty_transformer", 0)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'

    def operate(self, input_datanodes, target_fields=None):
        return input_datanodes
