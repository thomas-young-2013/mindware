from fe_components.transformers.base_transformer import *


class Empty(Transformer):
    def __init__(self):
        super().__init__("empty_transformer", 0)

    def operate(self, input_datanodes, target_fields=None):
        assert not isinstance(input_datanodes, list)
        return input_datanodes
