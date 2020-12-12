from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.utils.functions import is_imbalanced_dataset


class DataBalancer(Transformer):
    type = 33

    def __init__(self):
        super().__init__("smote_balancer")

    def operate(self, input_datanode, target_fields=None):
        output_datanode = input_datanode.copy_()

        output_datanode.trans_hist.append(self.type)
        if is_imbalanced_dataset(input_datanode):
            output_datanode.data_balance = 1
        return output_datanode
