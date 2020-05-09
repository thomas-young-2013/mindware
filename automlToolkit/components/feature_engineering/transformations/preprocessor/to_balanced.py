from automlToolkit.components.feature_engineering.transformations.base_transformer import *
from automlToolkit.utils.functions import is_unbalanced_dataset
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

class DataBalancer(Transformer):
    def __init__(self):
        super().__init__("to_balanced", 32)

    def operate(self, input_datanode, target_fields=None):
        output_datanode = input_datanode.copy_()

        output_datanode.trans_hist.append(self.type)
        if is_unbalanced_dataset(input_datanode):
            output_datanode.data_balance = 1
        return output_datanode
