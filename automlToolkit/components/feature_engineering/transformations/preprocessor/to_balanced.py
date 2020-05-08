from automlToolkit.components.feature_engineering.transformations.base_transformer import *
from automlToolkit.utils.functions import is_unbalanced_dataset
from imblearn.combine import SMOTEENN


class DataBalancer(Transformer):
    def __init__(self):
        super().__init__("to_balanced", 32)

    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        if self.model is None:
            self.model = SMOTEENN(random_state=self.random_state)

        if is_unbalanced_dataset(input_datanode):
            X_res, y_res = self.model.fit_resample(X, y)
        else:
            X_res, y_res = X, y

        X_output = X_res.copy()
        y_output = y_res.copy()

        output_datanode = DataNode((X_output, y_output), input_datanode.feature_types, input_datanode.task_type)
        output_datanode.trans_hist.append(self.type)

        return output_datanode
