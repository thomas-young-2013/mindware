from automlToolkit.components.feature_engineering.transformations.base_transformer import *
from imblearn.combine import SMOTEENN

class DataBalancer(Transformer):
    def __init__(self):
        super().__init__("data_balancer", -20)


    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        if self.model is None:
            self.model = SMOTEENN(random_state=self.random_state)
        X_res, y_res = self.model.fit_resample(X, y)


        X_output = X_res.copy()
        y_output = y_res.copy()

        output_datanode = DataNode((X_output, y_output), feature_types, input_datanode.task_type)
        output_datanode.data_balance = 1
        output_datanode.trans_hist.append(self.type)

        return output_datanode
