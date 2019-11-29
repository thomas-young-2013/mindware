from components.transformers.base_transformer import *


class DataBalancer(Transformer):
    def __init__(self):
        super().__init__("data_balancer", 20)

    def operate(self, input_datanode, target_fields=None):
        from imblearn.over_sampling import SMOTE

        X, y = input_datanode.data
        if y is None:
            data = (X, y)
        else:
            X_resampled, y_resampled = SMOTE().fit_resample(X, y)
            data = (X_resampled, y_resampled)

        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode(data, new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
