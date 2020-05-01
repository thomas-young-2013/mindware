from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class DiscreteCategorizer(Transformer):
    def __init__(self, max_unique=10):
        super().__init__("discrete_categorizer", 25)
        self.input_type = [DISCRETE]
        self.output_type = CATEGORICAL
        self.max_unique = max_unique
        self.target_fields = None

    def operate(self, input_datanode, target_fields=None):
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder

        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
            if len(target_fields) == 0:
                return input_datanode.copy_()

        X, y = input_datanode.data
        if self.target_fields is None:
            target_fields = [idx for idx in target_fields if len(set(X[:, idx])) <= self.max_unique]
            # Fetch the fields to transform.
            self.target_fields = target_fields

        if len(self.target_fields) == 0:
            return input_datanode.copy_()

        X_input = X[:, self.target_fields]
        if self.model is None:
            self.model = OneHotEncoder(handle_unknown='ignore')  # Ignore values out of range
            self.model.fit(X_input)

        new_X = self.model.transform(X_input).toarray()
        X_output = X.copy()

        # Delete the original columns.
        X_output = np.delete(X_output, np.s_[self.target_fields], axis=1)
        X_output = np.hstack((X_output, new_X))
        feature_types = input_datanode.feature_types.copy()
        feature_types = list(np.delete(feature_types, self.target_fields))
        feature_types.extend([CATEGORICAL] * new_X.shape[1])
        output_datanode = DataNode((X_output, y), feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.enable_balance = input_datanode.enable_balance
        output_datanode.data_balance = input_datanode.data_balance
        output_datanode.trans_hist.append(self.type)
        return output_datanode
