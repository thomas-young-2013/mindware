from solnml.components.feature_engineering.transformations.base_transformer import *


class ImputationTransformation(Transformer):
    type = 1

    def __init__(self, param='mean'):
        super().__init__("imputer")
        self.params = param

    def operate(self, input_datanode, target_fields=None):
        from sklearn.impute import SimpleImputer

        X, y = input_datanode.data
        self.target_fields = target_fields

        # Fetch the related fields.
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_input = X[:, target_fields]
        # Different imputation strategies applied for columns.
        if self.model is None:
            self.model = SimpleImputer(strategy=self.params, copy=False)
            self.model.fit(X_input)
        new_X = self.model.transform(X_input).reshape(-1, 1)
        X_output = X.copy()
        X_output[:, target_fields] = new_X
        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode((X_output, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
