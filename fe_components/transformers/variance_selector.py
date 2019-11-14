from fe_components.transformers.base_transformer import *


class VarianceSelector(Transformer):
    def __init__(self, param=1e-4):
        super().__init__("variance_selector", 11)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.params = param

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import VarianceThreshold

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
            X_new = X.copy()
        else:
            X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        is_selected = [True] * len(target_fields)

        if self.model is None:
            self.model = VarianceThreshold(threshold=self.params)
            self.model.fit(X_new)

        for idx, var in enumerate(self.model.variances_):
            is_selected[idx] = True if var >= self.params else False

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        _X = self.model.transform(X_new)

        if len(irrevalent_fields) > 0:
            new_X = np.hstack((_X, X[:, irrevalent_fields]))
        else:
            new_X = _X
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)

        return output_datanode
