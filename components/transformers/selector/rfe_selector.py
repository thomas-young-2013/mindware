from components.transformers.base_transformer import *


# TODO: Only for classification
class RecursiveFeatureEliminationSelector(Transformer):
    def __init__(self, param='lr', min_features=1):
        super().__init__("rfe_selector", 23)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.params = param
        self.min_features = min_features
        self.optional_params = ['lr', 'rf']

    def operate(self, input_datanode: DataNode, target_fields=None):
        from sklearn.feature_selection import RFECV

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        self.min_features = max(self.min_features, n_fields//20)
        if self.model is None:
            if self.params == 'lr':
                from sklearn.linear_model import LogisticRegression
                base_model = LogisticRegression(solver='lbfgs')
            elif self.params == 'rf':
                from sklearn.ensemble import ExtraTreesClassifier
                base_model = ExtraTreesClassifier(n_estimators=100)
            else:
                raise ValueError('Invalid base model!')

            self.model = RFECV(base_model, cv=3, min_features_to_select=self.min_features)
            self.model.fit(X_new, y)

        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
