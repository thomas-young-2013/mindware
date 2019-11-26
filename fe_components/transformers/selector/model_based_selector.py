from fe_components.transformers.base_transformer import *


class ModelBasedSelector(Transformer):
    def __init__(self, param='et', max_features=None):
        super().__init__("model_based_selector", 7)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.params = param
        self.optional_params = ['et', 'svc']
        self.max_features = max_features

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import SelectFromModel

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        if self.model is None:
            if self.params == 'lr':
                from sklearn.linear_model import LogisticRegression
                base_model = LogisticRegression()
            elif self.params == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(n_estimators=100, max_depth=6)
            elif self.params == 'et':
                from sklearn.ensemble import ExtraTreesClassifier
                base_model = ExtraTreesClassifier(n_estimators=100, max_depth=6)
            elif self.params == 'svc':
                from sklearn.svm import LinearSVC
                base_model = LinearSVC(multi_class='ovr')
            else:
                raise ValueError('Invalid base model!')

            base_model.fit(X_new, y)
            self.model = SelectFromModel(base_model, prefit=True, threshold='mean')
            # coef = self.model.estimator_.coef_
        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)

        return output_datanode
