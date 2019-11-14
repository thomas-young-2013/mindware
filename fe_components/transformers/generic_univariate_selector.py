from fe_components.transformers.base_transformer import *


# TODO: Only for classification
class GenericUnivariateSelector(Transformer):
    def __init__(self, param='chi2', feature_left=0.5):
        '''
        :param param: estimator
        :param feature_left: int, or float in (0,1). Kbest for int and Percentile for float
        '''
        super().__init__("generic_univariate_selector", 5)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.params = param
        self.optional_params = ['chi2', 'f', 'mutual_info']
        if param == 'chi2':
            from sklearn.feature_selection import chi2
            self.call_param = chi2
        elif param == 'f':
            from sklearn.feature_selection import f_classif
            self.call_param = f_classif
        elif param == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            self.call_param = mutual_info_classif
        else:
            raise ValueError("Unknown score function %s!" % str(param))
        self.feature_left = feature_left

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import GenericUnivariateSelect

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.params == 'chi2':
            X_new[X_new < 0] = 0.0

        if self.model is None:
            if isinstance(self.feature_left, int):
                from sklearn.feature_selection import SelectKBest
                self.model = SelectKBest(self.call_param, self.feature_left)
            elif isinstance(self.feature_left, float):
                from sklearn.feature_selection import SelectPercentile
                self.model = SelectPercentile(self.call_param, percentile=int(self.feature_left * 100))
            self.model.fit(X_new, y)

        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)

        return output_datanode
