from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from components.feature_engineering.transformations.base_transformer import *


class GenericUnivariateSelector(Transformer):
    def __init__(self, score_func='chi2', alpha=0.5, mode='fpr'):
        super().__init__("generic_univariate_selector", 6)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'

        self.score_func = score_func
        self.alpha = alpha
        self.mode = mode
        if score_func == 'chi2':
            from sklearn.feature_selection import chi2
            self.call_func = chi2
        elif score_func == 'f_classif':
            from sklearn.feature_selection import f_classif
            self.call_func = f_classif
        else:
            raise ValueError("Unknown score function %s!" % str(score_func))

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
        if self.score_func == 'chi2':
            X_new[X_new < 0] = 0.0

        if self.model is None:
            self.model = GenericUnivariateSelect(
                score_func=self.call_func, param=self.alpha, mode=self.mode)
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
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1)

        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=["chi2", "f_classif"],
            default_value="chi2")
        if dataset_properties is not None:
            # Chi2 can handle sparse data, so we respect this
            if 'sparse' in dataset_properties and dataset_properties['sparse']:
                score_func = Constant(
                    name="score_func", value="chi2")

        mode = CategoricalHyperparameter('mode', ['fpr', 'fdr', 'fwe'], 'fpr')

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        return cs
