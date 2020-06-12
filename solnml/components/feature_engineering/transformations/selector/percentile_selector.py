from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, Constant
from solnml.components.feature_engineering.transformations.base_transformer import *


class PercentileSelector(Transformer):
    def __init__(self, percentile=50, score_func='chi2'):
        super().__init__("percentile_selector", 8)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.score_func = score_func
        self.percentile = percentile

    def get_score_func(self):
        if self.score_func == 'chi2':
            from sklearn.feature_selection import chi2
            call_func = chi2
        elif self.score_func == 'f_classif':
            from sklearn.feature_selection import f_classif
            call_func = f_classif
        elif self.score_func == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            call_func = mutual_info_classif
        else:
            raise ValueError("Unknown score function %s!" % str(self.score_func))
        return call_func

    def operate(self, input_datanode, target_fields=None):
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
            from sklearn.feature_selection import SelectPercentile
            self.model = SelectPercentile(self.get_score_func(), percentile=self.percentile)
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
        output_datanode.enable_balance = input_datanode.enable_balance
        output_datanode.data_balance = input_datanode.data_balance
        self.target_fields = target_fields.copy()

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            percentile = UniformFloatHyperparameter(
                name="percentile", lower=5, upper=99, default_value=50)

            score_func = CategoricalHyperparameter(
                name="score_func",
                choices=["chi2", "f_classif", "mutual_info"],
                default_value="chi2"
            )
            if dataset_properties is not None:
                # Chi2 can handle sparse data, so we respect this
                if 'sparse' in dataset_properties and dataset_properties['sparse']:
                    score_func = Constant(
                        name="score_func", value="chi2")

            cs = ConfigurationSpace()
            cs.add_hyperparameters([percentile, score_func])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'percentile': hp.uniform('percentile_percentile', 5, 99),
                     'score_func': hp.choice('percentile_score_func', ['chi2', 'f_classif', 'mutual_info'])}
            return space
