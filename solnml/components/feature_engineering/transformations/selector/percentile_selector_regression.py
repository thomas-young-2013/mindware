from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, Constant
from solnml.components.feature_engineering.transformations.base_transformer import *


class PercentileSelectorRegression(Transformer):
    def __init__(self, percentile=10, score_func='f_regression', random_state=1):
        super().__init__("percentile_selector_reg", 30)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'

        import sklearn.feature_selection
        self.random_state = random_state
        self.percentile = int(float(percentile))
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_regression
        else:
            raise ValueError("Don't know this scoring function: %s" % score_func)

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
        if self.model is None:
            from sklearn.feature_selection import SelectPercentile
            self.model = SelectPercentile(self.score_func, percentile=self.percentile)
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
        self.target_fields = target_fields.copy()

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            percentile = UniformFloatHyperparameter(
                "percentile", lower=5, upper=60, default_value=10, q=5)

            # score_func = CategoricalHyperparameter(
            #     name="score_func", choices=["f_regression", "mutual_info"], default_value='f_regression')
            score_func = CategoricalHyperparameter(
                name="score_func", choices=["f_regression"], default_value='f_regression')
            cs = ConfigurationSpace()
            cs.add_hyperparameters([percentile, score_func])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'percentile': hp.uniform('percentilereg_percentile', 5, 60),
                     'score_func': 'f_regression'}
            return space
