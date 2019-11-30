import warnings
from components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition
from components.utils.configspace_utils import check_for_bool, check_none


class FastIcaDecomposer(Transformer):
    def __init__(self, algorithm='parallel', whiten='False', fun='logcosh', n_components=None,
                 random_state=None):
        super().__init__("fast_ica", 10)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components

        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        if self.model is None:
            import sklearn.decomposition

            self.whiten = check_for_bool(self.whiten)
            if check_none(self.n_components):
                n_components = None
            else:
                n_components = int(X.shape[1] * self.n_components)

            self.model = sklearn.decomposition.FastICA(
                n_components=n_components, algorithm=self.algorithm,
                fun=self.fun, whiten=self.whiten, random_state=self.random_state
            )
            # Make the RuntimeWarning an Exception!
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message='array must not contain infs or NaNs')
                try:
                    self.model.fit(X)
                except ValueError as e:
                    if 'array must not contain infs or NaNs' in e.args[0]:
                        raise ValueError("Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")

        X_new = self.model.transform(X)

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_components = UniformFloatHyperparameter(
            "n_components", 0.05, 1., q=0.05, default_value=0.3)
        algorithm = CategoricalHyperparameter('algorithm',
                                              ['parallel', 'deflation'], 'parallel')
        whiten = CategoricalHyperparameter('whiten',
                                           ['False', 'True'], 'False')
        fun = CategoricalHyperparameter(
            'fun', ['logcosh', 'exp', 'cube'], 'logcosh')
        cs.add_hyperparameters([n_components, algorithm, whiten, fun])

        cs.add_condition(EqualsCondition(n_components, whiten, "True"))

        return cs
