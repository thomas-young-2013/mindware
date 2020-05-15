from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.configspace_utils import check_for_bool


class PolynomialTransformation(Transformer):
    def __init__(self, degree=2, interaction_only='True', include_bias='False', random_state=1):
        super().__init__("polynomial", 17)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'concatenate'

        self.output_type = NUMERICAL
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields):
        from sklearn.preprocessing import PolynomialFeatures
        X, y = input_datanode.data
        X_new = X[:, target_fields]
        ori_length = X_new.shape[1]

        # Skip high-dimensional features.
        if X_new.shape[1] > 100 and self.degree == 2:
            return X_new.copy()

        if X_new.shape[1] > 15 and self.degree == 3:
            return X_new.copy()

        if not self.model:
            self.degree = int(self.degree)
            self.interaction_only = check_for_bool(self.interaction_only)
            self.include_bias = check_for_bool(self.include_bias)

            self.model = PolynomialFeatures(
                degree=self.degree, interaction_only=self.interaction_only,
                include_bias=self.include_bias)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        _X = _X[:, ori_length:]

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     ["False", "True"], "False")
        include_bias = UnParametrizedHyperparameter("include_bias", "False")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs
