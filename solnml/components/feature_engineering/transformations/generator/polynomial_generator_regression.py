import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *
from solnml.components.utils.configspace_utils import check_for_bool


class PolynomialTransformation(Transformer):
    def __init__(self, degree=2, interaction_only='True', include_bias='False', random_state=1):
        super().__init__("polynomial_regression", 34)
        self.input_type = [DISCRETE, NUMERICAL]
        self.compound_mode = 'concatenate'
        self.best_idxs = list()
        if degree == 2:
            self.bestn = 25
        elif degree == 3:
            self.bestn = 10
        elif degree == 4:
            self.bestn = 6

        self.output_type = NUMERICAL
        self.degree = degree
        self.interaction_only = check_for_bool(interaction_only)
        self.include_bias = check_for_bool(include_bias)
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields):
        from sklearn.preprocessing import PolynomialFeatures
        from lightgbm import LGBMRegressor
        X, y = input_datanode.data

        if not self.best_idxs:
            lgb = LGBMRegressor(random_state=1)
            lgb.fit(X, y)
            _importance = lgb.feature_importances_
            idx_importance = np.argsort(-_importance)
            cur_idx = 0
            while len(self.best_idxs) < self.bestn and cur_idx < len(_importance):
                if idx_importance[cur_idx] in target_fields:
                    self.best_idxs.append(idx_importance[cur_idx])
                cur_idx += 1

        X_new = X[:, self.best_idxs]
        if not self.model:
            self.degree = int(self.degree)

            self.model = PolynomialFeatures(
                degree=self.degree, interaction_only=self.interaction_only,
                include_bias=self.include_bias)
            self.model.fit(X_new)

        _X = self.model.transform(X_new)
        _X = _X[:, X_new.shape[1]:]
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        degree = UniformIntegerHyperparameter("degree", lower=2, upper=3, default_value=2)
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     ["False", "True"], default_value="False")
        include_bias = UnParametrizedHyperparameter("include_bias", "False")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])
        return cs
