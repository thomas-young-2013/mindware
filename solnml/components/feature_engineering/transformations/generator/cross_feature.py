from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from solnml.components.feature_engineering.transformations.base_transformer import *


class CrossFeatureTransformation(Transformer):
    def __init__(self, random_state=1):
        super().__init__("cross_features", 32)
        self.input_type = [CATEGORICAL]
        self.compound_mode = 'concatenate'
        self.output_type = CATEGORICAL
        self.features_ids = None
        self._model = None

        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields):
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import VarianceThreshold

        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            idxs = np.arange(X_new.shape[1])
            np.random.seed(self.random_state)
            np.random.shuffle(idxs)
            self.features_ids = idxs[:200]

            self.model = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.model.fit(X_new[:, self.features_ids])

        print(X_new.shape)
        _X = self.model.transform(X_new[:, self.features_ids])
        print(_X.shape)
        if not self._model:
            self._model = VarianceThreshold()
            self._model.fit(_X)
        _X = self._model.transform(_X)
        print(_X.shape)
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        random_state = UniformIntegerHyperparameter("random_state", 1, 100000, default_value=1)
        cs.add_hyperparameter(random_state)
        return cs
