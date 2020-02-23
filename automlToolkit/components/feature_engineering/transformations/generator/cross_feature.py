from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from automlToolkit.components.feature_engineering.transformations.base_transformer import *


class CrossFeatureTransformation(Transformer):
    def __init__(self, random_seed=1):
        super().__init__("cross_features", 32)
        self.input_type = [CATEGORICAL]
        self.compound_mode = 'concatenate'

        self.output_type = CATEGORICAL
        self.features_ids = None
        self._model = None

    @ease_trans
    def operate(self, input_datanode, target_fields):
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import VarianceThreshold

        X, y = input_datanode.data
        X_new = X[:, target_fields]

        if not self.model:
            idxs = np.arange(X_new.shape[1])
            np.random.shuffle(idxs)
            self.features_ids = idxs[:200]

            self.model = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.model.fit(X_new[:, self.features_ids])

        print(X_new.shape)
        X_new = X_new[:, self.features_ids]
        _X = self.model.transform(X_new)
        if not self._model:
            self._model = VarianceThreshold()
            self._model.fit(_X)

        print(_X.shape)
        _X = self._model.transform(_X)
        print(_X.shape)
        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
